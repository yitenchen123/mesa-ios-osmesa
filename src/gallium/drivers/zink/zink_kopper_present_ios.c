/*
 * Zink unilateral present for iOS (Kopper-lite).
 *
 * Presents OSMesa frames straight to the app's CAMetalLayer through a
 * self-managed Vulkan swapchain, bypassing the CPU readback path.
 *
 * Design rules (blind-Vulkan safety):
 *  - Only runs when AMETHYST_KOPPER_PRESENT is set in the environment.
 *  - Every failure returns false and the caller falls back to the normal
 *    readback path. Nothing here may crash or hang the process.
 *  - Textures: 2D, single sample, format must match the swapchain format.
 *  - Single global swapchain (one window apps); guarded by a mutex.
 *  - Synchronous: drains the queue before blitting (correct first, fast
 *    later with semaphore chaining).
 */

#include "zink_screen.h"
#include "zink_format.h"
#include "zink_resource.h"
#include "zink_public.h"

#include "pipe/p_context.h"
#include "pipe/p_state.h"
#include "util/log.h"
#include "util/simple_mtx.h"
#include "util/u_memory.h"
#include <stdbool.h>
#include <stdlib.h>

/* This translation unit is only built for Apple (Metal) targets. The zink
 * meson source list includes it unconditionally, so on every other host it
 * degrades to the no-op stub at the bottom of this file. */
#if defined(__APPLE__) && defined(VK_USE_PLATFORM_METAL_EXT)

#include <objc/runtime.h>
#include <objc/message.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

/* MoltenVK's MVKSurface::getNaturalExtent calls -naturalDrawableSizeMVK,
 * which exists only on CAMetalLayer. Handing it a plain CALayer raises
 * NSInvalidArgumentException and kills the process, so reject anything that
 * does not look like a Metal drawable before creating the surface. */
static bool
ios_layer_is_metal_drawable(void *layer)
{
   if (!layer)
      return false;
   id obj = (id)layer;
   SEL sel = sel_registerName("naturalDrawableSizeMVK");
   if (![obj respondsToSelector:sel])
      return false;
   /* Require the class chain too, so a lookalike object with a stray
    * selector cannot slip through. */
   Class metal = objc_getClass("CAMetalLayer");
   return metal ? [obj isKindOfClass:metal] : true;
}

static simple_mtx_t present_lock = SIMPLE_MTX_INITIALIZER;
static void *present_layer;
static VkSurfaceKHR present_surface;
static VkSwapchainKHR present_swapchain;
static VkExtent2D present_extent;
static VkFormat present_format;
static VkImage present_images[8];
static uint32_t present_image_count;
static VkCommandPool present_pool;
/* Set when this surface/device cannot support kopper, so we stop rebuilding
 * the swapchain on every frame and log-spamming while the game runs. */
static bool present_unsupported;

static void
present_teardown(struct zink_screen *screen)
{
   if (present_swapchain) {
      VKSCR(DestroySwapchainKHR)(screen->dev, present_swapchain, NULL);
      present_swapchain = VK_NULL_HANDLE;
   }
   if (present_surface) {
      VKSCR(DestroySurfaceKHR)(screen->instance, present_surface, NULL);
      present_surface = VK_NULL_HANDLE;
   }
   if (present_pool) {
      VKSCR(DestroyCommandPool)(screen->dev, present_pool, NULL);
      present_pool = VK_NULL_HANDLE;
   }
   present_layer = NULL;
   present_image_count = 0;
}

bool
zink_kopper_present_ios(struct pipe_screen *pscreen, struct pipe_context *pctx,
                        struct pipe_resource *pres, unsigned w, unsigned h,
                        void *metal_layer)
{
   if (!getenv("AMETHYST_KOPPER_PRESENT"))
      return false;
   if (!pscreen || !pctx || !pres || !metal_layer || !w || !h)
      return false;
   /* A plain CALayer here means the app's view is not CAMetalLayer-backed
    * (or was created before the kopper switch was read). MoltenVK would
    * throw on it; fall back to the readback path instead. */
   if (present_unsupported)
      return false;
   if (!ios_layer_is_metal_drawable(metal_layer)) {
      static bool warned;
      if (!warned) {
         warned = true;
         mesa_loge("ZINK: kopper present: drawable %p is not CAMetalLayer-backed, "
                   "skipping kopper (falling back to readback)", metal_layer);
      }
      return false;
   }
   /* OSMesa front buffers are RECT textures. */
   if ((pres->target != PIPE_TEXTURE_2D && pres->target != PIPE_TEXTURE_RECT) ||
       pres->nr_samples > 1)
      return false;

   struct zink_screen *screen = zink_screen(pscreen);
   struct zink_resource *res = zink_resource(pres);
   if (!res || !res->obj || !res->obj->image)
      return false;

   /* Source must be blittable 8-bit RGBA in either byte order. */
   VkFormat src_format = zink_get_format(screen, pres->format);
   if (src_format != VK_FORMAT_R8G8B8A8_UNORM &&
       src_format != VK_FORMAT_B8G8R8A8_UNORM)
      return false;

   simple_mtx_lock(&present_lock);

   /* (Re)create surface + swapchain when needed. */
   if (metal_layer != present_layer) {
      present_teardown(screen);
      present_layer = metal_layer;

      VkMetalSurfaceCreateInfoEXT sci = {
         .sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
         .pNext = NULL,
         .flags = 0,
         .pLayer = (const CAMetalLayer *)metal_layer,
      };
      if (VKSCR(CreateMetalSurfaceEXT)(screen->instance, &sci, NULL, &present_surface) != VK_SUCCESS ||
          !present_surface) {
         mesa_loge("ZINK: kopper present: CreateMetalSurface failed");
         present_layer = NULL;
         simple_mtx_unlock(&present_lock);
         return false;
      }

      VkBool32 supported = VK_FALSE;
      VKSCR(GetPhysicalDeviceSurfaceSupportKHR)(screen->pdev, screen->gfx_queue, present_surface, &supported);
      if (!supported) {
         mesa_loge("ZINK: kopper present: queue lacks surface support");
         present_teardown(screen);
         simple_mtx_unlock(&present_lock);
         return false;
      }
      mesa_loge("ZINK: kopper present: Metal surface ready");
   }

   VkSurfaceCapabilitiesKHR caps = {0};
   if (VKSCR(GetPhysicalDeviceSurfaceCapabilitiesKHR)(screen->pdev, present_surface, &caps) != VK_SUCCESS) {
      mesa_loge("ZINK: kopper present: surface caps query failed");
      present_teardown(screen);
      simple_mtx_unlock(&present_lock);
      return false;
   }

   uint32_t fmt_count = 0;
   VKSCR(GetPhysicalDeviceSurfaceFormatsKHR)(screen->pdev, present_surface, &fmt_count, NULL);
   VkSurfaceFormatKHR *fmts = malloc(sizeof(*fmts) * (fmt_count ? fmt_count : 1));
   if (!fmts || !fmt_count) {
      free(fmts);
      mesa_loge("ZINK: kopper present: no surface formats");
      present_teardown(screen);
      simple_mtx_unlock(&present_lock);
      return false;
   }
   VKSCR(GetPhysicalDeviceSurfaceFormatsKHR)(screen->pdev, present_surface, &fmt_count, fmts);
   /* MoltenVK surfaces usually report B8G8R8A8_UNORM (Metal BGRA8Unorm)
    * rather than R8G8B8A8_UNORM, so accept either byte order. Prefer the one
    * matching the source format, which keeps the blit a straight copy. */
   VkFormat swap_format = VK_FORMAT_UNDEFINED;
   for (uint32_t i = 0; i < fmt_count; i++) {
      if (fmts[i].format != VK_FORMAT_R8G8B8A8_UNORM &&
          fmts[i].format != VK_FORMAT_B8G8R8A8_UNORM)
         continue;
      if (fmts[i].format == src_format) {
         swap_format = fmts[i].format;
         break;
      }
      if (swap_format == VK_FORMAT_UNDEFINED)
         swap_format = fmts[i].format;
   }
   if (swap_format == VK_FORMAT_UNDEFINED) {
      /* Log the offered list once so the real capability set is visible in
       * the device log rather than guessed at. */
      static bool warned_fmt;
      if (!warned_fmt) {
         warned_fmt = true;
         mesa_loge("ZINK: kopper present: no RGB8/BGR8 swapchain format "
                   "(src fmt=%d, %u offered):", src_format, fmt_count);
         for (uint32_t i = 0; i < fmt_count && i < 16; i++)
            mesa_loge("ZINK: kopper present:   offered[%u] = %d (%s)",
                      i, fmts[i].format,
                      fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ?
                      "SRGB_NONLINEAR" : "other");
         mesa_loge("ZINK: kopper present: disabling kopper, using readback");
      }
      free(fmts);
      /* Keep the surface (it is valid) and latch the failure so we do not
       * rebuild it every frame. */
      present_unsupported = true;
      simple_mtx_unlock(&present_lock);
      return false;
   }
   free(fmts);
   if (swap_format != src_format)
      mesa_loge("ZINK: kopper present: src fmt=%d swap fmt=%d (byte swizzle)",
                src_format, swap_format);

   VkExtent2D extent;
   if (caps.currentExtent.width != 0xFFFFFFFFu) {
      extent = caps.currentExtent;
   } else {
      extent.width = w > caps.maxImageExtent.width ? caps.maxImageExtent.width : (w < caps.minImageExtent.width ? caps.minImageExtent.width : w);
      extent.height = h > caps.maxImageExtent.height ? caps.maxImageExtent.height : (h < caps.minImageExtent.height ? caps.minImageExtent.height : h);
   }
   if (extent.width == 0 || extent.height == 0) {
      simple_mtx_unlock(&present_lock);
      return false;
   }

   uint32_t mode_count = 0;
   VKSCR(GetPhysicalDeviceSurfacePresentModesKHR)(screen->pdev, present_surface, &mode_count, NULL);
   VkPresentModeKHR *modes = malloc(sizeof(*modes) * (mode_count ? mode_count : 1));
   VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
   if (modes && mode_count) {
      VKSCR(GetPhysicalDeviceSurfacePresentModesKHR)(screen->pdev, present_surface, &mode_count, modes);
      for (uint32_t i = 0; i < mode_count; i++) {
         if (modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
         }
      }
   }
   free(modes);

   if (!present_swapchain || present_format != swap_format ||
       present_extent.width != extent.width || present_extent.height != extent.height) {
      VkSwapchainKHR old = present_swapchain;
      present_swapchain = VK_NULL_HANDLE;

      uint32_t image_count = caps.minImageCount + 1;
      if (caps.maxImageCount > 0 && image_count > caps.maxImageCount)
         image_count = caps.maxImageCount;

      VkSwapchainCreateInfoKHR sci = {
         .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
         .pNext = NULL,
         .flags = 0,
         .surface = present_surface,
         .minImageCount = image_count,
         .imageFormat = swap_format,
         .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
         .imageExtent = extent,
         .imageArrayLayers = 1,
         .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
         .queueFamilyIndexCount = 0,
         .pQueueFamilyIndices = NULL,
         .preTransform = caps.currentTransform,
         .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
         .presentMode = present_mode,
         .clipped = VK_TRUE,
         .oldSwapchain = old,
      };
      VkResult res_create = VKSCR(CreateSwapchainKHR)(screen->dev, &sci, NULL, &present_swapchain);
      if (old)
         VKSCR(DestroySwapchainKHR)(screen->dev, old, NULL);
      if (res_create != VK_SUCCESS || !present_swapchain) {
         mesa_loge("ZINK: kopper present: CreateSwapchain failed (%d)", res_create);
         present_teardown(screen);
         simple_mtx_unlock(&present_lock);
         return false;
      }
      present_extent = extent;
      present_format = swap_format;
      present_image_count = 0;
      uint32_t n = 8;
      if (VKSCR(GetSwapchainImagesKHR)(screen->dev, present_swapchain, &n, NULL) == VK_SUCCESS && n <= 8) {
         present_image_count = n;
         VKSCR(GetSwapchainImagesKHR)(screen->dev, present_swapchain, &n, present_images);
      }
      if (!present_image_count) {
         mesa_loge("ZINK: kopper present: no swapchain images");
         present_teardown(screen);
         simple_mtx_unlock(&present_lock);
         return false;
      }
      mesa_loge("ZINK: kopper present: swapchain %ux%u fmt=%d mode=%d images=%u",
                extent.width, extent.height, swap_format, present_mode, present_image_count);
   }

   /* Drain Zink work, then blit + present synchronously (v1: correct first). */
   pctx->flush(pctx, NULL, 0);
   VKSCR(QueueWaitIdle)(screen->queue);

   if (!present_pool) {
      VkCommandPoolCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .pNext = NULL,
         .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
         .queueFamilyIndex = screen->gfx_queue,
      };
      if (VKSCR(CreateCommandPool)(screen->dev, &pci, NULL, &present_pool) != VK_SUCCESS) {
         mesa_loge("ZINK: kopper present: pool failed");
         simple_mtx_unlock(&present_lock);
         return false;
      }
   }

   uint32_t image_index = 0;
   VkResult acq = VKSCR(AcquireNextImageKHR)(screen->dev, present_swapchain, 1000000000ull,
                                                        VK_NULL_HANDLE, VK_NULL_HANDLE, &image_index);
   if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
      present_teardown(screen);
      simple_mtx_unlock(&present_lock);
      return false;
   }
   if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
      mesa_loge("ZINK: kopper present: acquire failed (%d)", acq);
      simple_mtx_unlock(&present_lock);
      return false;
   }
   if (image_index >= present_image_count) {
      simple_mtx_unlock(&present_lock);
      return false;
   }

   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = NULL,
      .commandPool = present_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd = VK_NULL_HANDLE;
   if (VKSCR(AllocateCommandBuffers)(screen->dev, &cai, &cmd) != VK_SUCCESS) {
      simple_mtx_unlock(&present_lock);
      return false;
   }

   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = NULL,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      .pInheritanceInfo = NULL,
   };
   bool ok = false;
   VkImageLayout orig_layout = res->layout;
   if (VKSCR(BeginCommandBuffer)(cmd, &bi) == VK_SUCCESS) {
      VkImageMemoryBarrier barriers[2] = {0};
      barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[0].srcAccessMask = 0;
      barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barriers[0].oldLayout = orig_layout;
      barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[0].image = res->obj->image;
      barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barriers[0].subresourceRange.baseMipLevel = 0;
      barriers[0].subresourceRange.levelCount = 1;
      barriers[0].subresourceRange.baseArrayLayer = 0;
      barriers[0].subresourceRange.layerCount = 1;
      barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[1].srcAccessMask = 0;
      barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[1].image = present_images[image_index];
      barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barriers[1].subresourceRange.baseMipLevel = 0;
      barriers[1].subresourceRange.levelCount = 1;
      barriers[1].subresourceRange.baseArrayLayer = 0;
      barriers[1].subresourceRange.layerCount = 1;
      VKSCR(CmdPipelineBarrier)(cmd,
         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
         0, NULL, 0, NULL, 2, barriers);

      VkImageBlit blit = {0};
      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel = 0;
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount = 1;
      blit.srcOffsets[1].x = w;
      blit.srcOffsets[1].y = h;
      blit.srcOffsets[1].z = 1;
      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel = 0;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount = 1;
      blit.dstOffsets[1].x = present_extent.width;
      blit.dstOffsets[1].y = present_extent.height;
      blit.dstOffsets[1].z = 1;
      VKSCR(CmdBlitImage)(cmd,
         res->obj->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         present_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         1, &blit, VK_FILTER_LINEAR);

      barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barriers[0].dstAccessMask = 0;
      barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barriers[0].newLayout = orig_layout;
      barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barriers[1].dstAccessMask = 0;
      barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barriers[1].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      VKSCR(CmdPipelineBarrier)(cmd,
         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
         0, NULL, 0, NULL, 2, barriers);

      if (VKSCR(EndCommandBuffer)(cmd) == VK_SUCCESS) {
         VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = NULL,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = NULL,
            .pWaitDstStageMask = NULL,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = NULL,
         };
         if (VKSCR(QueueSubmit)(screen->queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS &&
             VKSCR(QueueWaitIdle)(screen->queue) == VK_SUCCESS) {
            VkPresentInfoKHR pi = {
               .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
               .pNext = NULL,
               .waitSemaphoreCount = 0,
               .pWaitSemaphores = NULL,
               .swapchainCount = 1,
               .pSwapchains = &present_swapchain,
               .pImageIndices = &image_index,
               .pResults = NULL,
            };
            VkResult pres = VKSCR(QueuePresentKHR)(screen->queue, &pi);
            ok = (pres == VK_SUCCESS || pres == VK_SUBOPTIMAL_KHR);
            if (pres == VK_ERROR_OUT_OF_DATE_KHR)
               present_teardown(screen);
            else if (!ok)
               mesa_loge("ZINK: kopper present: present failed (%d)", pres);
            else {
               static bool logged;
               if (!logged) {
                  logged = true;
                  mesa_loge("ZINK: kopper present: first frame presented");
               }
            }
         }
      }
      res->layout = orig_layout;
   }
   VKSCR(FreeCommandBuffers)(screen->dev, present_pool, 1, &cmd);
   simple_mtx_unlock(&present_lock);
   return ok;
}

#else /* !(__APPLE__ && VK_USE_PLATFORM_METAL_EXT) */

/* Non-Apple hosts: kopper unilateral present is unavailable. The caller
 * always falls back to the OSMesa readback path when this returns false. */
bool
zink_kopper_present_ios(struct pipe_screen *pscreen, struct pipe_context *pctx,
                        struct pipe_resource *pres, unsigned w, unsigned h,
                        void *metal_layer)
{
   (void)pscreen; (void)pctx; (void)pres; (void)w; (void)h; (void)metal_layer;
   return false;
}

#endif /* __APPLE__ && VK_USE_PLATFORM_METAL_EXT */
