/*
 * EGL iOS platform (CAMetalLayer windows) for Zink+Kopper.
 *
 * Modeled on platform_wayland.c / platform_surfaceless.c, minus everything
 * iOS doesn't have (no wl_display connection, no DRM fd, no gralloc, no
 * event loop, no dmabuf). The app passes its CAMetalLayer* as the EGL
 * native window; sizes are re-read from the layer on every GetDrawableInfo
 * so rotation/resize take effect without caching.
 *
 * Display discovery: none (no WL/DRM probing). The gallium driver is fixed
 * to "zink"; dri2_detect_swrast_kopper() flips the kopper bit from the
 * driver name, and all rendering/present goes through the kopper WSI path.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_metal.h>
#include "util/format/u_formats.h"
#include "main/glconfig.h"
#include "pipe/p_screen.h"
#include "egl_dri2.h"
#include "eglglobals.h"
#include "kopper_interface.h"
#include "loader.h"
#include "loader_dri_helper.h"
#include "dri_screen.h"
#include "dri_util.h"
#include "dri_util.h"

struct dri2_ios_surface {
   struct dri2_egl_surface base;
   /* CAMetalLayer* (opaque here - UIKit types stay on the app side). */
   void *metal_layer;
   /* Last known good size, returned when the layer momentarily reports 0
    * (e.g. backgrounded) so kopper never sees a 0x0 drawable. */
   int last_w;
   int last_h;
};

/* CGSize is two doubles on 64-bit; declared locally to avoid pulling
 * CoreGraphics headers into a plain C file. Layout-identical, so the
 * arm64 register return convention matches. */
typedef struct {
   double width;
   double height;
} ios_layer_size;

static void
ios_layer_size_get(void *layer, int *w, int *h)
{
   *w = *h = 0;
   if (!layer)
      return;
   SEL szSel = sel_registerName("drawableSize");
   ios_layer_size (*szFn)(id, SEL) = (void *)objc_msgSend;
   ios_layer_size sz = szFn((id)layer, szSel);
   if (sz.width > 0 && sz.height > 0) {
      *w = (int)sz.width;
      *h = (int)sz.height;
   }
}

static void
dri2_ios_kopper_get_drawable_info(struct dri_drawable *draw, int *w,
                                  int *h, void *loaderPrivate)
{
   struct dri2_egl_surface *dri2_surf = loaderPrivate;
   struct dri2_ios_surface *ios_surf = (struct dri2_ios_surface *)dri2_surf;
   int lw = 0, lh = 0;

   ios_layer_size_get(ios_surf->metal_layer, &lw, &lh);
   if (lw > 0 && lh > 0) {
      ios_surf->last_w = lw;
      ios_surf->last_h = lh;
   }
   *w = ios_surf->last_w;
   *h = ios_surf->last_h;
}

static_assert(sizeof(struct kopper_vk_surface_create_storage) >=
                 sizeof(VkMetalSurfaceCreateInfoEXT),
              "");

static void
kopperSetSurfaceCreateInfo(void *_draw, struct kopper_loader_info *out)
{
   struct dri2_egl_surface *dri2_surf = _draw;
   struct dri2_ios_surface *ios_surf = (struct dri2_ios_surface *)dri2_surf;
   VkMetalSurfaceCreateInfoEXT *metal =
      (VkMetalSurfaceCreateInfoEXT *)&out->bos;

   metal->sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
   metal->pNext = NULL;
   metal->flags = 0;
   /* Forward the app's layer straight to MoltenVK. Lifetime belongs to
    * the app: the layer must outlive the EGL surface. */
   metal->pLayer = (const CAMetalLayer *)ios_surf->metal_layer;
   out->present_opaque = dri2_surf->base.PresentOpaque;
   out->compression = 0;
}

static const __DRIkopperLoaderExtension kopper_loader_extension = {
   .base = {__DRI_KOPPER_LOADER, 1},

   .SetSurfaceCreateInfo = kopperSetSurfaceCreateInfo,
   .GetDrawableInfo = dri2_ios_kopper_get_drawable_info,
};

static const __DRIextension *ios_kopper_loader_extensions[] = {
   &kopper_loader_extension.base,
   &image_lookup_extension.base,
   NULL,
};

static _EGLSurface *
dri2_ios_create_window_surface(_EGLDisplay *disp, _EGLConfig *conf,
                               void *native_window, const EGLint *attrib_list)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   struct dri2_egl_config *dri2_conf = dri2_egl_config(conf);
   struct dri2_ios_surface *ios_surf;
   struct dri2_egl_surface *dri2_surf;
   const struct dri_config *config;

   if (!native_window) {
      _eglError(EGL_BAD_NATIVE_WINDOW, "dri2_create_surface");
      return NULL;
   }

   ios_surf = calloc(1, sizeof *ios_surf);
   if (!ios_surf) {
      _eglError(EGL_BAD_ALLOC, "dri2_create_surface");
      return NULL;
   }
   dri2_surf = &ios_surf->base;
   ios_surf->metal_layer = native_window;

   if (!dri2_init_surface(&dri2_surf->base, disp, EGL_WINDOW_BIT, conf,
                         attrib_list, false, native_window))
      goto cleanup_surf;

   config = dri2_get_dri_config(dri2_conf, EGL_WINDOW_BIT,
                                ios_surf->base.base.GLColorspace);
   if (!config) {
      _eglError(EGL_BAD_MATCH,
                "Unsupported surfacetype/colorspace configuration");
      goto cleanup_surf;
   }

   /* Initial size; refreshed from the layer on every GetDrawableInfo. */
   ios_layer_size_get(ios_surf->metal_layer,
                      &ios_surf->last_w, &ios_surf->last_h);
   ios_surf->base.base.Width = ios_surf->last_w;
   ios_surf->base.base.Height = ios_surf->last_h;

   if (!dri2_create_drawable(dri2_dpy, config, dri2_surf, ios_surf)) {
      _eglError(EGL_BAD_ALLOC, "dri2_create_surface");
      goto cleanup_surf;
   }

   return &dri2_surf->base;

cleanup_surf:
   free(ios_surf);
   return NULL;
}

static EGLBoolean
dri2_ios_destroy_surface(_EGLDisplay *disp, _EGLSurface *surf)
{
   struct dri2_ios_surface *ios_surf = (struct dri2_ios_surface *)surf;
   struct dri2_egl_surface *dri2_surf = &ios_surf->base;

   (void)disp;
   if (dri2_surf->dri_drawable)
      driDestroyDrawable(dri2_surf->dri_drawable);
   dri2_fini_surface(&dri2_surf->base);
   free(ios_surf);

   return EGL_TRUE;
}

static EGLBoolean
dri2_ios_kopper_swap_buffers(_EGLDisplay *disp, _EGLSurface *draw)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(draw);

   (void)disp;
   kopperSwapBuffers(dri2_surf->dri_drawable,
                     __DRI2_FLUSH_CONTEXT | __DRI2_FLUSH_INVALIDATE_ANCILLARY);
   return EGL_TRUE;
}

static EGLBoolean
dri2_ios_kopper_swap_buffers_with_damage(_EGLDisplay *disp, _EGLSurface *draw,
                                         const EGLint *rects, EGLint n_rects)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(draw);

   (void)disp;
   kopperSwapBuffersWithDamage(dri2_surf->dri_drawable,
                               __DRI2_FLUSH_CONTEXT | __DRI2_FLUSH_INVALIDATE_ANCILLARY,
                               n_rects, rects);
   return EGL_TRUE;
}

static EGLint
dri2_ios_kopper_query_buffer_age(_EGLDisplay *disp, _EGLSurface *surface)
{
   struct dri2_egl_surface *dri2_surf = dri2_egl_surface(surface);

   (void)disp;
   return kopperQueryBufferAge(dri2_surf->dri_drawable);
}

static const struct dri2_egl_display_vtbl dri2_ios_kopper_display_vtbl = {
   .authenticate = NULL,
   .create_window_surface = dri2_ios_create_window_surface,
   .destroy_surface = dri2_ios_destroy_surface,
   .create_image = dri2_create_image_khr,
   .swap_buffers = dri2_ios_kopper_swap_buffers,
   .swap_buffers_with_damage = dri2_ios_kopper_swap_buffers_with_damage,
   .get_dri_drawable = dri2_surface_get_dri_drawable,
   .query_buffer_age = dri2_ios_kopper_query_buffer_age,
};

/* iOS visuals: plain RGB(A)8888 + RGB565, no WL/DRM format tables. */
static const struct {
   int visual_id;
   enum pipe_format pipe_format;
} dri2_ios_visuals[] = {
   /* BGRA8 first: this becomes the EGLConfig the app receives, and kopper
    * builds the swapchain's VkFormat from it. MoltenVK maps that to a
    * CAMetalLayer pixel format, and iOS refuses MTLPixelFormatRGBA8Unorm
    * (110) with 'CAMetalLayerInvalid' - Apple's supported layout is
    * BGRA8Unorm. RGBA stays as a secondary visual. */
   { 1, PIPE_FORMAT_B8G8R8A8_UNORM },
   { 2, PIPE_FORMAT_R8G8B8A8_UNORM },
   { 3, PIPE_FORMAT_RGBX8888_UNORM },
   { 4, PIPE_FORMAT_B5G6R5_UNORM },
};

static void
dri2_ios_add_configs_for_visuals(_EGLDisplay *disp)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   unsigned int i, j;

   /* Try to create an EGLConfig for every config the driver declares,
    * for every native visual (RGBA first so choosers prefer it).
    *
    * EGL_WINDOW_BIT|EGL_PBUFFER_BIT: eglChooseConfig requires the requested
    * surface type to be a subset of the config's, so advertising window-only
    * made every request that also wanted a pbuffer (the launcher's, and
    * zink's offscreen FBO work) match nothing and fail. These configs back
    * both kinds of surface on iOS. */
   for (i = 0; i < ARRAY_SIZE(dri2_ios_visuals); i++) {
      for (j = 0; dri2_dpy->driver_configs[j]; j++) {
         struct dri2_egl_config *dri2_conf;
         const struct gl_config *gl_config =
            (const struct gl_config *)dri2_dpy->driver_configs[j];
         EGLint attr_list[] = {
            EGL_NATIVE_VISUAL_ID, dri2_ios_visuals[i].visual_id,
            EGL_NONE,
         };

         /* Only BGRA8 drawables are usable on iOS: the colour format chosen
          * here flows down to the swapchain's VkFormat and then to
          * CAMetalLayer.pixelFormat, and CAMetalLayer refuses
          * MTLPixelFormatRGBA8Unorm (110) with 'CAMetalLayerInvalid'.
          * Filtering here stops eglChooseConfig from handing out a config
          * that cannot be presented. */
         if (gl_config->color_format != PIPE_FORMAT_B8G8R8A8_UNORM &&
             gl_config->color_format != PIPE_FORMAT_B8G8R8X8_UNORM)
            continue;

         dri2_conf = dri2_add_config(disp, dri2_dpy->driver_configs[j],
                                     EGL_WINDOW_BIT | EGL_PBUFFER_BIT,
                                     attr_list);
         if (!dri2_conf)
            continue;
      }
   }
}

EGLBoolean
dri2_initialize_ios(_EGLDisplay *disp)
{
   struct dri2_egl_display *dri2_dpy = dri2_egl_display(disp);
   const char *err;

   /* No DRM/gralloc on iOS: fixed gallium driver, fd-less screen. */
   dri2_dpy->driver_name = strdup("zink");
   if (!dri2_dpy->driver_name) {
      err = "DRI2: out of memory";
      goto cleanup;
   }
   dri2_dpy->loader_extensions = ios_kopper_loader_extensions;
   dri2_dpy->fd_render_gpu = -1;
   dri2_dpy->fd_display_gpu = -1;

   /* Flips dri2_dpy->kopper from the "zink" driver name. */
   dri2_detect_swrast_kopper(disp);

   if (!dri2_create_screen(disp)) {
      err = "DRI2: failed to create screen";
      goto cleanup;
   }

   dri2_setup_screen(disp);

   dri2_ios_add_configs_for_visuals(disp);

   disp->Extensions.EXT_buffer_age = EGL_TRUE;
   disp->Extensions.EXT_swap_buffers_with_damage = EGL_TRUE;
   disp->Extensions.EXT_present_opaque = EGL_TRUE;

   /* Fill vtbl last to prevent accidentally calling virtual function during
    * initialization.
    */
   dri2_dpy->vtbl = &dri2_ios_kopper_display_vtbl;

   return EGL_TRUE;

cleanup:
   _eglError(EGL_NOT_INITIALIZED, err);
   return EGL_FALSE;
}
