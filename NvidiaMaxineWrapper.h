/*###############################################################################
#
# Copyright (c) 2020 NVIDIA Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
###############################################################################*/
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <string>
#include <iostream>

#include "DDImage/PlanarIop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/NukeWrapper.h"

#include "nvVideoEffects.h"
#include "NvCVImage.h"           // NvCVImage utilities

#ifdef _MSC_VER
  #define strcasecmp _stricmp
  #include <Windows.h>
#else // !_MSC_VER
  #include <sys/stat.h>
#endif // _MSC_VER

#define BAIL_IF_ERR(err)                    do { if (0 != (err)) {                      goto bail; } } while(0)
#define BAIL_IF_NULL(x, err, code)          do { if ((void*)(x) == NULL)  { err = code; goto bail; } } while(0)
#define NVCV_ERR_HELP 411

#ifdef _WIN32
  #define DEFAULT_CODEC "avc1"
#else // !_WIN32
  #define DEFAULT_CODEC "H264"
#endif // _WIN32

using namespace DD::Image;

// Set this when using OTA Updates
// This path is used by nvVideoEffectsProxy.cpp to load the SDK dll
// when using  OTA Updates
char *g_nvVFXSDKPath = NULL;

static inline uint8_t f2u8(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    return (uint8_t)(v * 255.0f + 0.5f);
}

/**
 * Convert Nuke input -> NvCV RGBA8 (interleaved) in ONE pass.
 * - Expects 'dst' already allocated with allocRGBA8_CPU(dst, w, h).
 * - Writes with TOP-LEFT origin (what we want for NvCV/Maxine).
 * - Reads from Nuke with input->at(x, ny, Chan_*), flipping Y once.
 */
// Nuke input -> NvCV (interleaved), in one pass. Writes top-left origin (flip once).
static NvCV_Status inputToNvCV(Iop* input, NvCVImage* dst, bool expect_alpha = true)
{
    if (!input || !dst || !dst->pixels) return NVCV_ERR_PARAMETER;

    const ChannelSet ch = input->channels();
    const bool hasA = ch.contains(Chan_Alpha);

    const Format f = input->format();
    const int w = f.w(), h = f.h();

    const bool wantRGBA = (dst->pixelFormat == NVCV_RGBA);
    const bool wantBGR  = (dst->pixelFormat == NVCV_BGR);

    auto* base = static_cast<unsigned char*>(dst->pixels);
    const size_t pitch = dst->pitch;

    for (int y = 0; y < h; ++y) {
        const int ny = h - 1 - y;                  // flip Y once
        unsigned char* outRow = base + (size_t)y * pitch;

        for (int x = 0; x < w; ++x) {
            const float rf = ch.contains(Chan_Red)   ? input->at(x, ny, Chan_Red)   : 0.f;
            const float gf = ch.contains(Chan_Green) ? input->at(x, ny, Chan_Green) : 0.f;
            const float bf = ch.contains(Chan_Blue)  ? input->at(x, ny, Chan_Blue)  : 0.f;
            const float af = (hasA && expect_alpha)  ? input->at(x, ny, Chan_Alpha) : 1.f;

            if (wantRGBA) {
                const int di = x * 4;
                outRow[di + 0] = f2u8(rf);
                outRow[di + 1] = f2u8(gf);
                outRow[di + 2] = f2u8(bf);
                outRow[di + 3] = f2u8(af);
            } else if (wantBGR) {
                const int di = x * 3;
                outRow[di + 0] = f2u8(bf);  // B
                outRow[di + 1] = f2u8(gf);  // G
                outRow[di + 2] = f2u8(rf);  // R
            } else {
                return NVCV_ERR_PIXELFORMAT; // choose dst format to match your effect
            }
        }
    }
    return NVCV_SUCCESS;
}

static bool NvCVToImagePlane(const NvCVImage& src, DD::Image::ImagePlane& dst)
{
    using namespace DD::Image;

    // We support interleaved U8 RGBA and BGR outputs from your wrapper.
    const bool isRGBA8 = (src.pixelFormat == NVCV_RGBA && src.componentType == NVCV_U8 && src.planar == NVCV_INTERLEAVED);
    const bool isBGR8  = (src.pixelFormat == NVCV_BGR  && src.componentType == NVCV_U8 && src.planar == NVCV_INTERLEAVED);
    if (!isRGBA8 && !isBGR8 || !src.pixels) return false;

    const Box b = dst.bounds();
    const int W = b.w(), H = b.h();

    ChannelMask ch = dst.channels();
    Channel cr = ch.first();
    Channel cg = ch.next(cr);
    Channel cb = ch.next(cg);
    Channel ca = ch.next(cb);
    const bool hasA = (ca != Chan_Black);

    const int rNo = dst.chanNo(cr);
    const int gNo = dst.chanNo(cg);
    const int bNo = dst.chanNo(cb);
    const int aNo = hasA ? dst.chanNo(ca) : -1;

    const unsigned char* base = static_cast<const unsigned char*>(src.pixels);
    const size_t pitch = src.pitch;

    // Write once, flipping Y back for Nuke
    for (int y = 0; y < H; ++y) {
        const int ny = b.y() + (H - 1 - y);               // vertical flip
        const unsigned char* inRow = base + (size_t)y * pitch;

        for (int x = 0; x < W; ++x) {
            const int nx = b.x() + x;

            if (isRGBA8) {
                const int si = x * 4;
                dst.writableAt(nx, ny, rNo) = inRow[si + 0] / 255.0f;
                dst.writableAt(nx, ny, gNo) = inRow[si + 1] / 255.0f;
                dst.writableAt(nx, ny, bNo) = inRow[si + 2] / 255.0f;
                if (hasA) dst.writableAt(nx, ny, aNo) = inRow[si + 3] / 255.0f;
            } else { // BGR8
                const int si = x * 3;
                dst.writableAt(nx, ny, rNo) = inRow[si + 2] / 255.0f; // R from BGR
                dst.writableAt(nx, ny, gNo) = inRow[si + 1] / 255.0f; // G
                dst.writableAt(nx, ny, bNo) = inRow[si + 0] / 255.0f; // B
                if (hasA) dst.writableAt(nx, ny, aNo) = 1.0f;
            }
        }
    }
    return true;
}

struct FXApp {
    enum Err {
        errQuit               = +1,                         // Application errors
        errFlag               = +2,
        errRead               = +3,
        errWrite              = +4,
        errNone               = NVCV_SUCCESS,               // Video Effects SDK errors
        errGeneral            = NVCV_ERR_GENERAL,
        errUnimplemented      = NVCV_ERR_UNIMPLEMENTED,
        errMemory             = NVCV_ERR_MEMORY,
        errEffect             = NVCV_ERR_EFFECT,
        errSelector           = NVCV_ERR_SELECTOR,
        errBuffer             = NVCV_ERR_BUFFER,
        errParameter          = NVCV_ERR_PARAMETER,
        errMismatch           = NVCV_ERR_MISMATCH,
        errPixelFormat        = NVCV_ERR_PIXELFORMAT,
        errModel              = NVCV_ERR_MODEL,
        errLibrary            = NVCV_ERR_LIBRARY,
        errInitialization     = NVCV_ERR_INITIALIZATION,
        errFileNotFound       = NVCV_ERR_FILE,
        errFeatureNotFound    = NVCV_ERR_FEATURENOTFOUND,
        errMissingInput       = NVCV_ERR_MISSINGINPUT,
        errResolution         = NVCV_ERR_RESOLUTION,
        errUnsupportedGPU     = NVCV_ERR_UNSUPPORTEDGPU,
        errWrongGPU           = NVCV_ERR_WRONGGPU,
        errUnsupportedDriver  = NVCV_ERR_UNSUPPORTEDDRIVER,
        errCudaMemory         = NVCV_ERR_CUDA_MEMORY,       // CUDA errors
        errCudaValue          = NVCV_ERR_CUDA_VALUE,
        errCudaPitch          = NVCV_ERR_CUDA_PITCH,
        errCudaInit           = NVCV_ERR_CUDA_INIT,
        errCudaLaunch         = NVCV_ERR_CUDA_LAUNCH,
        errCudaKernel         = NVCV_ERR_CUDA_KERNEL,
        errCudaDriver         = NVCV_ERR_CUDA_DRIVER,
        errCudaUnsupported    = NVCV_ERR_CUDA_UNSUPPORTED,
        errCudaIllegalAddress = NVCV_ERR_CUDA_ILLEGAL_ADDRESS,
        errCuda               = NVCV_ERR_CUDA,
    };

    FXApp()   { _eff = nullptr; _effectName = nullptr; _inited = false; _progress = false; _enableEffect = true; }
    ~FXApp()  { NvVFX_DestroyEffect(_eff); }

    Err           createEffect(const char *effectSelector, const char *modelDir);
    void          destroyEffect();
    NvCV_Status   allocBuffers(unsigned width, unsigned height);
    NvCV_Status   allocTempBuffers();
    Err           processImage(Iop* input);
    Err           processKey(int key);
    Err           appErrFromVfxStatus(NvCV_Status status)  { return (Err)status; }
    const char*   errorStringFromCode(Err code);
    void          updateRes(int width,int height){src_width = width; src_height = height;}

    NvVFX_Handle  _eff;

    NvCVImage     _srcGpuBuf;
    NvCVImage     _dstGpuBuf;
    NvCVImage     _srcVFX;
    NvCVImage     _dstVFX;
    NvCVImage     _tmpVFX;  // We use the same temporary buffer for source and dst, since it auto-shapes as needed
    bool          _inited;
    bool          _progress;
    bool          _enableEffect;
    const char*   _effectName;
    float       FLAG_strength       = 0.f;
    int         FLAG_mode           = 0;
    int         FLAG_upscaleFactor     = 1;  
    int         src_width, src_height;
};

const char* FXApp::errorStringFromCode(Err code) {
  struct LutEntry { Err code; const char *str; };
  static const LutEntry lut[] = {
    { errRead,    "There was a problem reading a file"                    },
    { errWrite,   "There was a problem writing a file"                    },
    { errQuit,    "The user chose to quit the application"                },
    { errFlag,    "There was a problem with the command-line arguments"   },
  };
  if ((int)code <= 0) return NvCV_GetErrorStringFromCode((NvCV_Status)code);
  for (const LutEntry *p = lut; p != &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
    if (p->code == code) return p->str;
  return "UNKNOWN ERROR";
}

FXApp::Err FXApp::createEffect(const char *effectSelector, const char *modelDir) {
  NvCV_Status vfxErr;
  BAIL_IF_ERR(vfxErr = NvVFX_CreateEffect(effectSelector, &_eff));
  _effectName = effectSelector;
  // Do not set NVVFX_MODEL_DIRECTORY for NVVFX_FX_SR_UPSCALE feature as it is not a valid selector for that feature
  if (modelDir[0] != '\0' && strcmp(_effectName, NVVFX_FX_SR_UPSCALE)){
    BAIL_IF_ERR(vfxErr = NvVFX_SetString(_eff, NVVFX_MODEL_DIRECTORY, modelDir));
  }
bail:
  return appErrFromVfxStatus(vfxErr);
}

void FXApp::destroyEffect() {
  NvVFX_DestroyEffect(_eff);
  _eff = nullptr;
}

// Allocate one temp buffer to be used for input and output. Reshaping of the temp buffer in NvCVImage_Transfer() is done automatically,
// and is very low overhead. We expect the destination to be largest, so we allocate that first to minimize reallocs probablistically.
// Then we Realloc for the source to get the union of the two.
// This could alternately be done at runtime by feeding in an empty temp NvCVImage, but there are advantages to allocating all memory at load time.
NvCV_Status FXApp::allocTempBuffers() {
  NvCV_Status vfxErr;
  BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(  &_tmpVFX, _dstVFX.width, _dstVFX.height, _dstVFX.pixelFormat, _dstVFX.componentType, _dstVFX.planar, NVCV_GPU, 0));
  BAIL_IF_ERR(vfxErr = NvCVImage_Realloc(&_tmpVFX, _srcVFX.width, _srcVFX.height, _srcVFX.pixelFormat, _srcVFX.componentType, _srcVFX.planar, NVCV_GPU, 0));
bail:
  return vfxErr;
}

static NvCV_Status CheckScaleIsotropy(const NvCVImage *src, const NvCVImage *dst) {
  if (src->width * dst->height != src->height * dst->width) {
    printf("%ux%u --> %ux%u: different scale for width and height is not supported\n",
      src->width, src->height, dst->width, dst->height);
    return NVCV_ERR_RESOLUTION;
  }
  return NVCV_SUCCESS;
}

NvCV_Status FXApp::allocBuffers(unsigned /*width*/, unsigned /*height*/) {
    NvCV_Status vfxErr = NVCV_SUCCESS;
    if (_inited) return NVCV_SUCCESS;

    const int W = src_width;
    const int H = src_height;

    if (!strcmp(_effectName, NVVFX_FX_ARTIFACT_REDUCTION) ||
        !strcmp(_effectName, NVVFX_FX_SUPER_RES)) {

        // CPU staging: BGR U8 interleaved
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcVFX, W, H, NVCV_BGR,  NVCV_U8,  NVCV_INTERLEAVED, NVCV_CPU, 1));
        const int upW = (!strcmp(_effectName, NVVFX_FX_SUPER_RES)) ? W * FLAG_upscaleFactor : W;
        const int upH = (!strcmp(_effectName, NVVFX_FX_SUPER_RES)) ? H * FLAG_upscaleFactor : H;
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstVFX, upW, upH, NVCV_BGR,  NVCV_U8,  NVCV_INTERLEAVED, NVCV_CPU, 1));

        // GPU working: BGR F32 planar
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf, W,   H,   NVCV_BGR, NVCV_F32, NVCV_PLANAR,      NVCV_GPU, 1));
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, upW, upH, NVCV_BGR, NVCV_F32, NVCV_PLANAR,      NVCV_GPU, 1));
    }
    else if (!strcmp(_effectName, NVVFX_FX_SR_UPSCALE)) {
        // CPU staging: RGBA U8 interleaved
        const int upW = W * FLAG_upscaleFactor;
        const int upH = H * FLAG_upscaleFactor;
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcVFX, W,   H,   NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_CPU, 32));
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstVFX, upW, upH, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_CPU, 32));

        // GPU working: RGBA U8 interleaved (what Upscale wants)
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf, W,   H,   NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 32));
        BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, upW, upH, NVCV_RGBA, NVCV_U8, NVCV_INTERLEAVED, NVCV_GPU, 32));
        BAIL_IF_ERR(vfxErr = CheckScaleIsotropy(&_srcGpuBuf, &_dstGpuBuf));
    }

#ifndef ALLOC_TEMP_BUFFERS_AT_RUN_TIME
    // Now that _srcVFX/_dstVFX are valid, allocate one GPU temp that can reshape
    BAIL_IF_ERR(vfxErr = allocTempBuffers());
#endif

    _inited = true;
bail:
    return vfxErr;
}

FXApp::Err FXApp::processImage(Iop* input) {
    CUstream stream = 0;
    NvCV_Status vfxErr;

    if (!_eff) return errEffect;
    updateRes(input->format().w(),input->format().h());
    // Make sure sizes are set (call updateRes before)
    BAIL_IF_ERR(vfxErr = allocBuffers(src_width, src_height));
    // 1) Fill CPU staging (_srcVFX) in one pass, matching its pixelFormat
    BAIL_IF_ERR(vfxErr = inputToNvCV(input, &_srcVFX,
                      /*expect_alpha=*/ (_srcVFX.pixelFormat == NVCV_RGBA)));
    // 2) CPU->GPU (with conversion as needed)
    // - AR/SR: U8 interleaved BGR -> F32 planar BGR (scale 1/255.f)
    // - Upscale: U8 interleaved RGBA -> U8 interleaved RGBA (scale 1.f)
    const float uploadScale = (_srcGpuBuf.componentType == NVCV_F32) ? (1.f/255.f) : 1.f;
    BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_srcVFX, &_srcGpuBuf, uploadScale, stream, &_tmpVFX));
    // 3) Set images + params
    BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_INPUT_IMAGE,  &_srcGpuBuf));
    BAIL_IF_ERR(vfxErr = NvVFX_SetImage(_eff, NVVFX_OUTPUT_IMAGE, &_dstGpuBuf));
    BAIL_IF_ERR(vfxErr = NvVFX_SetCudaStream(_eff, NVVFX_CUDA_STREAM, stream));
    if (!strcmp(_effectName, NVVFX_FX_SUPER_RES) || !strcmp(_effectName, NVVFX_FX_ARTIFACT_REDUCTION)){
        BAIL_IF_ERR(vfxErr = NvVFX_SetU32(_eff, NVVFX_MODE, (unsigned)FLAG_mode));
        
    }
    if (!strcmp(_effectName, NVVFX_FX_SUPER_RES) || !strcmp(_effectName, NVVFX_FX_SR_UPSCALE)){
        BAIL_IF_ERR(vfxErr = NvVFX_SetF32(_eff, NVVFX_STRENGTH, FLAG_strength));
        
    }
    BAIL_IF_ERR(vfxErr = NvVFX_Load(_eff));
    BAIL_IF_ERR(vfxErr = NvVFX_Run(_eff, 0));

    // 4) GPU->CPU (with conversion as needed)
    const float downloadScale = (_dstGpuBuf.componentType == NVCV_F32) ? 255.f : 1.f;
    BAIL_IF_ERR(vfxErr = NvCVImage_Transfer(&_dstGpuBuf, &_dstVFX, downloadScale, stream, &_tmpVFX));

bail:
    return appErrFromVfxStatus(vfxErr);
}

