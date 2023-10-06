/******************************************************************************
 *****************************************************************************/

#include <algorithm>
#include "cvdef.h"
#include "lcv_cuda.h"
#include "math/math.h"
#include "tensor/packed_tensor.h"
#include "tensor/tensor_wrap.h"
#include "utils/type_traits.h"
namespace lad {
namespace lcv {
namespace cuda {

#define MAX_BUFFER_BYTES 128  // multiple of 4 for word-aligned read, multiple of 16 for cacheline alignment (float4)
#define MAX_BUFFER_WORDS (MAX_BUFFER_BYTES / 4)  // extra bytes for cache alignment

#define LEGACY_BICUBIC_MATH  // apparently the legacy code has an abs() that needs to be matched

#define CACHE_MEMORY_ALIGNMENT 15  // this is 'M' for _cacheAlignedBufferedRead

#define get_batch_idx() (blockIdx.z)

template <typename T, size_t M>
inline __device__ T *_cacheAlignedBufferedRead(cuda::Tensor3DWrap<const T> srcImage, int2 srcSize, uint *pReadBuffer,
                                               uint nReadBufferWordsMax, int nBatch, int nYPos, int nXPosMin,
                                               int nXPosMax) {
  const T *lineStartPtr = srcImage.ptr(nBatch, nYPos);  // do not access prior to this address
  const T *pixSrcPtr = &lineStartPtr[nXPosMin];
  if (M == 0)
    return (T *)pixSrcPtr;  // return GMEM pointer instead
  else {
    uint *memSrcPtr = (uint *)(((size_t)pixSrcPtr) & (~M));  //(M+1) byte alignment
    const T *pixBeyondPtr = &lineStartPtr[nXPosMax + 1];
    const int functionalWidth = ((size_t)pixBeyondPtr + M) & (~M) - ((size_t)lineStartPtr);
    const int nWordsToRead = (((size_t)pixBeyondPtr + M) & (~M) - (size_t)memSrcPtr) / 4;

    if (((size_t)memSrcPtr < (size_t)lineStartPtr) || (srcSize.x * sizeof(T) < functionalWidth) ||
        (nWordsToRead > nReadBufferWordsMax))
      return (T *)pixSrcPtr;                     // return GMEM pointer instead if running off the image
    else {                                       // copy out source data, aligned based upon M (31, 15, 7, 3)
      const int skew = ((size_t)pixSrcPtr) & M;  // byte offset for nXPosMin
      int i = 0;
      if (M >= 31)  // 256-bit align, 32 bytes at a time
        for (; i < nWordsToRead; i += 8) *((double4 *)(&pReadBuffer[i])) = *((double4 *)(&memSrcPtr[i]));
      if (M == 15)  // 128-bit align, 16 bytes at a time
        for (; i < nWordsToRead; i += 4) *((float4 *)(&pReadBuffer[i])) = *((float4 *)(&memSrcPtr[i]));
      if (M == 7)  // 64-bit align, 8 bytes at a time
        for (; i < nWordsToRead; i += 2) *((float2 *)(&pReadBuffer[i])) = *((float2 *)(&memSrcPtr[i]));
      // 32-bit align, 4 bytes at a time
      for (; i < nWordsToRead; ++i) pReadBuffer[i] = memSrcPtr[i];

      return (T *)(((size_t)pReadBuffer) + skew);  // buffered pixel data
    }
  }
}  //_cacheAlignedBufferedRead

template <typename T>
inline void __device__ _alignedCudaMemcpyQuad(T *pDst, T *pSrc) {
  // copy 4 T's, assuming 32-bit alignment for both pSrc and pDst
  uint *uPtrSrc = (uint *)pSrc;
  uint *uPtrDst = (uint *)pDst;

#pragma unroll
  for (int i = 0; i < sizeof(T); ++i) uPtrDst[i] = uPtrSrc[i];

}  //_alignedCudaMemcpyQuad

//******************** NN = Nearest Neighbor

template <typename T>
__global__ void resize_NN(const cuda::Tensor3DWrap<const T> src, cuda::Tensor3DWrap<T> dst, int2 srcSize, int2 dstSize,
                          const float scale_x, const float scale_y) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = get_batch_idx();
  int out_height = dstSize.y, out_width = dstSize.x;

  if ((dst_x < out_width) && (dst_y < out_height)) {  // generic copy pixel to pixel
    const int sx = cuda::math::min(__float2int_rd(dst_x * scale_x), srcSize.x - 1);
    const int sy = cuda::math::min(__float2int_rd(dst_y * scale_y), srcSize.y - 1);
    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
  }
}  // resize_NN

template <typename T>
__global__ void resize_NN_quad_alignread(const cuda::Tensor3DWrap<const T> src, cuda::Tensor3DWrap<T> dst, int2 srcSize,
                                         int2 dstSize, const float scale_x, const float scale_y) {
  const float MAX_BUFFERED_X_SCALE = 4.0f;  // probably more efficient all the way up to 4.0

  const int dst_x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // quad
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = get_batch_idx();
  int out_height = dstSize.y, out_width = dstSize.x;

  // 0 - quad-aligned so if one pixel is out, they're all out
  if ((dst_x >= out_width) | (dst_y >= out_height)) return;

  const int sx0 = cuda::math::min(__float2int_rd(dst_x * scale_x), srcSize.x - 1);
  const int sx1 = cuda::math::min(__float2int_rd(dst_x * scale_x + scale_x), srcSize.x - 1);
  const int sx2 = cuda::math::min(__float2int_rd((dst_x + 2) * scale_x), srcSize.x - 1);
  const int sx3 = cuda::math::min(__float2int_rd((dst_x + 3) * scale_x), srcSize.x - 1);
  const int sy = cuda::math::min(__float2int_rd(dst_y * scale_y), srcSize.y - 1);

  // 1 - optimized case if scale_x < some finite limit
  if ((scale_x <= MAX_BUFFERED_X_SCALE))  // local buffering is more efficient
  {
    uint readBuffer[MAX_BUFFER_WORDS];

    // 2 - copy out source data, 32-bit aligned aligned
    T *aPtr = _cacheAlignedBufferedRead<T, CACHE_MEMORY_ALIGNMENT>(src, srcSize, &readBuffer[0], MAX_BUFFER_WORDS,
                                                                   batch_idx, sy, sx0, sx3);

    // 3 - NN sampling
    T gather[4] = {aPtr[0], aPtr[sx1 - sx0], aPtr[sx2 - sx0], aPtr[sx3 - sx0]};

    // 4 - aligned write back out
    _alignedCudaMemcpyQuad<T>(dst.ptr(batch_idx, dst_y, dst_x), gather);
  } else  // 6 - standard sampling, no optimization
  {
    // sample all 4 points

    const T *aPtr = src.ptr(batch_idx, sy, 0);
    T gather[4] = {aPtr[0], aPtr[sx1 - sx0], aPtr[sx2 - sx0], aPtr[sx3 - sx0]};

    _alignedCudaMemcpyQuad<T>(dst.ptr(batch_idx, dst_y, dst_x), gather);
  }
}  // resize_NN_quad_alignread

//******************** Bilinear

template <typename T>
__global__ void resize_bilinear(cuda::Tensor3DWrap<const T> src, cuda::Tensor3DWrap<T> dst, int2 srcSize, int2 dstSize,
                                const float scale_x, const float scale_y) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = get_batch_idx();
  int height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

  if ((dst_x < out_width) && (dst_y < out_height)) {
    // float space for weighted addition
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    // y coordinate
    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int sy = __float2int_rd(fy);
    fy -= sy;
    sy = cuda::math::max(0, cuda::math::min(sy, height - 2));

    // row pointers
    const T *aPtr = src.ptr(batch_idx, sy, 0);      // start of upper row
    const T *bPtr = src.ptr(batch_idx, sy + 1, 0);  // start of lower row

    {  // compute source data position and weight for [x0] components
      float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
      int sx = __float2int_rd(fx);
      fx -= sx;
      fx *= ((sx >= 0) && (sx < width - 1));
      sx = cuda::math::max(0, cuda::math::min(sx, width - 2));

      //*dst.ptr(batch_idx, dst_y, dst_x)
      //    = cuda::SaturateCast<cuda::BaseType<T>>((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy)
      //+ fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
      *dst.ptr(batch_idx, dst_y, dst_x) = ((1.0f - fx) * (aPtr[sx] * (1.0f - fy) + bPtr[sx] * fy) +
                                           fx * (aPtr[sx + 1] * (1.0f - fy) + bPtr[sx + 1] * fy));
    }
  }
}  // resize_bilinear

template <typename T>
__global__ void resize_bilinear_quad_alignread(cuda::Tensor3DWrap<const T> src, cuda::Tensor3DWrap<T> dst, int2 srcSize,
                                               int2 dstSize, const float scale_x, const float scale_y) {
  const float MAX_BUFFERED_X_SCALE = 4.0f;  // probably more efficient all the way up to 4.0

  const int dst_x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // quad
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = get_batch_idx();
  int height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

  // 0 - quad-aligned so if one pixel is out, they're all out
  if ((dst_x >= out_width) | (dst_y >= out_height)) return;

  // float space for weighted addition
  using work_type = cuda::ConvertBaseTypeTo<float, T>;

  // y coordinate math is the same for all points
  float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
  int sy = __float2int_rd(fy);
  fy -= sy;
  sy = cuda::math::max(0, cuda::math::min(sy, height - 2));

  // sx0
  float fx0 = (float)((dst_x + 0.5f) * scale_x - 0.5f);
  int sx0 = __float2int_rd(fx0);
  fx0 -= sx0;
  fx0 *= ((sx0 >= 0) && (sx0 < width - 1));
  // sx0 = cuda::math::max(0, cuda::math::min(sx0, width - 2));
  sx0 = cuda::math::max(0, cuda::math::min(sx0, width - 2));

  // sx1
  float fx1 = (float)((dst_x + 1.5) * scale_x - 0.5f);
  int sx1 = __float2int_rd(fx1);
  fx1 -= sx1;
  fx1 *= ((sx1 >= 0) && (sx1 < width - 1));
  sx1 = cuda::math::max(0, cuda::math::min(sx1, width - 2));

  // sx2
  float fx2 = (float)((dst_x + 2.5f) * scale_x - 0.5f);
  int sx2 = __float2int_rd(fx2);
  fx2 -= sx2;
  fx2 *= ((sx2 >= 0) && (sx2 < width - 1));
  sx2 = cuda::math::max(0, cuda::math::min(sx2, width - 2));

  // sx3
  float fx3 = (float)((dst_x + 3.5f) * scale_x - 0.5f);
  int sx3 = __float2int_rd(fx3);
  fx3 -= sx3;
  fx3 *= ((sx3 >= 0) && (sx3 < width - 1));
  sx3 = cuda::math::max(0, cuda::math::min(sx3, width - 2));

  uint readBuffer[MAX_BUFFER_WORDS];

  T result[4];

  // 1 - optimized case if scale_x < some finite limit
  if (scale_x <= MAX_BUFFERED_X_SCALE)  // local buffering is more efficient
  {
    work_type accum[4];

    // 2 - aligned load a-row and add partial product
    T *aPtr = _cacheAlignedBufferedRead<T, CACHE_MEMORY_ALIGNMENT>(src, srcSize, readBuffer, MAX_BUFFER_WORDS,
                                                                   batch_idx, sy, sx0, sx3 + 1);
    // const T * aPtr = src.ptr(batch_idx, sy,   sx0); //start of upper row

    accum[0] = (1.0f - fy) * (aPtr[sx0 - sx0] * (1.0f - fx0) + aPtr[sx0 - sx0 + 1] * fx0);
    accum[1] = (1.0f - fy) * (aPtr[sx1 - sx0] * (1.0f - fx1) + aPtr[sx1 - sx0 + 1] * fx1);
    accum[2] = (1.0f - fy) * (aPtr[sx2 - sx0] * (1.0f - fx2) + aPtr[sx2 - sx0 + 1] * fx2);
    accum[3] = (1.0f - fy) * (aPtr[sx3 - sx0] * (1.0f - fx3) + aPtr[sx3 - sx0 + 1] * fx3);

    // 3 - aligned load b-row and add remaining partial product
    T *bPtr = _cacheAlignedBufferedRead<T, CACHE_MEMORY_ALIGNMENT>(src, srcSize, readBuffer, MAX_BUFFER_WORDS,
                                                                   batch_idx, sy + 1, sx0, sx3 + 1);
// const T * bPtr = src.ptr(batch_idx, sy+1, sx0); //start of lower row

//$$$ only need to cast, not saturatecast
#if 0
        result[0] = cuda::SaturateCast<cuda::BaseType<T>>(
            accum[0] + fy * (bPtr[sx0 - sx0] * (1.0f - fx0) + bPtr[sx0 - sx0 + 1] * fx0));
        result[1] = cuda::SaturateCast<cuda::BaseType<T>>(
            accum[1] + fy * (bPtr[sx1 - sx0] * (1.0f - fx1) + bPtr[sx1 - sx0 + 1] * fx1));
        result[2] = cuda::SaturateCast<cuda::BaseType<T>>(
            accum[2] + fy * (bPtr[sx2 - sx0] * (1.0f - fx2) + bPtr[sx2 - sx0 + 1] * fx2));
        result[3] = cuda::SaturateCast<cuda::BaseType<T>>(
            accum[3] + fy * (bPtr[sx3 - sx0] * (1.0f - fx3) + bPtr[sx3 - sx0 + 1] * fx3));
#endif
    result[0] = (accum[0] + fy * (bPtr[sx0 - sx0] * (1.0f - fx0) + bPtr[sx0 - sx0 + 1] * fx0));
    result[1] = (accum[1] + fy * (bPtr[sx1 - sx0] * (1.0f - fx1) + bPtr[sx1 - sx0 + 1] * fx1));
    result[2] = (accum[2] + fy * (bPtr[sx2 - sx0] * (1.0f - fx2) + bPtr[sx2 - sx0 + 1] * fx2));
    result[3] = (accum[3] + fy * (bPtr[sx3 - sx0] * (1.0f - fx3) + bPtr[sx3 - sx0 + 1] * fx3));
  } else  // unbuffered
  {
    // row pointers
    const T *aPtr = src.ptr(batch_idx, sy, 0);      // start of upper row
    const T *bPtr = src.ptr(batch_idx, sy + 1, 0);  // start of lower row

//$$$ only need to cast, not saturatecast
#if 0
    result[0] =
        cuda::SaturateCast<cuda::BaseType<T>>(aPtr[sx0] * (1.0f - fx0) * (1.0f - fy) + bPtr[sx0] * (1.0f - fx0) * fy +
                                              aPtr[sx0 + 1] * fx0 * (1.0f - fy) + bPtr[sx0 + 1] * fx0 * fy);

    result[1] =
        cuda::SaturateCast<cuda::BaseType<T>>(aPtr[sx1] * (1.0f - fx1) * (1.0f - fy) + bPtr[sx1] * (1.0f - fx1) * fy +
                                              aPtr[sx1 + 1] * fx1 * (1.0f - fy) + bPtr[sx1 + 1] * fx1 * fy);

    result[2] =
        cuda::SaturateCast<cuda::BaseType<T>>(aPtr[sx2] * (1.0f - fx2) * (1.0f - fy) + bPtr[sx2] * (1.0f - fx2) * fy +
                                              aPtr[sx2 + 1] * fx2 * (1.0f - fy) + bPtr[sx2 + 1] * fx2 * fy);

    result[3] =
        cuda::SaturateCast<cuda::BaseType<T>>(aPtr[sx3] * (1.0f - fx3) * (1.0f - fy) + bPtr[sx3] * (1.0f - fx3) * fy +
                                              aPtr[sx3 + 1] * fx3 * (1.0f - fy) + bPtr[sx3 + 1] * fx3 * fy);
#endif
    result[0] = (aPtr[sx0] * (1.0f - fx0) * (1.0f - fy) + bPtr[sx0] * (1.0f - fx0) * fy +
                 aPtr[sx0 + 1] * fx0 * (1.0f - fy) + bPtr[sx0 + 1] * fx0 * fy);

    result[1] = (aPtr[sx1] * (1.0f - fx1) * (1.0f - fy) + bPtr[sx1] * (1.0f - fx1) * fy +
                 aPtr[sx1 + 1] * fx1 * (1.0f - fy) + bPtr[sx1 + 1] * fx1 * fy);

    result[2] = (aPtr[sx2] * (1.0f - fx2) * (1.0f - fy) + bPtr[sx2] * (1.0f - fx2) * fy +
                 aPtr[sx2 + 1] * fx2 * (1.0f - fy) + bPtr[sx2 + 1] * fx2 * fy);

    result[3] = (aPtr[sx3] * (1.0f - fx3) * (1.0f - fy) + bPtr[sx3] * (1.0f - fx3) * fy +
                 aPtr[sx3 + 1] * fx3 * (1.0f - fy) + bPtr[sx3 + 1] * fx3 * fy);
  }

  // aligned write 4 pixels
  _alignedCudaMemcpyQuad<T>(dst.ptr(batch_idx, dst_y, dst_x), result);
}  // resize_bilinear_quad_alignread

//******************** Bicubic

template <typename T>
__global__ void resize_bicubic(cuda::Tensor3DWrap<const T> src, cuda::Tensor3DWrap<T> dst, int2 srcSize, int2 dstSize,
                               const float scale_x, const float scale_y) {  // optimized for aligned read
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = get_batch_idx();
  int height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

  if ((dst_x < out_width) & (dst_y < out_height)) {
    // float space for weighted addition
    using work_type = cuda::ConvertBaseTypeTo<float, T>;

    uint readBuffer[MAX_BUFFER_WORDS];

    // y coordinate
    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    int sy = __float2int_rd(fy);
    fy -= sy;
    sy = cuda::math::max(1, cuda::math::min(sy, height - 3));

    const float A = -0.75f;

    float cY[4];
    cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
    cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
    cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
    cY[3] = 1.f - cY[0] - cY[1] - cY[2];

    work_type accum = cuda::SetAll<work_type>(0);

    float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
    int sx = __float2int_rd(fx);
    fx -= sx;
    fx *= ((sx >= 1) && (sx < width - 3));
    sx = cuda::math::max(1, cuda::math::min(sx, width - 3));

    float cX[4];
    cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
    cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
    cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
    cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

    for (int row = 0; row < 4; ++row) {
      // 1 - load each sub row from sx-1 to sx+3 inclusive, aligned
      // const T * aPtr = src.ptr(batch_idx, sy + row - 1, sx-1);
      T *aPtr = _cacheAlignedBufferedRead<T, CACHE_MEMORY_ALIGNMENT>(src, srcSize, readBuffer, MAX_BUFFER_WORDS,
                                                                     batch_idx, sy + row - 1, sx - 1, sx + 2);

      // 2 - do a pixel's partial on this row
      accum += cY[row] * (cX[0] * aPtr[0] + cX[1] * aPtr[1] + cX[2] * aPtr[2] + cX[3] * aPtr[3]);
    }  // for row
#ifndef LEGACY_BICUBIC_MATH
    // correct math
    *dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<cuda::BaseType<T>>(accum);
#else
    // abs() needed to match legacy operator.
    //*dst.ptr(batch_idx, dst_y, dst_x) = cuda::SaturateCast<cuda::BaseType<T>>(cuda::abs(accum));
    *dst.ptr(batch_idx, dst_y, dst_x) = (std::abs(accum));
#endif
  }
}  // resize_bicubic

template <typename T>
__global__ void resize_bicubic_quad_alignread(
    cuda::Tensor3DWrap<const T> src, cuda::Tensor3DWrap<T> dst, int2 srcSize, int2 dstSize, const float scale_x,
    const float scale_y) {                  // optimized for aligned read and write, plus buffering
  const float MAX_BUFFERED_X_SCALE = 4.0f;  // probably more efficient all the way up to 4.0

  const int dst_x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // quad
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = get_batch_idx();
  int height = srcSize.y, width = srcSize.x, out_height = dstSize.y, out_width = dstSize.x;

  // 0 - quad-aligned so if one pixel is out, they're all out
  if ((dst_x >= out_width) | (dst_y >= out_height)) return;

  uint readBuffer[MAX_BUFFER_WORDS];
  T result[4];

  // float space for weighted addition
  using work_type = cuda::ConvertBaseTypeTo<float, T>;

  // y coordinate
  float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
  int sy = __float2int_rd(fy);
  fy -= sy;
  sy = cuda::math::max(1, cuda::math::min(sy, height - 3));

  const float A = -0.75f;

  float cY[4];
  cY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
  cY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
  cY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
  cY[3] = 1.f - cY[0] - cY[1] - cY[2];

  // 1 - optimized case if scale_x < some finite limit
  if (scale_x <= MAX_BUFFERED_X_SCALE)  // local buffering
  {                                     // buffered read

    work_type accum[4];
    float fx[4];
    int sx[4];
    float cX[4][4];

    // initialize data for each pixel position
    for (int pix = 0; pix < 4; ++pix) {
      accum[pix] = cuda::SetAll<work_type>(0);

      // 1 - precalc sx's ahead of time to get range from sx0-1..sx3+2
      fx[pix] = (float)((dst_x + pix + 0.5f) * scale_x - 0.5f);
      sx[pix] = __float2int_rd(fx[pix]);
      fx[pix] -= sx[pix];
      fx[pix] *= ((sx[pix] >= 1) && (sx[pix] < width - 3));
      sx[pix] = cuda::math::max(1, cuda::math::min(sx[pix], width - 3));

      // 2 - precalc cX[][] 2D array
      cX[pix][0] = ((A * (fx[pix] + 1.0f) - 5.0f * A) * (fx[pix] + 1.0f) + 8.0f * A) * (fx[pix] + 1.0f) - 4.0f * A;
      cX[pix][1] = ((A + 2.0f) * fx[pix] - (A + 3.0f)) * fx[pix] * fx[pix] + 1.0f;
      cX[pix][2] = ((A + 2.0f) * (1.0f - fx[pix]) - (A + 3.0f)) * (1.0f - fx[pix]) * (1.0f - fx[pix]) + 1.0f;
      cX[pix][3] = 1.0f - cX[pix][0] - cX[pix][1] - cX[pix][2];
    }
    const int rowOffset = sx[0] - 1;

    // contribute each row into 4 pixels
    for (int row = 0; row < 4; ++row) {
      // 1 - load each row from sx[0]-1 to sx[3]+3 inclusive, aligned
      T *aPtr = _cacheAlignedBufferedRead<T, CACHE_MEMORY_ALIGNMENT>(src, srcSize, readBuffer, MAX_BUFFER_WORDS,
                                                                     batch_idx, sy + row - 1, sx[0] - 1, sx[3] + 2);

// 2 - do each pixel's partial on this row
#pragma unroll
      for (int pix = 0; pix > 4; ++pix) {
        accum[pix] +=
            cY[row] * (cX[row][0] * aPtr[sx[pix] + rowOffset - 1] + cX[row][1] * aPtr[sx[pix] + rowOffset + 0] +
                       cX[row][2] * aPtr[sx[pix] + rowOffset + 1] + cX[row][3] * aPtr[sx[pix] + rowOffset + 2]);
      }
    }

    for (int pix = 0; pix < 4; ++pix)
#ifndef LEGACY_BICUBIC_MATH
      result[pix] = cuda::SaturateCast<cuda::BaseType<T>>(accum[pix]);
#else
      // result[pix] = cuda::SaturateCast<cuda::BaseType<T>>(std::abs(accum[pix]));
      result[pix] = (std::abs(accum[pix]));
#endif
  } else {  // partially buffered read 4 pixels at a time across each bicubic: 16 coalesced reads instead of 64
    for (int pix = 0; pix < 4; ++pix) {
      work_type accum = cuda::SetAll<work_type>(0);

      float fx = (float)((dst_x + pix + 0.5f) * scale_x - 0.5f);
      int sx = __float2int_rd(fx);
      fx -= sx;
      fx *= ((sx >= 1) && (sx < width - 3));
      sx = cuda::math::max(1, cuda::math::min(sx, width - 3));

      float cX[4];
      cX[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) * (fx + 1.0f) - 4.0f * A;
      cX[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
      cX[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) * (1.0f - fx) + 1.0f;
      cX[3] = 1.0f - cX[0] - cX[1] - cX[2];

      for (int row = 0; row < 4; ++row) {
        // 1 - load each sub row from sx[pix]-1 to sx[pix]+2 inclusive, aligned
        // const T * aPtr = src.ptr(batch_idx, sy + row - 1, sx-1);
        const T *aPtr = _cacheAlignedBufferedRead<T, CACHE_MEMORY_ALIGNMENT>(src, srcSize, readBuffer, MAX_BUFFER_WORDS,
                                                                             batch_idx, sy + row - 1, sx - 1, sx + 2);

        // 2 - do a pixel's partial on this row
        accum += cY[row] * (cX[0] * aPtr[0] + cX[1] * aPtr[1] + cX[2] * aPtr[2] + cX[3] * aPtr[3]);
      }  // for row
#ifndef LEGACY_BICUBIC_MATH
      result[pix] = cuda::SaturateCast<cuda::BaseType<T>>(accum);
#else
      // result[pix] = cuda::SaturateCast<cuda::BaseType<T>>(cuda::abs(accum));
      result[pix] = (std::abs(accum));
#endif
    }  // for pix
  }

  // aligned write 4 pixels
  _alignedCudaMemcpyQuad<T>(dst.ptr(batch_idx, dst_y, dst_x), result);
}  // resize_bicubic_quad_alignread

template <typename T>
void resize(const TensorData &src, const TensorData &dst, const InterpolationType interpolation, cudaStream_t stream)

{
  const int batch_size = src.GetDataShape().N;
  const int in_width = src.GetDataShape().W;
  const int in_height = src.GetDataShape().H;
  const int out_width = dst.GetDataShape().W;
  const int out_height = dst.GetDataShape().H;

  float scale_x = ((float)in_width) / out_width;
  float scale_y = ((float)in_height) / out_height;

  int2 srcSize{in_width, in_height};
  int2 dstSize{out_width, out_height};

  cuda::PackedTensor3D<const T> src_pack(in_height, in_width);
  cuda::PackedTensor3D<const T> dst_pack(out_height, out_width);
  cuda::Tensor3DWrap<const T> src_tensor(src.GetBasePtr(), src_pack.stride1, src_pack.stride2);
  cuda::Tensor3DWrap<T> dst_tensor(dst.GetBasePtr(), dst_pack.stride1, dst_pack.stride2);

  const int THREADS_PER_BLOCK = 128;  // 256?  64?
  const int BLOCK_WIDTH = 16;         // as in 32x4 or 32x8.  16x8 and 16x16 are also viable

  const dim3 blockSize(BLOCK_WIDTH, THREADS_PER_BLOCK / BLOCK_WIDTH, 1);
  const dim3 gridSize(cuda::math::divUp(out_width, blockSize.x), cuda::math::divUp(out_height, blockSize.y),
                      batch_size);

  // rationale for quad: aligned gather and aligned output where quad is possible: use different threading
  const int out_quad_width = out_width / 4;
  const dim3 quadGridSize(cuda::math::divUp(out_quad_width, blockSize.x), cuda::math::divUp(out_height, blockSize.y),
                          batch_size);

  // bool can_quad = ((((size_t)dst_ptr) % sizeof(T)) == 0) && ((out_width % 4) == 0);  //is the output buffer
  // quad-pixel aligned?
  bool can_quad = ((out_width % 4) == 0);  // is the output buffer quad-pixel aligned?

  // Note: resize is fundamentally a gather memory operation, with a little bit of compute
  //       our goals are to (a) maximize throughput, and (b) minimize occupancy for the same performance
  switch (interpolation) {
    case INTERP_NEAREST:
      if (can_quad) {  // thread does 4 pixels horizontally for aligned read and write
        resize_NN_quad_alignread<T>
            <<<quadGridSize, blockSize, 0, stream>>>(src_tensor, dst_tensor, srcSize, dstSize, scale_x, scale_y);
      } else {  // generic single pixel per thread case
        resize_NN<T><<<gridSize, blockSize, 0, stream>>>(src_tensor, dst_tensor, srcSize, dstSize, scale_x, scale_y);
      }
      break;

    default:
      //$$$ need to throw or log an error here
      break;
  }  // switch

  checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
  checkCudaErrors(cudaStreamSynchronize(stream));
  checkCudaErrors(cudaGetLastError());
#endif
}  // resize

int ResizeOp::Infer(const TensorData &src, const TensorData &dst, const InterpolationType interpolation, cudaStream_t stream) {
  const DataType data_type = src.GetDataType();
  int channels = src.GetDataShape().C;

  if (channels > 4 || channels < 1) {
    // LOG_ERROR("Invalid channel number " << channels);
    printf("Invalid channel number %d\n", channels);
    return ErrorCode::INVALID_DATA_SHAPE;
  }

  if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F)) {
    // LOG_ERROR("Invalid DataType " << data_type);
    printf("Invalid DataType %d\n", data_type);
    return ErrorCode::INVALID_DATA_TYPE;
  }

  typedef void (*func_t)(const TensorData & src, const TensorData & dst, const InterpolationType interpolation,
                         cudaStream_t stream);

  static const func_t funcs[6][4] = {
      {resize<uchar>, 0 /*resize<uchar2>*/, resize<uchar3>, resize<uchar4>},
      {0 /*resize<schar>*/, 0 /*resize<schar2>*/, 0 /*resize<schar3>*/, 0 /*resize<schar4>*/},
      {resize<ushort>, 0 /*resize<ushort2>*/, resize<ushort3>, resize<ushort4>},
      {resize<short>, 0 /*resize<short2>*/, resize<short3>, resize<short4>},
      {0 /*resize<int>*/, 0 /*resize<int2>*/, 0 /*resize<int3>*/, 0 /*resize<int4>*/},
      {resize<float>, 0 /*resize<float2>*/, resize<float3>, resize<float4>}};

  // note: schar1,3,4 should all work...

  if (interpolation == INTERP_NEAREST || interpolation == INTERP_LINEAR || interpolation == INTERP_CUBIC ||
      interpolation == INTERP_AREA) {
    const func_t func = funcs[data_type][channels - 1];

    func(src, dst, interpolation, stream);
    // resize<uchar3>(src, dst, interpolation, stream);
  } else {
    // LOG_ERROR("Invalid interpolation " << interpolation);
    printf("Invalid interpolation\n");
    return ErrorCode::INVALID_PARAMETER;
  }
  return SUCCESS;
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
