/******************************************************************************
 *****************************************************************************/

#include <cfloat>
#include "lcv_cuda.h"
#include "tensor/tensor_wrap.h"
// #include "log.h"
#include "math/math.h"
#include "math/saturate.h"
#include "tensor/packed_tensor.h"
#include "utils/type_traits.h"

static constexpr int ITUR_BT_601_SHIFT = 20;
static constexpr int ITUR_BT_601_CVR = 1673527;
static constexpr int ITUR_BT_601_CY = 1220542;
static constexpr int ITUR_BT_601_CVG = -852492;
static constexpr int ITUR_BT_601_CUG = -409993;
static constexpr int ITUR_BT_601_CUB = 2116026;
#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

#define BLOCK 32

namespace lad {
namespace lcv {
namespace cuda {

__device__ __forceinline__ void yuv42xxp_to_bgr_kernel(const int &Y, const int &U, const int &V, uchar &r, uchar &g,
                                                       uchar &b) {
  //  R = 1.164(Y - 16) + 1.596(V - 128)
  //  G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
  //  B = 1.164(Y - 16)                  + 2.018(U - 128)

  // R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
  // G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
  // B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20
  const int C0 = ITUR_BT_601_CY, C1 = ITUR_BT_601_CVR, C2 = ITUR_BT_601_CVG, C3 = ITUR_BT_601_CUG, C4 = ITUR_BT_601_CUB;
  const int yuv4xx_shift = ITUR_BT_601_SHIFT;

  int yy = cuda::math::max(0, Y - 16) * C0;
  int uu = U - 128;
  int vv = V - 128;

  r = cuda::saturate_cast<uchar>(CV_DESCALE((yy + C1 * vv), yuv4xx_shift));
  g = cuda::saturate_cast<uchar>(CV_DESCALE((yy + C2 * vv + C3 * uu), yuv4xx_shift));
  b = cuda::saturate_cast<uchar>(CV_DESCALE((yy + C4 * uu), yuv4xx_shift));
}

template <class T>
__global__ void yuv420sp_to_bgr_char_nhwc(cuda::Tensor4DWrap<T> src, cuda::Tensor4DWrap<T> dst, int2 dstSize, int dcn,
                                          int bidx, int uidx) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (dst_x >= dstSize.x || dst_y >= dstSize.y) return;
  const int batch_idx = get_batch_idx();

  int uv_x = (dst_x % 2 == 0) ? dst_x : (dst_x - 1);

  T Y = *src.ptr(batch_idx, dst_y, dst_x, 0);
  T U = *src.ptr(batch_idx, dstSize.y + dst_y / 2, uv_x + uidx);
  T V = *src.ptr(batch_idx, dstSize.y + dst_y / 2, uv_x + 1 - uidx);

  uchar r{0}, g{0}, b{0}, a{0xff};
  yuv42xxp_to_bgr_kernel(int(Y), int(U), int(V), r, g, b);

  *dst.ptr(batch_idx, dst_y, dst_x, bidx) = b;
  *dst.ptr(batch_idx, dst_y, dst_x, 1) = g;
  *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;

  if (dcn == 4) {
    *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
  }
}

template <class T>
__global__ void yuv420p_to_bgr_char_nhwc(cuda::Tensor4DWrap<T> src, cuda::Tensor4DWrap<T> dst, int2 dstSize, int dcn,
                                         int bidx, int uidx) {
  int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (dst_x >= dstSize.x || dst_y >= dstSize.y) return;

  const int batch_idx = get_batch_idx();
  int plane_y_step = dstSize.y * dstSize.x;
  int plane_uv_step = plane_y_step / 4;
  int uv_x = (dst_y % 4 < 2) ? dst_x / 2 : (dst_x / 2 + dstSize.x / 2);

  T Y = *src.ptr(batch_idx, dst_y, dst_x, 0);
  T U = *src.ptr(batch_idx, dstSize.y + dst_y / 4, uv_x + plane_uv_step * uidx);
  T V = *src.ptr(batch_idx, dstSize.y + dst_y / 4, uv_x + plane_uv_step * (1 - uidx));

  uchar r{0}, g{0}, b{0}, a{0xff};
  yuv42xxp_to_bgr_kernel(int(Y), int(U), int(V), r, g, b);

  *dst.ptr(batch_idx, dst_y, dst_x, bidx) = b;
  *dst.ptr(batch_idx, dst_y, dst_x, 1) = g;
  *dst.ptr(batch_idx, dst_y, dst_x, bidx ^ 2) = r;
  if (dcn == 4) {
    *dst.ptr(batch_idx, dst_y, dst_x, 3) = a;
  }
}

inline ErrorCode YUV420xp_to_BGR(const TensorData &src, const TensorData &dst, const ColorCVTCode code,
                                 cudaStream_t stream) {
  int bidx = (code == COLOR_YUV2BGR_NV12 || code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2BGR_NV21 ||
              code == COLOR_YUV2BGRA_NV21 || code == COLOR_YUV2BGR_YV12 || code == COLOR_YUV2BGRA_YV12 ||
              code == COLOR_YUV2BGR_IYUV || code == COLOR_YUV2BGRA_IYUV)
                 ? 0
                 : 2;

  int uidx = (code == COLOR_YUV2BGR_NV12 || code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2RGB_NV12 ||
              code == COLOR_YUV2RGBA_NV12 || code == COLOR_YUV2BGR_IYUV || code == COLOR_YUV2BGRA_IYUV ||
              code == COLOR_YUV2RGB_IYUV || code == COLOR_YUV2RGBA_IYUV)
                 ? 0
                 : 1;

  DataType in_data = src.GetDataType();
  DataType out_data = dst.GetDataType();

  const int32_t in_batch = src.GetDataShape().N;
  const int32_t in_channel = src.GetDataShape().C;
  const int32_t in_height = src.GetDataShape().H;
  const int32_t in_width = src.GetDataShape().W;

  const int32_t out_batch = dst.GetDataShape().N;
  const int32_t out_channel = dst.GetDataShape().C;
  const int32_t out_height = dst.GetDataShape().H;
  const int32_t out_width = dst.GetDataShape().W;

  if (out_channel != 3 && out_channel != 4) {
    // LOG_ERROR("Invalid output channel number " << outputShape.C);
    printf("Invalid output channel number\n");
    return ErrorCode::INVALID_DATA_SHAPE;
  }

  if (in_channel != 1) {
    // LOG_ERROR("Invalid input channel number " << inputShape.C);
    printf("Invalid input channel number\n");
    return ErrorCode::INVALID_DATA_SHAPE;
  }

  // if (in_height % 3 != 0 || in_width % 2 != 0) {
  // LOG_ERROR("Invalid input shape " << inputShape);
  // printf("Invalid input shape\n");
  // return ErrorCode::INVALID_DATA_SHAPE;
  // }
  if (in_data != kCV_8U || out_data != kCV_8U) {
    // LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
    return ErrorCode::INVALID_DATA_TYPE;
  }

  int rgb_width = in_width;
  // int rgb_height = in_height * 2 / 3;
  int rgb_height = in_height;
  if (out_height != rgb_height || out_width != rgb_width || out_batch != in_batch) {
    // LOG_ERROR("Invalid output shape " << outputShape);
    return ErrorCode::INVALID_DATA_SHAPE;
  }

  dim3 blockSize(BLOCK, BLOCK / 1, 1);
  dim3 gridSize(math::divUp(rgb_width, blockSize.x), math::divUp(rgb_height, blockSize.y), 1);

  int2 dstSize{out_width, out_height};
  int dcn = dst.GetDataShape().C;

  cuda::PackedTensor4D<unsigned char> src_pack(in_height, in_width, in_channel);
  cuda::Tensor4DWrap<unsigned char> src_ptr(src.GetBasePtr(), src_pack.stride1, src_pack.stride2, src_pack.stride3);
  cuda::Tensor4DWrap<unsigned char> dst_ptr(dst.GetBasePtr(), src_pack.stride1 * 3, src_pack.stride2 * 3,
                                            src_pack.stride3 * 3);

  switch (code) {
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2RGBA_NV21: {
      yuv420sp_to_bgr_char_nhwc<unsigned char>
          <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dstSize, dcn, bidx, uidx);
      checkKernelErrors();
    } break;
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2RGBA_IYUV: {
      yuv420p_to_bgr_char_nhwc<unsigned char>
          <<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, dstSize, dcn, bidx, uidx);
      checkKernelErrors();
    } break;

    default:
      // LOG_ERROR("Unsupported conversion code " << code);
      return ErrorCode::INVALID_PARAMETER;
  }
  return ErrorCode::SUCCESS;
}

__global__ void NV12ToYU12(const uint8_t *nv12, uint8_t *yu12, int image_width, int image_height) {
  int yIndex  = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIndex = 0;
  int uOffset = image_width * image_height;
  int vOffset = image_width * image_height + image_width * image_height / 4;

  if (yIndex < image_width * image_height) {
    yu12[yIndex] = nv12[yIndex];
  }

  uvIndex = yIndex;

  if (uvIndex < (image_width * image_height / 4)) {
    yu12[uOffset + uvIndex] = nv12[uOffset + uvIndex * 2];
    yu12[vOffset + uvIndex] = nv12[uOffset + uvIndex * 2 + 1];
  }
}

inline ErrorCode NV12ToYU12CUDA(const TensorData &src, const TensorData &dst, cudaStream_t stream) {
  const int32_t src_height = src.GetDataShape().H;
  const int32_t src_width = src.GetDataShape().W;
  int num_pixels = src_width * src_height;
  int num_uv_pixels = (src_width / 2) * (src_height / 2);
  int block_size = 256;
  int num_blocks = (num_pixels + block_size - 1) / block_size;
  // dim3 blockSize(BLOCK, BLOCK / 1, 1);
  // dim3 gridSize(math::divUp(src_width, blockSize.x), math::divUp(src_height, blockSize.y), 1);

  NV12ToYU12<<<num_blocks, block_size, 0, stream>>>(src.GetBasePtr(), dst.GetBasePtr(), src_width, src_height);

  return ErrorCode::SUCCESS;

  // cudaDeviceSynchronize();
}

ErrorCode CvtColor::Infer(const TensorData &src, const TensorData &dst, const ColorCVTCode code, cudaStream_t stream) {
  if (code == COLOR_NV122YU12) {
    return NV12ToYU12CUDA(src, dst, stream);
  } else {
    return YUV420xp_to_BGR(src, dst, code, stream);
  }
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
