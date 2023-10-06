/******************************************************************************
 *****************************************************************************/

#include <algorithm>
#include <cmath>
#include "cvdef.h"
#include "lcv_cuda.h"
#include "math/math.h"
#include "math/saturate.h"
#include "tensor/packed_tensor.h"
#include "tensor/tensor_wrap.h"
#include "utils/type_traits.h"
namespace lad {
namespace lcv {
namespace cuda {

#define get_batch_idx() (blockIdx.z)

//******************** NN = Nearest Neighbor

template <typename T>
__global__ void projector_NN(cuda::Tensor3DWrap<const T> src, cuda::Tensor3DWrap<T> dst, int2 srcSize, int2 dstSize,
                             const float scale_x, const float scale_y) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = get_batch_idx();
  int out_height = dstSize.y, out_width = dstSize.x;

#if 0
  //const float k_rinv[9] = {6595.48, 0, 1356.33, 0, 6595.48, 970.54, 0, 0, 1}; //front_far
  //const float k_rinv[9] = {1413.70, 0, 1920.21, 0, 1413.70, 1080.20, 0, 0, 1};  //front_wide
  const float k_rinv[9] = {1 / 1060.884429, 0, -1003.561856 / 582.4403620000001, 
                           0, 1 / 1060.884429, -582.4403620000001 / 1060.884429, 
                           0, 0, 1};  //front_wide

  const float scale = 1;

  float tmp_x = dst_x / scale;
  float tmp_y = dst_y / scale;

  float x_ = sinf(tmp_x);
  float y_ = tmp_y;
  float z_ = cosf(tmp_x);

  float z;
  float sx = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
  float sy = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
  z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

  if (z > 0) {
    sx /= z;
    sy /= z;
  } else {
    sx = sy = -1;
  }

  printf("dst_x = %d, dst_y = %d, x_ = %f, y_ = %f, z_ = %f, sx = %f, sy = %f\n", dst_x, dst_y, x_, y_, z_, sx, sy);

  if ((dst_x < out_width) && (dst_y < out_height) && (sx >= 0) && (sy >= 0)) {  // generic copy pixel to pixel
    const int sx_tmp = cuda::math::min(__float2int_rd(sx), srcSize.x - 1);
    const int sy_tmp = cuda::math::min(__float2int_rd(sy), srcSize.y - 1);
    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy_tmp, sx_tmp);
  }
#endif

#if 0
  float center_x = (out_width / 2);
  float center_y = (out_height / 2);
  float f = 600;

  float sx = ((f * tan((dst_x - center_x) / f) + center_x));
  //float sy = (((dst_y - center_y) / cos(atan((dst_x - center_x) / f)) + center_y));
  float sy = (((dst_y - center_y) / cos((dst_x - center_x) / f)) + center_y);

  //printf("dst_x = %d, dst_y = %d, sx = %f, sy = %f\n", dst_x, dst_y, sx, sy);
  /* if (sx < 0) {
    sx = 0;
  }

  if (sx >= srcSize.x) {
    sx = srcSize.x - 1;
  }

  if (sy < 0) {
    sy = 0;
  }

  if (sy >= srcSize.y) {
    sy = srcSize.y - 1;
  } */

  //printf("srcSize.x = %d, srcSize.y = %d, out_height = %d, out_width = %d, dst_x = %d, dst_y = %d, sx = %f, sy = %f, cvt_sx = %d, cvt_sy = %d\n", srcSize.x, srcSize.y, out_height, out_width, dst_x, dst_y, sx, sy, __float2int_rd(sx), __float2int_rd(sy));

  if ((dst_x < out_width) && (dst_y < out_height) && (sx >= 0) && (sy >= 0)) {  // generic copy pixel to pixel
    const int sx_tmp = cuda::math::min(__float2int_rd(sx), srcSize.x - 1);
    const int sy_tmp = cuda::math::min(__float2int_rd(sy), srcSize.y - 1);
    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy_tmp, sx_tmp);
  }
#endif

#if 0
  double R = out_width / 2;
  double k = R / sqrt(R * R + (dst_x - out_width / 2) * (dst_x - out_width / 2));
  float sx = (dst_x - out_width / 2) / k + out_width / 2;
  float sy = (dst_y - out_height / 2) / k + out_height / 2;

  printf("srcSize.x = %d, srcSize.y = %d, out_height = %d, out_width = %d, dst_x = %d, dst_y = %d, sx = %f, sy = %f, cvt_sx = %d, cvt_sy = %d\n", srcSize.x, srcSize.y, out_height, out_width, dst_x, dst_y, sx, sy, __float2int_rd(sx), __float2int_rd(sy));

  if ((dst_x < out_width) && (dst_y < out_height) && (sx >= 0) && (sy >= 0)) {  // generic copy pixel to pixel
    const int sx_tmp = cuda::math::min(__float2int_rd(sx), srcSize.x - 1);
    const int sy_tmp = cuda::math::min(__float2int_rd(sy), srcSize.y - 1);
    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy_tmp, sx_tmp);
  }
#endif

#if 1
  const float PI = 3.1415926;
  int w = srcSize.x;
  int h = srcSize.y;
  double n = 10;
  double sita = 2 * PI / n;
  double f = w / 2 / tan(sita / 2);
  
  double x = f * tan((dst_x / f) - sita / 2) + w / 2;
  double y = (dst_y - h / 2) * sqrt((x - w / 2) * (x - w / 2) + f * f) / f + h / 2;

  if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
    double u, v;
    //u = x - int(x);
    //v = y - int(y);
    //T s0, s1, s2, s3, s4;
    // s1 = imageIn.at<Vec3b>(int(y), int(x));
    // s2 = imageIn.at<Vec3b>(int(y), int(x) + 1);
    // s3 = imageIn.at<Vec3b>(int(y) + 1, int(x));
    // s4 = imageIn.at<Vec3b>(int(y) + 1, int(x) + 1);
    //s1 = *src.ptr(batch_idx, int(y), int(x));
    //s2 = *src.ptr(batch_idx, int(y), int(x) + 1);
    //s3 = *src.ptr(batch_idx, int(y) + 1, int(x));
    //s4 = *src.ptr(batch_idx, int(y) + 1, int(x) + 1);
    //s0 = (1 - u) * (1 - v) * s1 + (1 - u) * v * s3 + u * (1 - v) * s2 + u * v * s4;  // 使用双线性插值计算对应点的像素值
    // imageOut.at<Vec3b>(dst_y, dst_x) = s0;
    //*dst.ptr(batch_idx, dst_y, dst_x) = s0;

    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, __float2int_rd(y), __float2int_rd(x));
  }
  if (x == w - 1 || y == h - 1) {
    // imageOut.at<Vec3b>(dst_y, dst_x) = imageIn.at<Vec3b>(x, y);
    *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, __float2int_rd(y), __float2int_rd(x));
  }
#endif
}

template <typename T>
void cylindrical_projector(const TensorData &src, const TensorData &dst, const InterpolationType interpolation,
                           cudaStream_t stream)
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
  // const int out_quad_width = out_width / 4;
  const int out_quad_width = out_width;
  const dim3 quadGridSize(cuda::math::divUp(out_quad_width, blockSize.x), cuda::math::divUp(out_height, blockSize.y),
                          batch_size);

  // bool can_quad = ((((size_t)dst_ptr) % sizeof(T)) == 0) && ((out_width % 4) == 0);  //is the output buffer
  // quad-pixel aligned?
  bool can_quad = ((out_width % 4) == 0);  // is the output buffer quad-pixel aligned?

  // Note: resize is fundamentally a gather memory operation, with a little bit of compute
  //       our goals are to (a) maximize throughput, and (b) minimize occupancy for the same performance
  switch (interpolation) {
    case INTERP_NEAREST:
      projector_NN<T>
          <<<quadGridSize, blockSize, 0, stream>>>(src_tensor, dst_tensor, srcSize, dstSize, scale_x, scale_y);
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

ErrorCode CylindricalProjector::Infer(const TensorData &src, const TensorData &dst, const InterpolationType interpolation,
                                      cudaStream_t stream) {
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

  typedef void (*func_t)(const TensorData &src, const TensorData &dst, const InterpolationType interpolation,
                         cudaStream_t stream);

  static const func_t funcs[6][4] = {{cylindrical_projector<uchar>, 0 /*cylindrical_projector<uchar2>*/,
                                      cylindrical_projector<uchar3>, cylindrical_projector<uchar4>},
                                     {0 /*cylindrical_projector<schar>*/, 0 /*cylindrical_projector<schar2>*/,
                                      0 /*cylindrical_projector<schar3>*/, 0 /*cylindrical_projector<schar4>*/},
                                     {cylindrical_projector<ushort>, 0 /*cylindrical_projector<ushort2>*/,
                                      cylindrical_projector<ushort3>, cylindrical_projector<ushort4>},
                                     {cylindrical_projector<short>, 0 /*cylindrical_projector<short2>*/,
                                      cylindrical_projector<short3>, cylindrical_projector<short4>},
                                     {0 /*cylindrical_projector<int>*/, 0 /*cylindrical_projector<int2>*/,
                                      0 /*cylindrical_projector<int3>*/, 0 /*cylindrical_projector<int4>*/},
                                     {cylindrical_projector<float>, 0 /*cylindrical_projector<float2>*/,
                                      cylindrical_projector<float3>, cylindrical_projector<float4>}};

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
