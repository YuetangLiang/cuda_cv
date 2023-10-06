/******************************************************************************
 *****************************************************************************/

#include "cvdef.h"
#include "lcv_cuda.h"
#include "math/math.h"
#include "tensor/packed_tensor.h"
#include "tensor/tensor_wrap.h"

namespace lad {
namespace lcv {
namespace cuda {

__global__ void undistort_(unsigned char* src, unsigned char* dst, float* const camera_matrix, float* const distCoeffs,
                           const int* dist_len, const int* dist_len_current, const int batchSize, const int width,
                           const int height, const int channel) {
  int xIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int yIndex = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = yIndex * width * channel + xIndex * channel;
  for (int i = blockIdx.z; i < batchSize; i += gridDim.z) {
    float k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0, k4 = 0.0, k5 = 0.0, k6 = 0.0;
    if (dist_len[i] == 5) {
      k1 = distCoeffs[dist_len_current[i] + 0];
      k2 = distCoeffs[dist_len_current[i] + 1];
      p1 = distCoeffs[dist_len_current[i] + 2];
      p2 = distCoeffs[dist_len_current[i] + 3];
      k3 = distCoeffs[dist_len_current[i] + 4];
    } else {
      k1 = distCoeffs[dist_len_current[i] + 0];
      k2 = distCoeffs[dist_len_current[i] + 1];
      p1 = distCoeffs[dist_len_current[i] + 2];
      p2 = distCoeffs[dist_len_current[i] + 3];
      k3 = distCoeffs[dist_len_current[i] + 4];
      k4 = distCoeffs[dist_len_current[i] + 5];
      k5 = distCoeffs[dist_len_current[i] + 6];
      k6 = distCoeffs[dist_len_current[i] + 7];
    }

    float fx = camera_matrix[4 * i + 0];
    float fy = camera_matrix[4 * i + 1];
    float cx = camera_matrix[4 * i + 2];
    float cy = camera_matrix[4 * i + 3];

    float x = ((float)xIndex - cx) / fx;
    float y = ((float)yIndex - cy) / fy;
    float r_square = x * x + y * y;
    float scale = 1 + k1 * r_square + k2 * r_square * r_square + k3 * r_square * r_square * r_square;
    float divisor = 1 + k4 * r_square + k5 * r_square * r_square + k6 * r_square * r_square * r_square;
    float x_tangent_distort = 2 * p1 * x * y + p2 * (r_square + 2 * x * x);
    float y_tangent_distort = p1 * (r_square + 2 * y * y) + 2 * p2 * x * y;

    float x0 = x * scale / divisor + x_tangent_distort;
    float y0 = y * scale / divisor + y_tangent_distort;
    int x_undist = int(x0 * fx + cx);
    int y_undist = int(y0 * fy + cy);

    // 最近邻
    if (x_undist >= 0 && y_undist >= 0 && x_undist < width && y_undist < height) {
      dst[i * width * height * channel + idx + 0] =
          src[i * width * height * channel + y_undist * width * channel + x_undist * channel + 0];
      dst[i * width * height * channel + idx + 1] =
          src[i * width * height * channel + y_undist * width * channel + x_undist * channel + 1];
      dst[i * width * height * channel + idx + 2] =
          src[i * width * height * channel + y_undist * width * channel + x_undist * channel + 2];
    } else {
      dst[i * width * height * channel + idx + 0] = 0;
      dst[i * width * height * channel + idx + 1] = 0;
      dst[i * width * height * channel + idx + 2] = 0;
    }
  }
}

ErrorCode UnDistort::Infer(const TensorData& src, const TensorData& dst, float* const camera_matrix, float* const distCoeffs,
                           const int* dist_len, const int* dist_len_current, cudaStream_t stream) {
  const int batch = src.GetDataShape().N;
  const int channel = src.GetDataShape().C;
  const int height = src.GetDataShape().H;
  const int width = src.GetDataShape().W;

  const dim3 blockDim(32, 32, 1);
  const dim3 gridDim(math::divUp(width, blockDim.x), math::divUp(height, blockDim.y), batch);
  undistort_<<<gridDim, blockDim, 0, stream>>>((unsigned char*)src.GetBasePtr(), (unsigned char*)dst.GetBasePtr(),
                                                     camera_matrix, distCoeffs, dist_len, dist_len_current, batch,
                                                     width, height, channel);

  return ErrorCode::SUCCESS;
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
