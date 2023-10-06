/******************************************************************************
 *****************************************************************************/

#include "lcv_cuda.h"
#include "math/math.h"

namespace lad {
namespace lcv {
namespace cuda {

#define BLOCK 32

__global__ void remap_(const unsigned char *src, int src_width, int src_height, unsigned char *dst, int dst_width,
                       int dst_height, const float *map_x, const float *map_y) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < dst_width && y < dst_height) {
    int index = (y * dst_width + x) * 3;

    float src_x = map_x[index / 3];
    float src_y = map_y[index / 3];

    if (src_x >= 0 && src_x < src_width - 1 && src_y >= 0 && src_y < src_height - 1) {
      int x0 = floorf(src_x);
      int y0 = floorf(src_y);
      int x1 = x0 + 1;
      int y1 = y0 + 1;

      float tx = src_x - x0;
      float ty = src_y - y0;

      int src_index00 = (y0 * src_width + x0) * 3;
      int src_index10 = (y0 * src_width + x1) * 3;
      int src_index01 = (y1 * src_width + x0) * 3;
      int src_index11 = (y1 * src_width + x1) * 3;

      for (int i = 0; i < 3; i++) {
        float value00 = src[src_index00 + i];
        float value10 = src[src_index10 + i];
        float value01 = src[src_index01 + i];
        float value11 = src[src_index11 + i];

        float value0 = value00 * (1.0f - tx) + value10 * tx;
        float value1 = value01 * (1.0f - tx) + value11 * tx;

        float value = value0 * (1.0f - ty) + value1 * ty;

        dst[index + i] = static_cast<unsigned char>(value);
      }
    }
  }
}

ErrorCode Remap::Infer(const TensorData &src, const TensorData &dst, const float *map_x, const float *map_y,
                       cudaStream_t stream) {
  const int in_batch = src.GetDataShape().N;
  const int in_channel = src.GetDataShape().C;
  const int in_height = src.GetDataShape().H;
  const int in_width = src.GetDataShape().W;
  uint8_t *in_ptr = src.GetBasePtr();

  const int out_batch = dst.GetDataShape().N;
  const int out_channel = dst.GetDataShape().C;
  const int out_height = dst.GetDataShape().H;
  const int out_width = dst.GetDataShape().W;
  uint8_t *out_ptr = dst.GetBasePtr();

  // Ptr2dNHWC<T> src_tensor(in_batch, in_height, in_width, in_channel, reinterpret_cast<T *>(in_ptr));
  // Ptr2dNHWC<T> dst_tensor(out_batch, out_height, out_width, out_channel, reinterpret_cast<T *>(out_ptr));

  dim3 block(BLOCK, BLOCK, out_batch);
  dim3 grid(math::divUp(out_width + block.x - 1, block.x), math::divUp(out_height + block.y - 1, block.y), out_batch);
  remap_<<<grid, block, 0, stream>>>(in_ptr, in_width, in_height, out_ptr, out_width, out_height, map_x, map_y);

  return SUCCESS;
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
