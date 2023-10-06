/******************************************************************************
 *****************************************************************************/

#include "lcv_cuda.h"

namespace lad {
namespace lcv {
namespace cuda {

__global__ void avm_stitch_merge(const unsigned char* src1, const unsigned char* src2, const float* weights,
                                 unsigned char* dst, int left, int up, int src_width, int src_height, int dst_width) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < src_width && y < src_height) {
    int src_index = (y * src_width + x);
    int dst_index = ((up + y) * dst_width + (left + x)) * 3;
    for (int i = 0; i < 3; i++) {
      dst[dst_index + i] = static_cast<unsigned char>(src1[dst_index + i] * weights[src_index] +
                                                      src2[dst_index + i] * (1 - weights[src_index]));
    }
  }
}

__global__ void avm_stitch_copy(const unsigned char* src, unsigned char* dst, int left, int up, int copy_width,
                                int copy_height, int dst_width) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < copy_width && y < copy_height) {
    int index = ((up + y) * dst_width + (left + x)) * 3;
    for (int i = 0; i < 3; i++) {
      dst[index + i] = src[index + i];
    }
  }
}

void AvmStitch::Merge(const unsigned char* src1, const unsigned char* src2, const float* weights, unsigned char* dst,
                      int left, int up, int src_width, int src_height, int dst_width, cudaStream_t stream) {
  const dim3 blockDim(32, 32, 1);
  const dim3 gridDim((src_width + blockDim.x - 1) / blockDim.x, (src_height + blockDim.y - 1) / blockDim.y, 1);
  avm_stitch_merge<<<gridDim, blockDim, 0, stream>>>(src1, src2, weights, dst, left, up, src_width, src_height,
                                                     dst_width);
}

void AvmStitch::Copy(const unsigned char* src, unsigned char* dst, int left, int up, int copy_width, int copy_height,
                     int dst_width, cudaStream_t stream) {
  const dim3 blockDim(32, 32, 1);
  const dim3 gridDim((copy_width + blockDim.x - 1) / blockDim.x, (copy_height + blockDim.y - 1) / blockDim.y, 1);
  avm_stitch_copy<<<gridDim, blockDim, 0, stream>>>(src, dst, left, up, copy_width, copy_height, dst_width);
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
