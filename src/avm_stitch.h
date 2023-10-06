/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "image_data.h"
#include "types.h"
namespace lad {
namespace lcv {
class AvmStitch {
 private:
  /* data */
 public:
  AvmStitch(/* args */);
  ~AvmStitch();
  void Merge(const unsigned char* src1, const unsigned char* src2, const float* weights, unsigned char* dst, int left,
             int up, int src_width, int src_height, int dst_width, cudaStream_t stream);
  void Copy(const unsigned char* src, unsigned char* dst, int left, int up, int copy_width, int copy_height,
            int dst_width, cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad