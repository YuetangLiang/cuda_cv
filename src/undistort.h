/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "image_data.h"
#include "types.h"
namespace lad {
namespace lcv {
class UnDistort {
 private:
  /* data */
 public:
  UnDistort(/* args */);
  ~UnDistort();
  ErrorCode Operator(const ImageData &src, const ImageData &dst, float* const camera_matrix, float* const distCoeffs, 
                            const int* dist_len, const int* dist_len_current, cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad