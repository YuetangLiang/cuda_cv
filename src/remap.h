/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "image_data.h"
#include "types.h"
namespace lad {
namespace lcv {
class Remap {
 private:
  /* data */
 public:
  Remap(/* args */);
  ~Remap();
  ErrorCode Operator(const ImageData& src, const ImageData& dst, const float* map_x, const float* map_y,
                     cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad