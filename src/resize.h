/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "types.h"
#include "image_data.h"
namespace lad {
namespace lcv {
class Resize {
 private:
  /* data */
 public:
  Resize(/* args */);
  ~Resize();
  ErrorCode Operator(const ImageData &src, const ImageData &dst, const InterpolationType interpolation,
                     cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad