/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "types.h"
#include "image_data.h"
namespace lad {
namespace lcv {
class CylindricalProjector {
 private:
  /* data */
 public:
  CylindricalProjector(/* args */);
  ~CylindricalProjector();
  ErrorCode Operator(const ImageData &src, const ImageData &dst, const InterpolationType interpolation,
                     cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad