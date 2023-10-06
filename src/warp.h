/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "image_data.h"
#include "types.h"
namespace lad {
namespace lcv {
class WarpPerspective {
 private:
  /* data */
 public:
  WarpPerspective(/* args */);
  ~WarpPerspective();
  ErrorCode Operator(const ImageData &src, const ImageData &dst, const float *xform, const int32_t flags,
                     const BorderType borderMode, const float4 borderValue, cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad