/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "types.h"
#include "image_data.h"
namespace lad {
namespace lcv {
class CvtColor {
 private:
  /* data */
 public:
  CvtColor(/* args */);
  ~CvtColor();
  ErrorCode Operator(const ImageData &src, const ImageData &dst, const ColorCVTCode code, cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad