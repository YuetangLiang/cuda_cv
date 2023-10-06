/******************************************************************************
 *****************************************************************************/
#pragma once
#include <cuda_runtime.h>
#include "tensor/tensor_data.h"
#include "types.h"
namespace lad {
namespace lcv {
namespace cuda {
class ResizeOp {
 public:
  ResizeOp(){};
  ~ResizeOp(){};
  int Infer(const TensorData &src, const TensorData &dst, const InterpolationType interpolation, cudaStream_t stream);
};

class CvtColor {
 public:
  CvtColor(){};
  ~CvtColor(){};
  ErrorCode Infer(const TensorData &src, const TensorData &dst, const ColorCVTCode code, cudaStream_t stream);
};

class CylindricalProjector {
 public:
  CylindricalProjector(){};
  ~CylindricalProjector(){};
  ErrorCode Infer(const TensorData &src, const TensorData &dst, const InterpolationType interpolation,
                  cudaStream_t stream);
};

class WarpPerspective {
 public:
  WarpPerspective(){};
  ~WarpPerspective(){};
  ErrorCode Infer(const TensorData &src, const TensorData &dst, const float *transMatrix, const int interpolation,
                  int borderMode, const float4 borderValue, cudaStream_t stream);
};

class UnDistort {
 public:
  UnDistort(){};
  ~UnDistort(){};
  ErrorCode Infer(const TensorData &src, const TensorData &dst, float *const camera_matrix, float *const distCoeffs,
                  const int *dist_len, const int *dist_len_current, cudaStream_t stream);
};

class Remap {
 public:
  Remap(){};
  ~Remap(){};
  ErrorCode Infer(const TensorData &src, const TensorData &dst, const float *map_x, const float *map_y,
                  cudaStream_t stream);
};

class AvmStitch {
 public:
  AvmStitch(){};
  ~AvmStitch(){};
  void Merge(const unsigned char *src1, const unsigned char *src2, const float *weights, unsigned char *dst, int left,
             int up, int src_width, int src_height, int dst_width, cudaStream_t stream);

  void Copy(const unsigned char *src, unsigned char *dst, int left, int up, int copy_width, int copy_height,
            int dst_width, cudaStream_t stream);
};

}  // namespace cuda
}  // namespace lcv
}  // namespace lad