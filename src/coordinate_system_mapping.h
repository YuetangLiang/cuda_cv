
#pragma once
#include <cuda_runtime.h>
#include "image_data.h"
#include "types.h"

namespace lad {
namespace lcv {
class ≤CoordinateSystemMapping {
 private:
  /* data */
 public:
  CoordinateSystemMapping(/* args */);
  ~CoordinateSystemMapping();
  ErrorCode Operator(float *map_x, float *map_y, const float *map_params, int width, int height, int sensor_width,
                     int sensor_high, bool is_fisheye, cudaStream_t stream);
};

}  // namespace lcv
}  // namespace lad