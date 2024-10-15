
#include "coordinate_system_mapping.h"
#include "lcv_cuda.h"
namespace lad {
namespace lcv {

CoordinateSystemMapping::CoordinateSystemMapping(/* args */) { ; }
CoordinateSystemMapping::~CoordinateSystemMapping() { ; }
ErrorCode CoordinateSystemMapping::Operator(float *map_x, float *map_y, const float *map_params, int width, int height,
                                            int sensor_width, int sensor_high, bool is_fisheye, cudaStream_t stream) {
  cuda::CoordinateSystemMapping coordinate_sys_mapping;
  coordinate_sys_mapping.Infer(map_x, map_y, map_params, width, height, sensor_width, sensor_high, is_fisheye, stream);
  return SUCCESS;
}
}  // namespace lcv
}  // namespace lad