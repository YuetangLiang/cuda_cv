
#include "lcv_cuda.h"
#include "math/math.h"

#define BLOCK 32

namespace lad {
namespace lcv {
namespace cuda {

__global__ void mapping_kernel(float *map_x, float *map_y, const float *map_params, int width, int height,
                               int sensor_width, int sensor_high) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < width && j < height) {
    double cylindar_lamda = (static_cast<double>(1023 - j) / 1024.) * M_PI;
    double cylindar_fai = (static_cast<double>(i) / 1024.) * M_PI;
    double rotated_point[3] = {0., 0., 0.};
    double sphere_vec[3] = {-std::sin(cylindar_fai) * std::cos(cylindar_lamda), 
			                      std::cos(cylindar_fai),
                            std::sin(cylindar_fai) * std::sin(cylindar_lamda)};
    for (size_t row = 0; row < 3; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        rotated_point[row] += map_params[row * 3 + col] * sphere_vec[col];
      }
    }

    double theta = std::acos(rotated_point[2]);
    double theta_d = theta + map_params[9] * std::pow(theta, 3.f) + map_params[10] * std::pow(theta, 5.f) +
                     map_params[11] * std::pow(theta, 7.f) + map_params[12] * std::pow(theta, 9.f);
    double alpha = std::atan2(rotated_point[1], rotated_point[0]);
    auto ux = math::Clamp(static_cast<float>(-map_params[13] * theta_d * std::cos(alpha) + map_params[14]), 
													0.f,
                          static_cast<float>(sensor_width - 1));
    auto vy = math::Clamp(static_cast<float>(-map_params[13] * theta_d * std::sin(alpha) + map_params[15]), 
													0.f,
                          static_cast<float>(sensor_high - 1));

    map_x[i * width + j] = ux;
    map_y[i * width + j] = vy;
  }
}


__global__ void mapping_pinhole_kernel(float *map_x, float *map_y, const float *map_params, int width, int height,
                                        int sensor_width, int sensor_high) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < width && j < height) {

    double rotated_point[3] = {0., 0., 0.};
    double sphere_vec[3] = {static_cast<double>(i - width/2) / map_params[16], static_cast<double>(j - height/2) / map_params[16],
                            1};
    for (size_t row = 0; row < 3; ++row) {
      for (size_t col = 0; col < 3; ++col) {
        rotated_point[row] += map_params[row * 3 + col] * sphere_vec[col];
      }
    }
    double u = rotated_point[0] / rotated_point[2];
    double v = rotated_point[1] / rotated_point[2];
    double r = std::sqrt(u * u + v * v);
    double theta = std::atan(r);
    double theta_d = theta + map_params[9] * std::pow(theta, 3.f) + map_params[10] * std::pow(theta, 5.f) +
                     map_params[11] * std::pow(theta, 7.f) + map_params[12] * std::pow(theta, 9.f);
    // float scale = (r > 1e-8f) ? theta_d / r : 1.0f;
    double scale = theta_d / r;
    auto ux = math::Clamp(static_cast<float>(map_params[13] * scale * u + map_params[14]), 
													0.f, static_cast<float>(sensor_width - 1));
    auto vy = math::Clamp(static_cast<float>(map_params[13] * scale * v + map_params[15]), 
													0.f, static_cast<float>(sensor_high - 1));

    map_x[j * width + i] = ux;
    map_y[j * width + i] = vy;
  }
}

ErrorCode CoordinateSystemMapping::Infer(float *map_x, float *map_y, const float *map_params, int width, int height,
                                         int sensor_width, int sensor_high, bool is_fisheye, cudaStream_t stream) {
  dim3 blockSize(BLOCK, BLOCK / 1, 1);
  dim3 gridSize(math::divUp(width, blockSize.x), math::divUp(height, blockSize.y), 1);
  if (is_fisheye) {
    mapping_kernel<<<gridSize, blockSize, 0, stream>>>(map_x, map_y, map_params, width, height, sensor_width, sensor_high);
  } else {
    mapping_pinhole_kernel<<<gridSize, blockSize, 0, stream>>>(map_x, map_y, map_params, width, height, sensor_width, sensor_high);
  }
  

  return ErrorCode::SUCCESS;
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
