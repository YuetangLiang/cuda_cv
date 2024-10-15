#include "coordinate_system_mapping.h"
#include "remap.h"

lcv::CoordinateSystemMapping coor_sys_map_;


void InitFisheyeMap(const Eigen::Matrix3d& trans_mat, const std::vector<float>& distort_k,
                                   const Eigen::Matrix3f& ins_mat, const bool use_lcv) {
  if (use_lcv) {
    float* temp_device_map_x = nullptr;
    float* temp_device_map_y = nullptr;
    float* temp_device_params = nullptr;
    float temp_params[16] = {trans_mat.cast<float>()(0, 0),
                             trans_mat.cast<float>()(0, 1),
                             trans_mat.cast<float>()(0, 2),
                             trans_mat.cast<float>()(1, 0),
                             trans_mat.cast<float>()(1, 1),
                             trans_mat.cast<float>()(1, 2),
                             trans_mat.cast<float>()(2, 0),
                             trans_mat.cast<float>()(2, 1),
                             trans_mat.cast<float>()(2, 2),
                             distort_k[0],
                             distort_k[1],
                             distort_k[2],
                             distort_k[3],
                             ins_mat(0, 0),
                             ins_mat(0, 2),
                             ins_mat(1, 2)};
    cudaMallocAsync(&temp_device_map_x, 1024 * 1024 * sizeof(float), reinterpret_cast<cudaStream_t>(stream_));
    cudaMallocAsync(&temp_device_map_y, 1024 * 1024 * sizeof(float), reinterpret_cast<cudaStream_t>(stream_));
    cudaMallocAsync(&temp_device_params, 16 * sizeof(float), reinterpret_cast<cudaStream_t>(stream_));
    cudaMemcpyAsync(reinterpret_cast<void*>(temp_device_params), reinterpret_cast<void*>(temp_params),
                    16 * sizeof(float), cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_));
    // lcv function
    coor_sys_map_.Operator(temp_device_map_x, temp_device_map_y, temp_device_params, 1024, 1024, sensor_size_.width,
                           sensor_size_.height, is_fisheye_, reinterpret_cast<cudaStream_t>(stream_));
    device_map_x_ = temp_device_map_x;
    device_map_y_ = temp_device_map_y;
    cudaFreeAsync(temp_device_params, reinterpret_cast<cudaStream_t>(stream_));
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  } else {
    map_x_ = cv::Mat::zeros(1024, 1024, CV_32F);
    map_y_ = cv::Mat::zeros(1024, 1024, CV_32F);
    for (int i = 0; i < 1024; i++) {    // width
      for (int j = 0; j < 1024; j++) {  // height
        double cylindar_lamda = (static_cast<double>(1023 - j) / 1024.) * M_PI;
        double cylindar_fai = (static_cast<double>(i) / 1024.) * M_PI;
        Eigen::Vector3d temp_sphere_vec;
        temp_sphere_vec << -std::sin(cylindar_fai) * std::cos(cylindar_lamda), std::cos(cylindar_fai),
            std::sin(cylindar_fai) * std::sin(cylindar_lamda);
        auto rotated_point = (trans_mat * temp_sphere_vec).cast<float>();

        auto theta = std::acos(rotated_point(2));
        auto theta_d = theta + distort_k[0] * std::pow(theta, 3.f) + distort_k[1] * std::pow(theta, 5.f) +
                       distort_k[2] * std::pow(theta, 7.f) + distort_k[3] * std::pow(theta, 9.f);
        auto alpha = std::atan2(rotated_point[1], rotated_point[0]);
        auto ux = Clamp(-ins_mat(0, 0) * theta_d * std::cos(alpha) + ins_mat(0, 2), 0.f,
                        static_cast<float>(sensor_size_.width - 1));
        auto vy = Clamp(-ins_mat(0, 0) * theta_d * std::sin(alpha) + ins_mat(1, 2), 0.f,
                        static_cast<float>(sensor_size_.height - 1));
        map_x_.at<float>(i, j) = ux;
        map_y_.at<float>(i, j) = vy;
      }
    }
  }
}

bool Undistort(const lad::lcv::ImageData& img_data, cv::Mat* const /*dst_image*/) const {
  lad::lcv::Remap remap;
  lad::lcv::ImageData lcv_output;
  lad::lcv::ImageData lcv_input = img_data;

  if (virtual_sensor_info_name_ == "camera_front_mid_virtual") {
    size_t input_size = static_cast<size_t>(lcv_input.width * lcv_input.height * 3 * static_cast<int>(sizeof(uint8_t)));
    linfer::util::memcpy_d2d(dst_tensor_->get_buf(), reinterpret_cast<void*>(img_data.base_ptr), input_size, stream_);
    lcv_input.base_ptr = reinterpret_cast<uint8_t*>(dst_tensor_->get_buf());
  }

  lcv_output.format = lad::lcv::kBGR_8U;
  if (is_fisheye_) {
    lcv_output.width = 1024;
    lcv_output.height = 1024;
  } else {
    lcv_output.width = virtual_size_.width;
    lcv_output.height = virtual_size_.height;
  }
  lcv_output.row_stride = lcv_output.width;
  lcv_output.base_ptr = reinterpret_cast<uint8_t*>(src_tensor_->get_buf());
  remap.Operator(lcv_input, lcv_output, device_map_x_, device_map_y_, reinterpret_cast<cudaStream_t>(stream_));
  return true;
}



