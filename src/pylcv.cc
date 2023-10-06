
#include <stdexcept>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "remap.h"
#include "warp.h"

namespace {

namespace py = pybind11;

void check_cuda(cudaError_t err, char const* const func, const char* const file, int const line) {
  if (cudaSuccess != err) {
    auto msg = "call " + std::string(func) + "(), code: " + std::to_string(err);
    throw std::runtime_error(msg);
  }
}

#define CHECK_CUDA(f) check_cuda((f), #f, __FILE__, __LINE__)

lad::lcv::ImageFormat convert2format(const std::string& img_format) {
  if ("NV12_8U" == img_format) {
    return lad::lcv::ImageFormat::kNV12_8U;
  } else if ("IYUV_8U" == img_format) {
    return lad::lcv::ImageFormat::kIYUV_8U;
  } else if ("BGR_8U" == img_format) {
    return lad::lcv::ImageFormat::kBGR_8U;
  } else if ("RGB_8U" == img_format) {
    return lad::lcv::ImageFormat::kRGB_8U;
  } else {
    throw std::runtime_error("unsupport image format: " + img_format);
  }
}

size_t get_img_size(size_t height, size_t width, lad::lcv::ImageFormat format) {
  switch (format) {
    case lad::lcv::ImageFormat::kNV12_8U:
    case lad::lcv::ImageFormat::kIYUV_8U:
      return height * width * 3 / 2;
    case lad::lcv::ImageFormat::kBGR_8U:
    case lad::lcv::ImageFormat::kRGB_8U:
      return height * width * 3;
    default:
      throw std::runtime_error("unsupport image format: " + std::to_string(format));
      break;
  }
}

class PyRemap {
 private:
  cudaStream_t stream_;
  lad::lcv::Remap remap_;
  size_t height_, width_, img_size_;
  void* img_ptr_cuda_src_;
  void* img_ptr_cuda_dest_;
  lad::lcv::ImageFormat img_format_;

 public:
  PyRemap(size_t height, size_t width, const std::string& format) {
    img_format_ = convert2format(format);
    CHECK_CUDA(cudaStreamCreate(&stream_));
    height_ = static_cast<size_t>(height);
    width_ = static_cast<size_t>(width);
    img_size_ = get_img_size(height_, width_, img_format_);
    printf("img_size_: %lu\n", img_size_);
    CHECK_CUDA(cudaMalloc(&img_ptr_cuda_src_, img_size_));
    CHECK_CUDA(cudaMalloc(&img_ptr_cuda_dest_, img_size_));
  }

  int Operator(const py::array_t<uint8_t>& img_src, py::array_t<uint8_t>& img_dest, const py::array_t<float>& map_x,
               const py::array_t<float>& map_y) {
    lad::lcv::ImageData img_data_src, img_data_dest;
    py::buffer_info src_buf_info = img_src.request();
    py::buffer_info dest_buf_info = img_dest.request();
    py::buffer_info map_x_buf_info = map_x.request();
    py::buffer_info map_y_buf_info = map_y.request();

    if (src_buf_info.shape[0] != height_ || src_buf_info.shape[1] != width_ || dest_buf_info.shape[0] != height_ ||
        dest_buf_info.shape[1] != width_) {
      throw std::logic_error("size of src/dest is not match with the initialize parameter");
    }

    img_data_src.format = img_format_;
    img_data_src.height = src_buf_info.shape[0];
    img_data_src.width = src_buf_info.shape[1];
    img_data_src.row_stride = src_buf_info.shape[1];
    img_data_src.base_ptr = static_cast<uint8_t*>(img_ptr_cuda_src_);
    // printf("src info, format: %s, height: %d, width: %d\n", img_format.c_str(), img_data_src.height,
    //        img_data_src.width);

    img_data_dest.format = img_format_;
    img_data_dest.height = dest_buf_info.shape[0];
    img_data_dest.width = dest_buf_info.shape[1];
    img_data_dest.row_stride = dest_buf_info.shape[1];
    img_data_dest.base_ptr = static_cast<uint8_t*>(img_ptr_cuda_dest_);
    // printf("dest info, format: %s, height: %d, width: %d\n", img_format.c_str(), img_data_dest.height,
    //        img_data_dest.width);

    cudaMemcpyAsync(img_ptr_cuda_src_, static_cast<uint8_t*>(src_buf_info.ptr), img_size_, cudaMemcpyHostToDevice,
                    stream_);
    auto ret = remap_.Operator(img_data_src, img_data_dest, static_cast<float*>(map_x_buf_info.ptr),
                               static_cast<float*>(map_y_buf_info.ptr), stream_);
    cudaMemcpyAsync(static_cast<uint8_t*>(dest_buf_info.ptr), img_ptr_cuda_dest_, img_size_, cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);
    return static_cast<int>(ret);
  }

  ~PyRemap() {
    cudaFree(img_ptr_cuda_src_);
    cudaFree(img_ptr_cuda_dest_);
  }
};

class PyWarpPerspective {
 private:
  cudaStream_t stream_;
  lad::lcv::WarpPerspective warp_perspective_;
  size_t height_, width_, img_size_;
  void* img_ptr_cuda_src_;
  void* img_ptr_cuda_dest_;
  lad::lcv::ImageFormat img_format_;

 public:
  PyWarpPerspective(size_t height, size_t width, const std::string& format) {
    img_format_ = convert2format(format);
    CHECK_CUDA(cudaStreamCreate(&stream_));
    height_ = static_cast<size_t>(height);
    width_ = static_cast<size_t>(width);
    img_size_ = get_img_size(height_, width_, img_format_);
    printf("height_: %lu, width_:%lu, img_size_: %lu\n", height_, width_, img_size_);
    CHECK_CUDA(cudaMalloc(&img_ptr_cuda_src_, img_size_));
    CHECK_CUDA(cudaMalloc(&img_ptr_cuda_dest_, img_size_));
  }

  int Operator(const py::array_t<uint8_t>& img_src, py::array_t<uint8_t>& img_dest, const py::array_t<float>& xform,
               int flags, int border_mode, const py::array_t<float>& border_value) {
    lad::lcv::ImageData img_data_src, img_data_dest;
    py::buffer_info src_buf_info = img_src.request();
    py::buffer_info dest_buf_info = img_dest.request();
    py::buffer_info xform_buf_info = xform.request();
    py::buffer_info border_value_buf_info = border_value.request();

    if (src_buf_info.shape[0] != height_ || src_buf_info.shape[1] != width_ || dest_buf_info.shape[0] != height_ ||
        dest_buf_info.shape[1] != width_) {
      throw std::logic_error("size of src/dest is not match with the initialize parameter");
    }

    img_data_src.format = img_format_;
    img_data_src.height = src_buf_info.shape[0];
    img_data_src.width = src_buf_info.shape[1];
    img_data_src.row_stride = src_buf_info.shape[1];
    img_data_src.base_ptr = static_cast<uint8_t*>(img_ptr_cuda_src_);  // static_cast<uint8_t*>(src_buf_info.ptr);
    // printf("src info, format: %s, height: %d, width: %d, data: %u\n", img_format.c_str(), img_data_src.height,
    //        img_data_src.width, img_data_dest.base_ptr[0]);

    img_data_dest.format = img_format_;
    img_data_dest.height = dest_buf_info.shape[0];
    img_data_dest.width = dest_buf_info.shape[1];
    img_data_dest.row_stride = dest_buf_info.shape[1];
    img_data_dest.base_ptr = static_cast<uint8_t*>(img_ptr_cuda_dest_);  // static_cast<uint8_t*>(dest_buf_info.ptr);
    // printf("dest info, format: %s, height: %d, width: %d, data: %u\n", img_format.c_str(), img_data_dest.height,
    //        img_data_dest.width, img_data_dest.base_ptr[0]);

    float4 border_val_f4;
    float* border_value_f = static_cast<float*>(border_value_buf_info.ptr);
    border_val_f4.x = border_value_f[0];
    border_val_f4.y = border_value_f[1];
    border_val_f4.z = border_value_f[2];
    border_val_f4.w = border_value_f[3];
    float* xform_ptr = static_cast<float*>(xform_buf_info.ptr);

    // printf("border_val_f4.x: %f\n", border_val_f4.x);
    // printf("xform: %f\n", xform_ptr[0]);
    // printf("flags: %d\n", flags);
    // printf("border_mode: %d\n", border_mode);

    cudaMemcpyAsync(img_ptr_cuda_src_, static_cast<uint8_t*>(src_buf_info.ptr), img_size_, cudaMemcpyHostToDevice,
                    stream_);
    auto ret = warp_perspective_.Operator(img_data_src, img_data_dest, xform_ptr, flags,
                                          static_cast<lad::lcv::BorderType>(border_mode), border_val_f4, stream_);
    cudaMemcpyAsync(static_cast<uint8_t*>(dest_buf_info.ptr), img_ptr_cuda_dest_, img_size_, cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);
    return static_cast<int>(ret);
  }

  ~PyWarpPerspective() {
    cudaFree(img_ptr_cuda_src_);
    cudaFree(img_ptr_cuda_dest_);
  }
};

PYBIND11_MODULE(pylcv, m) {
  py::class_<PyRemap>(m, "Remap", py::buffer_protocol())
      .def(py::init<size_t, size_t, const std::string&>())
      .def("Operator", &PyRemap::Operator, "Operator", py::arg("img_src"), py::arg("img_dest"), py::arg("map_x"),
           py::arg("map_y"));

  py::class_<PyWarpPerspective>(m, "WarpPerspective", py::buffer_protocol())
      .def(py::init<size_t, size_t, const std::string&>())
      .def("Operator", &PyWarpPerspective::Operator, "Operator", py::arg("img_src"), py::arg("img_dest"),
           py::arg("xform"), py::arg("flags"), py::arg("borderMode"), py::arg("borderValue"));
}

}  // namespace