
/******************************************************************************
 *****************************************************************************/
#include "undistort.h"
#include "lcv_cuda.h"

namespace lad {
namespace lcv {

UnDistort::UnDistort(/* args */) { ; }
UnDistort::~UnDistort() { ; }
ErrorCode UnDistort::Operator(const ImageData& src, const ImageData& dst, float* const camera_matrix,
                              float* const distCoeffs, const int* dist_len, const int* dist_len_current,
                              cudaStream_t stream) {
  ImageFormat input_format = src.format;
  ImageFormat output_format = dst.format;

  if (input_format != output_format) {
    // LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
    return ErrorCode::INVALID_DATA_FORMAT;
  }

  if (!(input_format == kNV12_8U || input_format == kBGR_8U || input_format == kRGB_8U)) {
    // LOG_ERROR("Invalid DataFormat " << format);
    printf("Invalid DataFormat\n");
    return ErrorCode::INVALID_DATA_FORMAT;
  }

  int src_channel = 3;
  int dst_channel = 3;
  cuda::DataType data_type = cuda::kCV_8U;

  cuda::TensorData cu_src(src_channel, src.height, src.row_stride, data_type, src.base_ptr);
  cuda::TensorData cu_dst(dst_channel, dst.height, dst.row_stride, data_type, dst.base_ptr);

  cuda::UnDistort undistort;
  undistort.Infer(cu_src, cu_dst, camera_matrix, distCoeffs, dist_len, dist_len_current, stream);

  return SUCCESS;
}
}  // namespace lcv
}  // namespace lad