/******************************************************************************
 *****************************************************************************/
#include "remap.h"
#include "lcv_cuda.h"

namespace lad {
namespace lcv {

Remap::Remap(/* args */) { ; }
Remap::~Remap() { ; }
ErrorCode Remap::Operator(const ImageData& src, const ImageData& dst, const float* map_x, const float* map_y,
                          cudaStream_t stream) {
  ImageFormat input_format = src.format;
  ImageFormat output_format = dst.format;

  if (input_format != output_format) {
    return ErrorCode::INVALID_DATA_FORMAT;
  }

  if (!(input_format == kNV12_8U || input_format == kBGR_8U || input_format == kRGB_8U)) {
    printf("Invalid DataFormat\n");
    return ErrorCode::INVALID_DATA_FORMAT;
  }

  int src_channel = 3;
  int dst_channel = 3;
  cuda::DataType data_type = cuda::kCV_8U;

  cuda::TensorData cu_src(src_channel, src.height, src.row_stride, data_type, src.base_ptr);
  cuda::TensorData cu_dst(dst_channel, dst.height, dst.row_stride, data_type, dst.base_ptr);

  cuda::Remap remap;
  remap.Infer(cu_src, cu_dst, map_x, map_y, stream);

  return SUCCESS;
}
}  // namespace lcv
}  // namespace lad