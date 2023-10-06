
/******************************************************************************
 *****************************************************************************/
#include "warp.h"
#include "lcv_cuda.h"

namespace lad {
namespace lcv {

WarpPerspective::WarpPerspective(/* args */) { ; }
WarpPerspective::~WarpPerspective() { ; }
ErrorCode WarpPerspective::Operator(const ImageData &src, const ImageData &dst, const float *xform, const int32_t flags,
                                    const BorderType borderMode, const float4 borderValue, cudaStream_t stream) {
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

  const int interpolation = flags & INTERP_MAX;
  int src_channel = 3;
  int dst_channel = 3;
  cuda::DataType data_type = cuda::kCV_8U;

  cuda::TensorData cu_src(src_channel, src.height, src.row_stride, data_type, src.base_ptr);
  cuda::TensorData cu_dst(dst_channel, dst.height, dst.row_stride, data_type, dst.base_ptr);

  cuda::WarpPerspective warp;
  warp.Infer(cu_src, cu_dst, xform, interpolation, borderMode, borderValue, stream);

  return SUCCESS;
}
}  // namespace lcv
}  // namespace lad