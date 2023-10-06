
/******************************************************************************
 *****************************************************************************/
#include "cvt_color.h"
#include <stdio.h>
#include "lcv_cuda.h"
namespace lad {
namespace lcv {

CvtColor::CvtColor(/* args */) { ; }
CvtColor::~CvtColor() { ; }
ErrorCode CvtColor::Operator(const ImageData &src, const ImageData &dst, const ColorCVTCode code, cudaStream_t stream) {
  ImageFormat input_format = src.format;
  ImageFormat output_format = dst.format;

  if (!(input_format == kNV12_8U || input_format == kIYUV_8U) ||
      !(output_format == kBGR_8U || output_format == kRGB_8U || output_format == kIYUV_8U)) {
    // LOG_ERROR("Invalid DataFormat " << format);
    printf("Invalid DataFormat\n");
    return ErrorCode::INVALID_DATA_FORMAT;
  }

  int src_channel = 1;
  int dst_channel = 1;
  cuda::DataType data_type;

  switch (code) {
    case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2BGRA_NV12:
    case COLOR_YUV2BGRA_NV21:
    case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGR_YV12:
    case COLOR_YUV2BGR_IYUV:
    case COLOR_YUV2BGRA_YV12:
    case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2RGB_IYUV:
    case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2RGBA_IYUV: {
      src_channel = 1;
      dst_channel = 3;
      data_type = cuda::kCV_8U;
    } break;

    case COLOR_NV122YU12: {
      src_channel = 1;
      dst_channel = 1;
      data_type = cuda::kCV_8U;
    } break;

    default:
      break;
  }

  cuda::TensorData cu_src(src_channel, src.height, src.row_stride, data_type, src.base_ptr);
  cuda::TensorData cu_dst(dst_channel, dst.height, dst.row_stride, data_type, dst.base_ptr);

  cuda::CvtColor cvt_color;
  cvt_color.Infer(cu_src, cu_dst, code, stream);
  return SUCCESS;
}
}  // namespace lcv
}  // namespace lad