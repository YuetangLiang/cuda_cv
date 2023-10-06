/******************************************************************************
 *****************************************************************************/
#pragma once
#include <stdint.h>
namespace lad {
namespace lcv {
enum ImageFormat {
  // NV12
  kNV12_8U,
  kNV12_8S,
  kNV12_16U,
  kNV12_16S,
  kNV12_32S,
  kNV12_32F,
  kNV12_64F,
  kNV12_16F,

  // IYUV(I420)
  kIYUV_8U,
  kIYUV_8S,
  kIYUV_16U,
  kIYUV_16S,
  kIYUV_32S,
  kIYUV_32F,
  kIYUV_64F,
  kIYUV_16F,

  // BGR
  kBGR_8U,
  kBGR_8S,
  kBGR_16U,
  kBGR_16S,
  kBGR_32S,
  kBGR_32F,
  kBGR_64F,
  kBGR_16F,

  // RGB
  kRGB_8U,
  kRGB_8S,
  kRGB_16U,
  kRGB_16S,
  kRGB_32S,
  kRGB_32F,
  kRGB_64F,
  kRGB_16F,
};

struct ImageData {
  ImageFormat format;
  int32_t width;
  int32_t height;
  int32_t row_stride;
  uint8_t *base_ptr;
};

}  // namespace lcv
}  // namespace lad
