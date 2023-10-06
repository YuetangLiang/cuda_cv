/******************************************************************************
 *****************************************************************************/
#pragma once

namespace lad {
namespace lcv {
namespace cuda {
template <typename T>
struct PackedTensor3D {  // PackedTensor3D extends std::array in three dimensions
  int stride1 = 0;
  int stride2 = 0;

  PackedTensor3D(int height, int width) {
    stride2 = width * sizeof(T);
    stride1 = height * stride2;
  };
};

template <typename T>
struct PackedTensor4D {  // PackedTensor4D extends std::array in four dimensions
  int stride1 = 0;
  int stride2 = 0;
  int stride3 = 0;

  PackedTensor4D(int height, int width, int channel) {
    stride3 = channel * sizeof(T);
    stride2 = width * stride3;
    stride1 = height * stride2;
  };
};

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
