/******************************************************************************
 *****************************************************************************/
#pragma once
#include <ostream>
#include "types.h"

namespace lad {
namespace lcv {
namespace cuda {

enum DataType {
  kCV_8U = 0,
  kCV_8S = 1,
  kCV_16U = 2,
  kCV_16S = 3,
  kCV_32S = 4,
  kCV_32F = 5,
  kCV_64F = 6,
  kCV_16F = 7,
};

struct DataShape {
  DataShape() : N(1), C(0), H(0), W(0){};
  DataShape(int32_t n, int32_t c, int32_t h, int32_t w) : N(n), C(c), H(h), W(w){};
  DataShape(int32_t c, int32_t h, int32_t w) : N(1), C(c), H(h), W(w){};

  bool operator==(const DataShape &s) { return s.N == N && s.H == H && s.W == W && s.C == C; }

  bool operator!=(const DataShape &s) { return !(*this == s); }

  friend std::ostream &operator<<(std::ostream &out, const DataShape &s) {
    out << "(N = " << s.N << ", H = " << s.H << ", W = " << s.W << ", C = " << s.C << ")";
    return out;
  }

  int32_t N = 1;  // batch
  int32_t C;      // channel
  int32_t H;      // height
  int32_t W;      // width
};

class TensorData {
 public:
  TensorData(int32_t channel, int32_t height, int32_t width, DataType data_type, uint8_t *data_ptr)
      : data_shape_(1, channel, height, width), data_type_(data_type), base_ptr_(data_ptr) {
    /* data_shape_.N = 1;
    data_shape_.C = channel;
    data_shape_.H = image.height;
    data_shape_.W = image.row_stride;

    data_type_ = kCV_8U;
    base_ptr_ = image.base_ptr; */
  }

  const DataShape &GetDataShape() const { return data_shape_; }

  DataType GetDataType() const { return data_type_; }

  uint8_t *GetBasePtr() const { return base_ptr_; }

 private:
  DataShape data_shape_;
  DataType data_type_;
  uint8_t *base_ptr_;
};

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
