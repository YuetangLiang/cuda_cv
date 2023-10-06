/******************************************************************************
 *****************************************************************************/
#pragma once
#include "utils/type_traits.h"
#include "math/saturate.h"

namespace lad {
namespace lcv {
namespace cuda {

struct PerspectiveTransform {
  PerspectiveTransform(const float *transMatrix) {
    xform[0] = transMatrix[0];
    xform[1] = transMatrix[1];
    xform[2] = transMatrix[2];
    xform[3] = transMatrix[3];
    xform[4] = transMatrix[4];
    xform[5] = transMatrix[5];
    xform[6] = transMatrix[6];
    xform[7] = transMatrix[7];
    xform[8] = transMatrix[8];
  }

  static __device__ __forceinline__ float2 calcCoord(const float *c_warpMat, int x, int y) {
    const double coeff = (double)1.0 / ((double)c_warpMat[6] * x + (double)c_warpMat[7] * y + c_warpMat[8]);

    const float xcoo = coeff * ((double)c_warpMat[0] * x + (double)c_warpMat[1] * y + c_warpMat[2]);
    const float ycoo = coeff * ((double)c_warpMat[3] * x + (double)c_warpMat[4] * y + c_warpMat[5]);

    return make_float2(xcoo, ycoo);
  }

  float xform[9];
};

template <typename D>
struct BrdConstant {
  typedef D result_type;

  __host__ __device__ __forceinline__ BrdConstant(int height_, int width_, const D &val_ = SetAll<D>(0))
      : height(height_), width(width_), val(val_) {}

  template <typename Ptr2D>
  __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const {
    /* return ((float)x >= 0 && x < src.at_cols(b) && (float)y >= 0 && y < src.at_rows(b))
               //? SaturateCast<BaseType<D>>(*src.ptr(b, y, x))
               //? (*src.ptr(b, y, x))
               ? val
               : val; */

    printf("00 BrdConstant\n");

    if ((float)x >= 0 && x < src.at_cols(b) && (float)y >= 0 && y < src.at_rows(b)) {
      D tmp;
      for (int e = 0; e < NumElements<D>; ++e) {
        (reinterpret_cast<BaseType<D> *>(&tmp))[e] =
            saturate_cast<BaseType<D>>(reinterpret_cast<const uchar *>(&(*src.ptr(b, y, x)))[e]);
      }

      return tmp;
    } else {
      return val;
    }
  }

  int height;
  int width;
  D val;
};

template <typename D>
struct BrdReplicate {
  typedef D result_type;

  __host__ __device__ __forceinline__ BrdReplicate(int height, int width) : last_row(height - 1), last_col(width - 1) {}

  template <typename U>
  __host__ __device__ __forceinline__ BrdReplicate(int height, int width, U)
      : last_row(height - 1), last_col(width - 1) {}

  __device__ __forceinline__ int idx_row_low(int y) const { return math::max(y, 0); }

  __device__ __forceinline__ int idx_row_high(int y, int last_row_) const { return math::min(y, last_row_); }

  __device__ __forceinline__ int idx_row(int y, int last_row_) const { return idx_row_low(idx_row_high(y, last_row_)); }

  __device__ __forceinline__ int idx_col_low(int x) const { return math::max(x, 0); }

  __device__ __forceinline__ int idx_col_high(int x, int last_col_) const { return math::min(x, last_col_); }

  __device__ __forceinline__ int idx_col(int x, int last_col_) const { return idx_col_low(idx_col_high(x, last_col_)); }

  template <typename Ptr2D>
  __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const {
    // return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(
    return (*src.ptr(b, idx_row(y, src.at_rows(b) - 1), idx_col(x, src.at_cols(b) - 1)));
  }

  int last_row;
  int last_col;
};

template <typename D>
struct BrdReflect101 {
  typedef D result_type;

  __host__ __device__ __forceinline__ BrdReflect101(int height, int width)
      : last_row(height - 1), last_col(width - 1) {}

  template <typename U>
  __host__ __device__ __forceinline__ BrdReflect101(int height, int width, U)
      : last_row(height - 1), last_col(width - 1) {}

  __device__ __forceinline__ int idx_row_low(int y, int last_row_) const { return ::abs(y) % (last_row_ + 1); }

  __device__ __forceinline__ int idx_row_high(int y, int last_row_) const {
    return ::abs(last_row_ - ::abs(last_row_ - y)) % (last_row_ + 1);
  }

  __device__ __forceinline__ int idx_row(int y, int last_row_) const {
    return idx_row_low(idx_row_high(y, last_row_), last_row_);
  }

  __device__ __forceinline__ int idx_col_low(int x, int last_col_) const { return ::abs(x) % (last_col_ + 1); }

  __device__ __forceinline__ int idx_col_high(int x, int last_col_) const {
    return ::abs(last_col_ - ::abs(last_col_ - x)) % (last_col_ + 1);
  }

  __device__ __forceinline__ int idx_col(int x, int last_col_) const {
    return idx_col_low(idx_col_high(x, last_col_), last_col_);
  }

  template <typename Ptr2D>
  __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const {
    // return nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<D>>(
    return (*src.ptr(b, idx_row(y, src.at_rows(b) - 1), idx_col(x, src.at_cols(b) - 1)));
  }

  int last_row;
  int last_col;
};

template <typename D>
struct BrdReflect {
  typedef D result_type;

  __host__ __device__ __forceinline__ BrdReflect(int height, int width) : last_row(height - 1), last_col(width - 1) {}

  template <typename U>
  __host__ __device__ __forceinline__ BrdReflect(int height, int width, U)
      : last_row(height - 1), last_col(width - 1) {}

  __device__ __forceinline__ int idx_row_low(int y, int last_row_) const {
    return (::abs(y) - (y < 0)) % (last_row_ + 1);
  }

  __device__ __forceinline__ int idx_row_high(int y, int last_row_) const {
    return /*::abs*/ (last_row_ - ::abs(last_row_ - y) + (y > last_row_)) /*% (last_row + 1)*/;
  }

  __device__ __forceinline__ int idx_row(int y, int last_row_) const {
    return idx_row_low(idx_row_high(y, last_row_), last_row_);
  }

  __device__ __forceinline__ int idx_col_low(int x, int last_col_) const {
    return (::abs(x) - (x < 0)) % (last_col_ + 1);
  }

  __device__ __forceinline__ int idx_col_high(int x, int last_col_) const {
    return (last_col_ - ::abs(last_col_ - x) + (x > last_col_));
  }

  __device__ __forceinline__ int idx_col(int x, int last_col_) const {
    return idx_col_low(idx_col_high(x, last_col_), last_col_);
  }

  template <typename Ptr2D>
  __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const {
    // return SaturateCast<BaseType<D>>(
    return (*src.ptr(b, idx_row(y, src.at_rows(b) - 1), idx_col(x, src.at_cols(b) - 1)));
  }

  int last_row;
  int last_col;
};

template <typename D>
struct BrdWrap {
  typedef D result_type;

  __host__ __device__ __forceinline__ BrdWrap(int height_, int width_) : height(height_), width(width_) {}

  template <typename U>
  __host__ __device__ __forceinline__ BrdWrap(int height_, int width_, U) : height(height_), width(width_) {}

  __device__ __forceinline__ int idx_row_low(int y, int height_) const {
    return (y >= 0) ? y : (y - ((y - height_ + 1) / height_) * height_);
  }

  __device__ __forceinline__ int idx_row_high(int y, int height_) const { return (y < height_) ? y : (y % height_); }

  __device__ __forceinline__ int idx_row(int y, int height_) const {
    return idx_row_high(idx_row_low(y, height_), height_);
  }

  __device__ __forceinline__ int idx_col_low(int x, int width_) const {
    return (x >= 0) ? x : (x - ((x - width_ + 1) / width_) * width_);
  }

  __device__ __forceinline__ int idx_col_high(int x, int width_) const { return (x < width_) ? x : (x % width_); }

  __device__ __forceinline__ int idx_col(int x, int width_) const {
    return idx_col_high(idx_col_low(x, width_), width_);
  }

  template <typename Ptr2D>
  __device__ __forceinline__ D at(int b, int y, int x, const Ptr2D &src) const {
    // return SaturateCast<BaseType<D>>(
    return (*src.ptr(b, idx_row(y, src.at_rows(b)), idx_col(x, src.at_cols(b))));
  }

  int height;
  int width;
};

template <typename Ptr2D, typename B>
struct BorderReader {
  typedef typename B::result_type elem_type;

  __host__ __device__ __forceinline__ BorderReader(const Ptr2D &ptr_, const B &b_) : ptr(ptr_), b(b_) {}

  __device__ __forceinline__ elem_type operator()(int bidx, int y, int x) const { return b.at(bidx, y, x, ptr); }

  __host__ __device__ __forceinline__ int at_rows(int b) { return ptr.at_rows(b); }

  __host__ __device__ __forceinline__ int at_rows(int b) const { return ptr.at_rows(b); }

  __host__ __device__ __forceinline__ int at_cols(int b) { return ptr.at_cols(b); }

  __host__ __device__ __forceinline__ int at_cols(int b) const { return ptr.at_cols(b); }

  Ptr2D ptr;
  B b;
};

template <typename Ptr2D, typename D>
struct BorderReader<Ptr2D, BrdConstant<D>> {
  typedef typename BrdConstant<D>::result_type elem_type;

  __host__ __device__ __forceinline__ BorderReader(const Ptr2D &ptr_, const BrdConstant<D> &b)
      : ptr(ptr_), height(b.height), width(b.width), val(b.val) {}

  __device__ __forceinline__ D operator()(int bidx, int y, int x) const {
    /* return ((float)x >= 0 && x < ptr.at_cols(bidx) && (float)y >= 0 && y < ptr.at_rows(bidx))
               //? SaturateCast<BaseType<D>>(*ptr.ptr(bidx, y, x))
               //? *(reinterpret_cast<const D *>(ptr.ptr(bidx, y, x)))
               : val; */

    if ((float)x >= 0 && x < ptr.at_cols(bidx) && (float)y >= 0 && y < ptr.at_rows(bidx)) {
      D tmp;
      for (int e = 0; e < NumElements<D>; ++e) {
        /* (reinterpret_cast<BaseType<D> *>(&tmp))[e] =
            static_cast<BaseType<D>>((reinterpret_cast<const uchar *>(&(*ptr.ptr(bidx, y, x))))[e]); */
        auto p = &((reinterpret_cast<BaseType<D> *>(&tmp))[e]);
        *p = static_cast<BaseType<D>>((reinterpret_cast<const uchar *>(&(*ptr.ptr(bidx, y, x))))[e]);
        //printf("((reinterpret_cast<BaseType<D> *>(&tmp))[e]) = %f, *p = %f, (reinterpret_cast<const uchar *>(&(*ptr.ptr(bidx, y, x))))[e] = %u\n", ((reinterpret_cast<BaseType<D> *>(&tmp))[e]), *p, (reinterpret_cast<const uchar *>(&(*ptr.ptr(bidx, y, x))))[e]);
        //printf("(*ptr.ptr(bidx, y, x).x = %u, (*ptr.ptr(bidx, y, x).y = %u, (*ptr.ptr(bidx, y, x).z = %u\n", (*ptr.ptr(bidx, y, x)).x, (*ptr.ptr(bidx, y, x)).y, (*ptr.ptr(bidx, y, x)).z);
      }
      
      //printf("tmp.x = %f, tmp.y = %f, tmp.z = %f\n", tmp.x, tmp.y, tmp.z);
      return tmp;
    } else {
      return val;
    }
  }

  __host__ __device__ __forceinline__ int at_rows(int b) { return ptr.at_rows(b); }

  __host__ __device__ __forceinline__ int at_rows(int b) const { return ptr.at_rows(b); }

  __host__ __device__ __forceinline__ int at_cols(int b) { return ptr.at_cols(b); }

  __host__ __device__ __forceinline__ int at_cols(int b) const { return ptr.at_cols(b); }

  Ptr2D ptr;
  int height;
  int width;
  D val;
};

template <typename BrdReader>
struct PointFilter {
  typedef typename BrdReader::elem_type elem_type;

  explicit __host__ __device__ __forceinline__ PointFilter(const BrdReader &src_, float fx = 0.f, float fy = 0.f)
      : src(src_) {}

  __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const {
    return src(bidx, __float2int_rz(y), __float2int_rz(x));
  }

  BrdReader src;
};

template <typename BrdReader>
struct LinearFilter {
  typedef typename BrdReader::elem_type elem_type;

  explicit __host__ __device__ __forceinline__ LinearFilter(const BrdReader &src_, float fx = 0.f, float fy = 0.f)
      : src(src_) {}

  __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const {
    using work_type = ConvertBaseTypeTo<float, elem_type>;
    work_type out = SetAll<work_type>(0);

    // to prevent -2147483648 > 0 in border
    // float x_float = x >= std::numeric_limits<int>::max() ? ((float) std::numeric_limits<int>::max() - 1):
    //     ((x <= std::numeric_limits<int>::min() + 1) ? ((float) std::numeric_limits<int>::min() + 1): x);
    // float y_float = y >= std::numeric_limits<int>::max() ? ((float) std::numeric_limits<int>::max() - 1):
    //     ((y <= std::numeric_limits<int>::min() + 1) ? ((float) std::numeric_limits<int>::min() + 1): y);

    const int x1 = __float2int_rd(x);
    const int y1 = __float2int_rd(y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;

    elem_type src_reg = src(bidx, y1, x1);
    out = out + src_reg * ((x2 - x) * (y2 - y));

    src_reg = src(bidx, y1, x2);
    out = out + src_reg * ((x - x1) * (y2 - y));

    src_reg = src(bidx, y2, x1);
    out = out + src_reg * ((x2 - x) * (y - y1));

    src_reg = src(bidx, y2, x2);
    out = out + src_reg * ((x - x1) * (y - y1));

    // return SaturateCast<BaseType<elem_type>>(out);
    return (out);
  }

  BrdReader src;
};

template <typename BrdReader>
struct CubicFilter {
  typedef typename BrdReader::elem_type elem_type;
  using work_type = ConvertBaseTypeTo<float, elem_type>;

  explicit __host__ __device__ __forceinline__ CubicFilter(const BrdReader &src_, float fx = 0.f, float fy = 0.f)
      : src(src_) {}

  static __device__ __forceinline__ float bicubicCoeff(float x_) {
    float x = fabsf(x_);
    if (x <= 1.0f) {
      return x * x * (1.5f * x - 2.5f) + 1.0f;
    } else if (x < 2.0f) {
      return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
    } else {
      return 0.0f;
    }
  }

  __device__ elem_type operator()(int bidx, float y, float x) const {
    const float xmin = ceilf(x - 2.0f);
    const float xmax = floorf(x + 2.0f);

    const float ymin = ceilf(y - 2.0f);
    const float ymax = floorf(y + 2.0f);

    work_type sum = SetAll<work_type>(0);
    float wsum = 0.0f;

    for (float cy = ymin; cy <= ymax; cy += 1.0f) {
      for (float cx = xmin; cx <= xmax; cx += 1.0f) {
        const float w = bicubicCoeff(x - cx) * bicubicCoeff(y - cy);
        sum = sum + w * src(bidx, __float2int_rd(cy), __float2int_rd(cx));
        wsum += w;
      }
    }

    work_type res = (!wsum) ? SetAll<work_type>(0) : sum / wsum;

    // return SaturateCast<BaseType<elem_type>>(res);
    return (res);
  }

  BrdReader src;
};

// for integer scaling
template <typename BrdReader>
struct IntegerAreaFilter {
  typedef typename BrdReader::elem_type elem_type;

  explicit __host__ __device__ __forceinline__ IntegerAreaFilter(const BrdReader &src_, float scale_x_, float scale_y_)
      : src(src_), scale_x(scale_x_), scale_y(scale_y_), scale(1.f / (scale_x * scale_y)) {}

  __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const {
    float fsx1 = x * scale_x;
    float fsx2 = fsx1 + scale_x;

    int sx1 = __float2int_ru(fsx1);
    int sx2 = __float2int_rd(fsx2);

    float fsy1 = y * scale_y;
    float fsy2 = fsy1 + scale_y;

    int sy1 = __float2int_ru(fsy1);
    int sy2 = __float2int_rd(fsy2);

    using work_type = ConvertBaseTypeTo<float, elem_type>;
    work_type out = SetAll<work_type>(0.f);

    for (int dy = sy1; dy < sy2; ++dy)
      for (int dx = sx1; dx < sx2; ++dx) {
        out = out + src(bidx, dy, dx) * scale;
      }

    // return SaturateCast<BaseType<elem_type>>(out);
    return (out);
  }

  BrdReader src;
  float scale_x, scale_y, scale;
};

template <typename BrdReader>
struct AreaFilter {
  typedef typename BrdReader::elem_type elem_type;

  explicit __host__ __device__ __forceinline__ AreaFilter(const BrdReader &src_, float scale_x_, float scale_y_)
      : src(src_), scale_x(scale_x_), scale_y(scale_y_) {}

  __device__ __forceinline__ elem_type operator()(int bidx, float y, float x) const {
    float fsx1 = x * scale_x;
    float fsx2 = fsx1 + scale_x;

    int sx1 = __float2int_ru(fsx1);
    int sx2 = __float2int_rd(fsx2);

    float fsy1 = y * scale_y;
    float fsy2 = fsy1 + scale_y;

    int sy1 = __float2int_ru(fsy1);
    int sy2 = __float2int_rd(fsy2);

    float scale = 1.f / (fminf(scale_x, src.at_cols(bidx) - fsx1) * fminf(scale_y, src.at_rows(bidx) - fsy1));

    using work_type = ConvertBaseTypeTo<float, elem_type>;
    work_type out = SetAll<work_type>(0.f);

    for (int dy = sy1; dy < sy2; ++dy) {
      for (int dx = sx1; dx < sx2; ++dx) out = out + src(bidx, dy, dx) * scale;

      if (sx1 > fsx1) out = out + src(bidx, dy, (sx1 - 1)) * ((sx1 - fsx1) * scale);

      if (sx2 < fsx2) out = out + src(bidx, dy, sx2) * ((fsx2 - sx2) * scale);
    }

    if (sy1 > fsy1)
      for (int dx = sx1; dx < sx2; ++dx) out = out + src(bidx, (sy1 - 1), dx) * ((sy1 - fsy1) * scale);

    if (sy2 < fsy2)
      for (int dx = sx1; dx < sx2; ++dx) out = out + src(bidx, sy2, dx) * ((fsy2 - sy2) * scale);

    if ((sy1 > fsy1) && (sx1 > fsx1))
      out = out + src(bidx, (sy1 - 1), (sx1 - 1)) * ((sy1 - fsy1) * (sx1 - fsx1) * scale);

    if ((sy1 > fsy1) && (sx2 < fsx2)) out = out + src(bidx, (sy1 - 1), sx2) * ((sy1 - fsy1) * (fsx2 - sx2) * scale);

    if ((sy2 < fsy2) && (sx2 < fsx2)) out = out + src(bidx, sy2, sx2) * ((fsy2 - sy2) * (fsx2 - sx2) * scale);

    if ((sy2 < fsy2) && (sx1 > fsx1)) out = out + src(bidx, sy2, (sx1 - 1)) * ((fsy2 - sy2) * (sx1 - fsx1) * scale);

    // return SaturateCast<BaseType<elem_type>>(out);
    return (out);
  }

  BrdReader src;
  float scale_x, scale_y;
  int width, haight;
};

}  // namespace cuda
}  // namespace lcv
}  // namespace lad