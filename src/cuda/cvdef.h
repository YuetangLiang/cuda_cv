/******************************************************************************
 *****************************************************************************/

/**
 * @file cvdef.h
 *
 * @brief Define relevant types for internal use
 */

#pragma once
#include <cstddef>

namespace lad {
namespace lcv {
namespace cuda {

typedef unsigned char uchar;
typedef signed char schar;
#define get_batch_idx() (blockIdx.z)
#define get_lid()       (threadIdx.y * blockDim.x + threadIdx.x)

#ifndef checkKernelErrors
#define checkKernelErrors(expr)                                                         \
  do {                                                                                  \
    expr;                                                                               \
                                                                                        \
    cudaError_t __err = cudaGetLastError();                                             \
    if (__err != cudaSuccess) {                                                         \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
      abort();                                                                          \
    }                                                                                   \
  } while (0)
#endif

#if 0
struct uchar3 {
  unsigned char x, y, z;

  uchar3 operator*(float multiplier) {
    uchar3 result;
    result.x = (unsigned char)(multiplier * x);
    result.y = (unsigned char)(multiplier * y);
    result.z = (unsigned char)(multiplier * z);

    return result;
  }

  uchar3 operator+(const uchar3 &add_num) {
    uchar3 result;

    result.x = add_num.x + x;
    result.y = add_num.y + y;
    result.z = add_num.z + z;

    return result;
  }

};
#endif

typedef union Cv32suf {
  int i;
  unsigned u;
  float f;
} Cv32suf;

class float16_t {
 public:
#if CV_FP16_TYPE

  float16_t() : h(0) {}
  explicit float16_t(float x) { h = (__fp16)x; }
  operator float() const { return (float)h; }
  static float16_t fromBits(ushort w) {
    Cv16suf u;
    u.u = w;
    float16_t result;
    result.h = u.h;
    return result;
  }
  static float16_t zero() {
    float16_t result;
    result.h = (__fp16)0;
    return result;
  }
  ushort bits() const {
    Cv16suf u;
    u.h = h;
    return u.u;
  }

 protected:
  __fp16 h;

#else
  float16_t() : w(0) {}
  explicit float16_t(float x) {
#if CV_FP16
    __m128 v = _mm_load_ss(&x);
    w = (ushort)_mm_cvtsi128_si32(_mm_cvtps_ph(v, 0));
#else
    Cv32suf in;
    in.f = x;
    unsigned sign = in.u & 0x80000000;
    in.u ^= sign;

    if (in.u >= 0x47800000)
      w = (ushort)(in.u > 0x7f800000 ? 0x7e00 : 0x7c00);
    else {
      if (in.u < 0x38800000) {
        in.f += 0.5f;
        w = (ushort)(in.u - 0x3f000000);
      } else {
        unsigned t = in.u + 0xc8000fff;
        w = (ushort)((t + ((in.u >> 13) & 1)) >> 13);
      }
    }

    w = (ushort)(w | (sign >> 16));
#endif
  }

  operator float() const {
#if CV_FP16
    float f;
    _mm_store_ss(&f, _mm_cvtph_ps(_mm_cvtsi32_si128(w)));
    return f;
#else
    Cv32suf out;

    unsigned t = ((w & 0x7fff) << 13) + 0x38000000;
    unsigned sign = (w & 0x8000) << 16;
    unsigned e = w & 0x7c00;

    out.u = t + (1 << 23);
    out.u = (e >= 0x7c00 ? t + 0x38000000 : e == 0 ? (static_cast<void>(out.f -= 6.103515625e-05f), out.u) : t) | sign;
    return out.f;
#endif
  }

  static float16_t fromBits(ushort b) {
    float16_t result;
    result.w = b;
    return result;
  }
  static float16_t zero() {
    float16_t result;
    result.w = (ushort)0;
    return result;
  }
  ushort bits() const { return w; }

 protected:
  ushort w;

#endif
};

template<class T> // base type
__host__ __device__ int32_t CalcNCHWImageStride(int rows, int cols, int channels)
{
    return rows * cols * channels * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNCHWRowStride(int cols, int channels)
{
    return cols * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNHWCImageStride(int rows, int cols, int channels)
{
    return rows * cols * channels * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNHWCRowStride(int cols, int channels)
{
    return cols * channels * sizeof(T);
}

// Used to disambiguate between the constructors that accept legacy memory buffers,
// and the ones that accept the new ones. Just pass NewAPI as first parameter.
struct NewAPITag
{
};

constexpr NewAPITag NewAPI = {};

template<typename T>
struct Ptr2dNHWC
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dNHWC()
        : batches(0)
        , rows(0)
        , cols(0)
        , imgStride(0)
        , rowStride(0)
        , ch(0)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int rows_, int cols_, int ch_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(0)
        , rowStride(CalcNHWCRowStride<T>(cols_, ch_))
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(CalcNHWCImageStride<T>(rows_, cols_, ch_))
        , rowStride(CalcNHWCRowStride<T>(cols_, ch_))
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(NewAPITag, int rows_, int cols_, int ch_, int rowStride_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgStride(0)
        , rowStride(rowStride_)
        , data(data_)
    {
    }

    // ptr for uchar1/3/4, ushort1/3/4, float1/3/4, typename T -> uchar3 etc.
    // each fetch operation get a x-channel elements
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x)
    {
        return (T *)(data + b * rows * cols + y * cols + x);
        //return (T *)(reinterpret_cast<std::byte *>(data) + b * imgStride + y * rowStride + x * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x) const
    {
        return (const T *)(data + b * rows * cols + y * cols + x);
        //return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgStride + y * rowStride + x * sizeof(T));
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        return (T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
        //return (T *)(reinterpret_cast<std::byte *>(data) + b * imgStride + y * rowStride + (x * ch + c) * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        return (const T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
        //return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgStride + y * rowStride
                          // + (x * ch + c) * sizeof(T));
    }

    __host__ __device__ __forceinline__ int at_rows(int b)
    {
        return rows;
    }

    __host__ __device__ __forceinline__ int at_rows(int b) const
    {
        return rows;
    }

    __host__ __device__ __forceinline__ int at_cols(int b)
    {
        return cols;
    }

    __host__ __device__ __forceinline__ int at_cols(int b) const
    {
        return cols;
    }

    int batches;
    int rows;
    int cols;
    int ch;
    int imgStride;
    int rowStride;
    T  *data;
};

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
