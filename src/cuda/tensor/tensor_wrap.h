/******************************************************************************
 *****************************************************************************/

/**
 * @file tensor_wrap.h
 *
 * @brief Defines N-D tensor wrapper with N pitches in bytes divided in compile- and run-time pitches.
 */

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <utility>

namespace lad {
namespace lcv {
namespace cuda {

/**
 * @defgroup NVCV_CPP_CUDATOOLS_TENSORWRAP TensorWrap classes
 * @{
 */

/**
 * TensorWrap class is a non-owning wrap of a N-D tensor used for easy access of its elements in CUDA device.
 *
 * TensorWrap is a wrapper of a multi-dimensional tensor that can have one or more of its N dimension strides, or
 * pitches, defined either at compile-time or at run-time.  Each pitch in \p Strides represents the offset in bytes
 * as a compile-time template parameter that will be applied from the first (slowest changing) dimension to the
 * last (fastest changing) dimension of the tensor, in that order.  Each dimension with run-time pitch is specified
 * as -1 in the \p Strides template parameter.
 *
 * Template arguments:
 * - T type of the values inside the tensor
 * - Strides sequence of compile- or run-time pitches (-1 indicates run-time)
 *   - Y compile-time pitches
 *   - X run-time pitches
 *   - N dimensions, where N = X + Y
 *
 * For example, in the code below a wrap is defined for an NHWC 4D tensor where each sample image in N has a
 * run-time image pitch (first -1 in template argument), and each row in H has a run-time row pitch (second -1), a
 * pixel in W has a compile-time constant pitch as the size of the pixel type and a channel in C has also a
 * compile-time constant pitch as the size of the channel type.
 *
 * @code
 * using DataType = ...;
 * using ChannelType = BaseType<DataType>;
 * using TensorWrap = TensorWrap<ChannelType, -1, -1, sizeof(DataType), sizeof(ChannelType)>;
 * void *imageData = ...;
 * int imgStride = ...;
 * int rowStride = ...;
 * TensorWrap tensorWrap(imageData, imgStride, rowStride);
 * // Elements may be accessed via operator[] using an int4 argument.  They can also be accessed via pointer using
 * // the ptr method with up to 4 integer arguments.
 * @endcode
 *
 * @sa NVCV_CPP_CUDATOOLS_TENSORWRAPS
 *
 * @tparam T Type (it can be const) of each element inside the tensor wrapper.
 * @tparam Strides Each compile-time (use -1 for run-time) pitch in bytes from first to last dimension.
 */
template <typename T, int... Strides>
class TensorWrap;

template <typename T, int... Strides>
class TensorWrap<const T, Strides...> {
  // static_assert(HasTypeTraits<T>, "TensorWrap<T> can only be used if T has type traits");

 public:
  using ValueType = const T;

  static constexpr int kNumDimensions = sizeof...(Strides);
  // static constexpr int kVariableStrides = ((Strides == -1) + ...);
  static constexpr int kVariableStrides = kNumDimensions > 1 ? (kNumDimensions - 1) : 0;
  static constexpr int kConstantStrides = kNumDimensions - kVariableStrides;

  TensorWrap() = default;

  /**
   * Constructs a constant TensorWrap by wrapping a const \p data pointer argument.
   *
   * @param[in] data Pointer to the data that will be wrapped
   * @param[in] strides0..D Each run-time pitch in bytes from first to last dimension
   */
  template <typename... Args>
  explicit __host__ __device__ TensorWrap(const void *data, Args... strides)
      : m_data(data), m_strides{std::forward<int>(strides)...} {
    // static_assert(std::conjunction<std::is_same<int, Args>...>::value, "std::conjunction<std::is_same<int,
    // Args>...>::value"); printf("sizeof...(Args) %d, kVariableStrides %d\n", sizeof...(Args), kVariableStrides);
    // static_assert(sizeof...(Args) == kVariableStrides, "sizeof...(Args) == kVariableStrides");
  }

  /**
   * Get run-time pitch in bytes.
   *
   * @return The const array (as a pointer) containing run-time pitches in bytes.
   */
  __host__ __device__ const int *strides() const { return m_strides; }

  /**
   * Subscript operator for read-only access.
   *
   * @param[in] c 1D coordinate (x first dimension) to be accessed
   *
   * @return Accessed const reference
   */
  inline const __host__ __device__ T &operator[](int1 c) const { return *doGetPtr(c.x); }

  /**
   * Subscript operator for read-only access.
   *
   * @param[in] c 2D coordinates (y first and x second dimension) to be accessed
   *
   * @return Accessed const reference
   */
  inline const __host__ __device__ T &operator[](int2 c) const { return *doGetPtr(c.y, c.x); }

  /**
   * Subscript operator for read-only access.
   *
   * @param[in] c 3D coordinates (z first, y second and x third dimension) to be accessed
   *
   * @return Accessed const reference
   */
  inline const __host__ __device__ T &operator[](int3 c) const { return *doGetPtr(c.z, c.y, c.x); }

  /**
   * Subscript operator for read-only access.
   *
   * @param[in] c 4D coordinates (w first, z second, y third, and x fourth dimension) to be accessed
   *
   * @return Accessed const reference
   */
  inline const __host__ __device__ T &operator[](int4 c) const { return *doGetPtr(c.w, c.z, c.y, c.x); }

  /**
   * Get a read-only proxy (as pointer) at the Dth dimension.
   *
   * @param[in] c0..D Each coordinate from first to last dimension
   *
   * @return The const pointer to the beginning of the Dth dimension
   */
  template <typename... Args>
  inline const __host__ __device__ T *ptr(Args... c) const {
    return doGetPtr(c...);
  }

 protected:
  template <typename... Args>
  inline const __host__ __device__ T *doGetPtr(Args... c) const {
    // static_assert(std::conjunction<std::is_same<int, Args>...>::value);
    static_assert(sizeof...(Args) <= kNumDimensions, "sizeof...(Args) <= kNumDimensions");

    constexpr int kArgSize = sizeof...(Args);
    constexpr int kVarSize = kArgSize < kVariableStrides ? kArgSize : kVariableStrides;
    constexpr int kDimSize = kArgSize < kNumDimensions ? kArgSize : kNumDimensions;
    constexpr int kStride[] = {std::forward<int>(Strides)...};

    int coords[] = {std::forward<int>(c)...};
    // printf("kNumDimensions %d\n", kNumDimensions);
    // printf("kArgSize %d kVarSize %d, kDimSize %d, kStride[0] %d, kStride[1] %d, kStride[2] %d, kStride[3] %d\n",
    // kArgSize, kVarSize, kDimSize, kStride[0], kStride[1], kStride[2], kStride[3]); printf("coords[0] %d coords[1] %d,
    // coords[2] %d, coords[3] %d\n", coords[0], coords[1], coords[2], coords[3]); printf("m_strides[0] %d m_strides[1]
    // %d, m_strides[2] %d\n", m_strides[0], m_strides[1], m_strides[2]);

    // Computing offset first potentially postpones or avoids 64-bit math during addressing
    int offset = 0;
#pragma unroll
    for (int i = 0; i < kVarSize; ++i) {
      offset += coords[i] * m_strides[i];
    }
#pragma unroll
    for (int i = kVariableStrides; i < kDimSize; ++i) {
      offset += coords[i] * kStride[i];
    }
    return reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(m_data) + offset);
  }

 private:
  const void *m_data = nullptr;
  int m_strides[kVariableStrides] = {};
};

/**
 * Tensor wrapper class specialized for non-constant value type.
 *
 * @tparam T Type (non-const) of each element inside the tensor wrapper.
 * @tparam Strides Each compile-time (use -1 for run-time) pitch in bytes from first to last dimension.
 */
template <typename T, int... Strides>
class TensorWrap : public TensorWrap<const T, Strides...> {
  using Base = TensorWrap<const T, Strides...>;

 public:
  using ValueType = T;

  using Base::kConstantStrides;
  using Base::kNumDimensions;
  using Base::kVariableStrides;

  TensorWrap() = default;

  /**
   * Constructs a TensorWrap by wrapping a \p data pointer argument.
   *
   * @param[in] data Pointer to the data that will be wrapped
   * @param[in] strides0..N Each run-time pitch in bytes from first to last dimension
   */
  template <typename... Args>
  explicit __host__ __device__ TensorWrap(void *data, Args... strides) : Base(data, strides...) {}

  /**
   * Subscript operator for read-and-write access.
   *
   * @param[in] c 1D coordinate (x first dimension) to be accessed
   *
   * @return Accessed reference
   */
  inline __host__ __device__ T &operator[](int1 c) const { return *doGetPtr(c.x); }

  /**
   * Subscript operator for read-and-write access.
   *
   * @param[in] c 2D coordinates (y first and x second dimension) to be accessed
   *
   * @return Accessed reference
   */
  inline __host__ __device__ T &operator[](int2 c) const { return *doGetPtr(c.y, c.x); }

  /**
   * Subscript operator for read-and-write access.
   *
   * @param[in] c 3D coordinates (z first, y second and x third dimension) to be accessed
   *
   * @return Accessed reference
   */
  inline __host__ __device__ T &operator[](int3 c) const { return *doGetPtr(c.z, c.y, c.x); }

  /**
   * Subscript operator for read-and-write access.
   *
   * @param[in] c 4D coordinates (w first, z second, y third, and x fourth dimension) to be accessed
   *
   * @return Accessed reference
   */
  inline __host__ __device__ T &operator[](int4 c) const { return *doGetPtr(c.w, c.z, c.y, c.x); }

  /**
   * Get a read-and-write proxy (as pointer) at the Dth dimension.
   *
   * @param[in] c0..D Each coordinate from first to last dimension
   *
   * @return The pointer to the beginning of the Dth dimension
   */
  template <typename... Args>
  inline __host__ __device__ T *ptr(Args... c) const {
    return doGetPtr(c...);
  }

 protected:
  template <typename... Args>
  inline __host__ __device__ T *doGetPtr(Args... c) const {
    // The const_cast here is the *only* place where it is used to remove the base pointer constness
    return const_cast<T *>(Base::doGetPtr(c...));
  }
};

/**@}*/

/**
 *  Specializes \ref TensorWrap template classes to different dimensions.
 *
 *  The specializations have the last dimension as the only compile-time dimension as size of T.  All other
 *  dimensions have run-time pitch and must be provided.
 *
 *  Template arguments:
 *  - T data type of each element in \ref TensorWrap
 *
 *  @sa NVCV_CPP_CUDATOOLS_TENSORWRAP
 *
 *  @defgroup NVCV_CPP_CUDATOOLS_TENSORWRAPS TensorWrap shortcuts
 *  @{
 */

template <typename T>
using Tensor1DWrap = TensorWrap<T, sizeof(T)>;

template <typename T>
using Tensor2DWrap = TensorWrap<T, -1, sizeof(T)>;

template <typename T>
using Tensor3DWrap = TensorWrap<T, -1, -1, sizeof(T)>;

template <typename T>
using Tensor4DWrap = TensorWrap<T, -1, -1, -1, sizeof(T)>;

/**@}*/

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
