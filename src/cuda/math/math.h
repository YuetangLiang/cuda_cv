/******************************************************************************
 *****************************************************************************/
#pragma once
#include <vector>
#include <assert.h>

namespace lad {
namespace lcv {
namespace cuda {
namespace math {

template <typename _Tp>
_GLIBCXX14_CONSTEXPR inline __host__ __device__ const _Tp& min(const _Tp& __a, const _Tp& __b) {
  // concept requirements
  __glibcxx_function_requires(_LessThanComparableConcept<_Tp>)
      // return __b < __a ? __b : __a;
      if (__b < __a) return __b;
  return __a;
}

template <typename _Tp>
_GLIBCXX14_CONSTEXPR inline __host__ __device__ const _Tp& max(const _Tp& __a, const _Tp& __b) {
  // concept requirements
  __glibcxx_function_requires(_LessThanComparableConcept<_Tp>)
      // return  __a < __b ? __b : __a;
      if (__a < __b) return __b;
  return __a;
}

inline int divUp(int a, int b) {
  // assert(b > 0);
  return ceil((float)a / b);
};

/** @brief Rounds floating-point number to the nearest integer

 @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
 result is not defined.
 */
static inline int cvRound(double value) {
#if defined CV_INLINE_ROUND_DBL
  CV_INLINE_ROUND_DBL(value);
#elif (defined _MSC_VER && defined _M_X64) && !defined(__CUDACC__)
  __m128d t = _mm_set_sd(value);
  return _mm_cvtsd_si32(t);
#elif defined _MSC_VER && defined _M_IX86
  int t;
  __asm
  {
        fld value;
        fistp t;
  }
  return t;
#elif defined CV__FASTMATH_ENABLE_GCC_MATH_BUILTINS || defined CV__FASTMATH_ENABLE_CLANG_MATH_BUILTINS
  return (int)__builtin_lrint(value);
#else
  return (int)lrint(value);
#endif
}

template<class T>
constexpr __host__ __device__ void swap(T &a, T &b)
{
#ifdef __CUDA_ARCH__
    T c = a;
    a   = b;
    b   = c;
#else
    std::swap(a, b);
#endif
}

/**
 * @defgroup NVCV_CPP_CUDATOOLS_LINALG Linear algebra
 * @{
 */

/**
 * @brief Vector class to represent small vectors.
 *
 * @tparam T Vector value type.
 * @tparam N Number of elements.
 */
template<class T, int N>
class Vector
{
public:
    // @brief Type of values in this vector.
    using Type = T;

    /**
     * @brief Get size (number of elements) of this vector
     *
     * @return Vector size
     */
    constexpr __host__ __device__ int size() const
    {
        return N;
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] i Position to access
     *
     * @return Value (constant reference) at given position
     */
    constexpr const __host__ __device__ T &operator[](int i) const
    {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] i Position to access
     *
     * @return Value (reference) at given position
     */
    constexpr __host__ __device__ T &operator[](int i)
    {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     * @brief Pointer-access operator (constant)
     *
     * @return Pointer to the first element of this vector
     */
    constexpr __host__ __device__ operator const T *() const
    {
        return &m_data[0];
    }

    /**
     * @brief Pointer-access operator
     *
     * @return Pointer to the first element of this vector
     */
    constexpr __host__ __device__ operator T *()
    {
        return &m_data[0];
    }

    /**
     * @brief Begin (constant) pointer access
     *
     * @return Pointer to the first element of this vector
     */
    constexpr const __host__ __device__ T *cbegin() const
    {
        return &m_data[0];
    }

    /**
     * @brief Begin pointer access
     *
     * @return Pointer to the first element of this vector
     */
    constexpr __host__ __device__ T *begin()
    {
        return &m_data[0];
    }

    /**
     * @brief End (constant) pointer access
     *
     * @return Pointer to the one-past-last element of this vector
     */
    constexpr const __host__ __device__ T *cend() const
    {
        return &m_data[0] + size();
    }

    /**
     * @brief End pointer access
     *
     * @return Pointer to the one-past-last element of this vector
     */
    constexpr __host__ __device__ T *end()
    {
        return &m_data[0] + size();
    }

    /**
     * @brief Convert a vector of this class to an stl vector
     *
     * @return STL std::vector
     */
    std::vector<T> to_vector() const
    {
        return std::vector<T>(cbegin(), cend());
    }

    /**
     * @brief Get a sub-vector of this vector
     *
     * @param[in] beg Position to start getting values from this vector
     *
     * @return Vector of value from given index to the end
     *
     * @tparam R Size of the sub-vector to be returned
     */
    template<int R>
    constexpr __host__ __device__ Vector<T, R> subv(int beg) const
    {
        Vector<T, R> v;
        for (int i = beg; i < beg + R; ++i)
        {
            v[i - beg] = m_data[i];
        }
        return v;
    }

    // @brief On-purpose public data to allow POD-class direct initialization
    T m_data[N] = {};
};


/**
 * @brief Matrix class to represent small matrices.
 *
 * @tparam T Matrix value type.
 * @tparam M Number of rows.
 * @tparam N Number of columns. Default is M.
 */
template<class T, int M, int N = M>
class Matrix
{
public:
    // @brief Type of values in this matrix.
    using Type = T;

    /**
     * @brief Get number of rows of this matrix
     *
     * @return Number of rows
     */
    constexpr __host__ __device__ int rows() const
    {
        return M;
    }

    /**
     * @brief Get number of columns of this matrix
     *
     * @return Number of columns
     */
    constexpr __host__ __device__ int cols() const
    {
        return N;
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] i Row of the matrix to access
     *
     * @return Vector (constant reference) of the corresponding row
     */
    constexpr const __host__ __device__ Vector<T, N> &operator[](int i) const
    {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] i Row of the matrix to access
     *
     * @return Vector (reference) of the corresponding row
     */
    constexpr __host__ __device__ Vector<T, N> &operator[](int i)
    {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     * @brief Subscript operator for read-only access of matrix elements.
     *
     * @param[in] c Coordinates (y row and x column) of the matrix element to access
     *
     * @return Element (constant reference) of the corresponding row and column
     */
    constexpr const __host__ __device__ T &operator[](int2 c) const
    {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return m_data[c.y][c.x];
    }

    /**
     * @brief Subscript operator for read-and-write access of matrix elements.
     *
     * @param[in] c Coordinates (y row and x column) of the matrix element to access
     *
     * @return Element (reference) of the corresponding row and column
     */
    constexpr __host__ __device__ T &operator[](int2 c)
    {
        assert(c.y >= 0 && c.y < rows());
        assert(c.x >= 0 && c.x < cols());
        return m_data[c.y][c.x];
    }

    /**
     * @brief Get column j of this matrix
     *
     * @param[in] j Index of column to get
     *
     * @return Column j (copied) as a vector
     */
    constexpr __host__ __device__ Vector<T, M> col(int j) const
    {
        Vector<T, M> c;
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            c[i] = m_data[i][j];
        }
        return c;
    }

    /**
     * @brief Set column j of this matrix
     *
     * @param[in] j Index of column to set
     * @param[in] c Vector to place in matrix column
     */
    constexpr __host__ __device__ void set_col(int j, const Vector<T, M> &c)
    {
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            m_data[i][j] = c[i];
        }
    }

    // @overload void set_col(int j, const T *c)
    constexpr __host__ __device__ void set_col(int j, const T *c)
    {
#pragma unroll
        for (int i = 0; i < rows(); ++i)
        {
            m_data[i][j] = c[i];
        }
    }

    /**
     * @brief Get a sub-matrix of this matrix
     *
     * @param[in] skip_i Row to skip when getting values from this matrix
     * @param[in] skip_j Column to skip when getting values from this matrix
     *
     * @return Matrix with one less row and one less column
     */
    constexpr __host__ __device__ Matrix<T, M - 1, N - 1> subm(int skip_i, int skip_j) const
    {
        Matrix<T, M - 1, N - 1> ret;
        int                     ri = 0;
        for (int i = 0; i < rows(); ++i)
        {
            if (i == skip_i)
            {
                continue;
            }
            int rj = 0;
            for (int j = 0; j < cols(); ++j)
            {
                if (j == skip_j)
                {
                    continue;
                }
                ret[ri][rj] = (*this)[i][j];
                ++rj;
            }
            ++ri;
        }
        return ret;
    }

    // @brief On-purpose public data to allow POD-class direct initialization
    Vector<T, N> m_data[M];
};

// Determinant -----------------------------------------------------------------

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 0, 0> &m)
{
    return T{1};
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 1, 1> &m)
{
    return m[0][0];
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 2, 2> &m)
{
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

template<class T>
constexpr __host__ __device__ T det(const Matrix<T, 3, 3> &m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
         + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

template<class T, int M>
constexpr __host__ __device__ T det(const Matrix<T, M, M> &m)
{
    T d = T{0};
#pragma unroll
    for (int i = 0; i < M; ++i)
    {
        d += ((i % 2 == 0 ? 1 : -1) * m[0][i] * det(m.subm(0, i)));
    }
    return d;
}

// Matrix Inverse --------------------------------------------------------------

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 1, 1> &m)
{
    m[0][0] = T{1} / m[0][0];
}

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 2, 2> &m)
{
    T d = det(m);

    swap(m[0][0], m[1][1]);
    m[0][0] /= d;
    m[1][1] /= d;

    m[0][1] = -m[0][1] / d;
    m[1][0] = -m[1][0] / d;
}

template<class T>
constexpr __host__ __device__ void inv_inplace(Matrix<T, 3, 3> &m)
{
    T d = det(m);

    Matrix<T, 3, 3> A;
    A[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    A[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / d;
    A[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    A[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / d;
    A[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    A[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / d;
    A[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    A[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / d;
    A[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;

    m = A;
}


}  // namespace math
}  // namespace cuda
}  // namespace lcv
}  // namespace lad
