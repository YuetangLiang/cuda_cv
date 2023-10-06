/******************************************************************************
 *****************************************************************************/

#include "cvdef.h"
#include "lcv_cuda.h"
#include "math/math.h"
#include "math/saturate.h"
#include "tensor/packed_tensor.h"
#include "tensor/tensor_wrap.h"
#include "utils/transform.h"

namespace lad {
namespace lcv {
namespace cuda {

#define BLOCK 32

template <class Transform, class Filter, typename T>
__global__ void warp(const Filter src, Ptr2dNHWC<T> dst, Transform transform) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int lid = get_lid();
  const int batch_idx = get_batch_idx();
  extern __shared__ float coeff[];
  if (lid < 9) {
    coeff[lid] = transform.xform[lid];
  }
  __syncthreads();
  if (x < dst.cols && y < dst.rows) {
    const float2 coord = Transform::calcCoord(coeff, x, y);
    //*dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<T>>(src(batch_idx, coord.y, coord.x));

    for (int e = 0; e < NumElements<T>; ++e) {
      auto p = &((reinterpret_cast<uchar *>(&(*dst.ptr(batch_idx, y, x))))[e]);
      *p = static_cast<const uchar>(reinterpret_cast<const float *>(&src(batch_idx, coord.y, coord.x))[e]);

      // printf("e = %d, *dst.ptr = %d, *p = %d\n", e, (reinterpret_cast<uchar *>(&(*dst.ptr(batch_idx, y, x))))[e],
      // *p);
    }
  }
}

template <class Transform, template <typename> class Filter, template <typename> class B, typename T>
struct WarpDispatcher {
  static void call(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, Transform transform, const float4 borderValue,
                   cudaStream_t stream) {
    // using work_type = nvcv::cuda::ConvertBaseTypeTo<float, T>;
    const int in_width = src.cols;
    const int in_height = src.rows;
    const int out_width = dst.cols;
    const int out_height = dst.rows;
    const int out_batch = dst.batches;

    using work_type = ConvertBaseTypeTo<float, T>;

    dim3 block(BLOCK, BLOCK / 4);
    dim3 grid(math::divUp(out_width, block.x), math::divUp(out_height, block.y), out_batch);

    // work_type borderVal = DropCast<NumComponents<T>>(borderValue);
    work_type borderVal;
    for (size_t i = 0; i < NumComponents<T>; ++i) {
      auto p = &((reinterpret_cast<float *>(&borderVal))[i]);
      *p = (reinterpret_cast<const float *>(&borderValue))[i];
      // printf("i = %d, borderVal[i] = %f, *p = %f\n", i, ((reinterpret_cast<float *>(&borderVal))[i]), *p);
    }

    B<work_type> brd(in_height, in_width, borderVal);
    BorderReader<Ptr2dNHWC<T>, B<work_type>> brdSrc(src, brd);
    Filter<BorderReader<Ptr2dNHWC<T>, B<work_type>>> filter_src(brdSrc);
    size_t smem_size = 9 * sizeof(float);
    warp<Transform><<<grid, block, smem_size, stream>>>(filter_src, dst, transform);
    checkKernelErrors();
  }
};

template <class Transform, typename T>
void warp_caller(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, Transform transform, int interpolation, int borderMode,
                 const float4 borderValue, cudaStream_t stream) {
  typedef void (*func_t)(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, Transform transform, const float4 borderValue,
                         cudaStream_t stream);

  static const func_t funcs[1][1] = {{WarpDispatcher<Transform, PointFilter, BrdConstant, T>::call,
                                      /* WarpDispatcher<Transform, PointFilter, BrdReplicate, T>::call,
                                      WarpDispatcher<Transform, PointFilter, BrdReflect, T>::call,
                                      WarpDispatcher<Transform, PointFilter, BrdWrap, T>::call,
                                      WarpDispatcher<Transform, PointFilter, BrdReflect101, T>::call},
                                     {WarpDispatcher<Transform, LinearFilter, BrdConstant, T>::call,
                                      WarpDispatcher<Transform, LinearFilter, BrdReplicate, T>::call,
                                      WarpDispatcher<Transform, LinearFilter, BrdReflect, T>::call,
                                      WarpDispatcher<Transform, LinearFilter, BrdWrap, T>::call,
                                      WarpDispatcher<Transform, LinearFilter, BrdReflect101, T>::call},
                                     {WarpDispatcher<Transform, CubicFilter, BrdConstant, T>::call,
                                      WarpDispatcher<Transform, CubicFilter, BrdReplicate, T>::call,
                                      WarpDispatcher<Transform, CubicFilter, BrdReflect, T>::call,
                                      WarpDispatcher<Transform, CubicFilter, BrdWrap, T>::call,
                                      WarpDispatcher<Transform, CubicFilter, BrdReflect101, T>::call} */}};

  // funcs[interpolation][borderMode](src, dst, transform, borderValue, stream);
  funcs[0][0](src, dst, transform, borderValue, stream);
}

template <typename T>
void warpPerspective(const TensorData &src, const TensorData &dst, PerspectiveTransform transform, const int interpolation,
                     int borderMode, const float4 borderValue, cudaStream_t stream) {
  const int in_batch = src.GetDataShape().N;
  const int in_channel = src.GetDataShape().C;
  const int in_height = src.GetDataShape().H;
  const int in_width = src.GetDataShape().W;
  uint8_t *in_ptr = src.GetBasePtr();

  const int out_batch = dst.GetDataShape().N;
  const int out_channel = dst.GetDataShape().C;
  const int out_height = dst.GetDataShape().H;
  const int out_width = dst.GetDataShape().W;
  uint8_t *out_ptr = dst.GetBasePtr();

  Ptr2dNHWC<T> src_tensor(in_batch, in_height, in_width, in_channel, reinterpret_cast<T *>(in_ptr));
  Ptr2dNHWC<T> dst_tensor(out_batch, out_height, out_width, out_channel, reinterpret_cast<T *>(out_ptr));

  warp_caller<PerspectiveTransform, T>(src_tensor, dst_tensor, transform, interpolation, borderMode, borderValue,
                                       stream);
}

static void invertMat(const float *M, float *h_aCoeffs) {
  // M is stored in row-major format M[0,0], M[0,1], M[0,2], M[1,0], M[1,1], M[1,2]
  float den = M[0] * M[4] - M[1] * M[3];
  den = std::abs(den) > 1e-5 ? 1. / den : .0;
  h_aCoeffs[0] = (float)M[5] * den;
  h_aCoeffs[1] = (float)-M[1] * den;
  h_aCoeffs[2] = (float)(M[1] * M[5] - M[4] * M[2]) * den;
  h_aCoeffs[3] = (float)-M[3] * den;
  h_aCoeffs[4] = (float)M[0] * den;
  h_aCoeffs[5] = (float)(M[3] * M[2] - M[0] * M[5]) * den;
}

size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type) {
  return 9 * sizeof(float);
}

ErrorCode WarpPerspective::Infer(const TensorData &src, const TensorData &dst, const float *transMatrix,
                                 const int interpolation, int borderMode, const float4 borderValue,
                                 cudaStream_t stream) {
  const int channels = src.GetDataShape().C;
  const DataType data_type = src.GetDataType();

  typedef void (*func_t)(const TensorData &src, TensorData &dst, PerspectiveTransform transform,
                         const int interpolation, int borderMode, const float4 borderValue, cudaStream_t stream);
#if 0
  static const func_t funcs[6][4] = {
      {warpPerspective<uchar>, 0 /*warpPerspective<uchar2>*/, warpPerspective<uchar3>, warpPerspective<uchar4>},
      {0 /*warpPerspective<schar>*/, 0 /*warpPerspective<char2>*/, 0 /*warpPerspective<char3>*/,
       0 /*warpPerspective<char4>*/},
      {warpPerspective<ushort>, 0 /*warpPerspective<ushort2>*/, warpPerspective<ushort3>, warpPerspective<ushort4>},
      {warpPerspective<short>, 0 /*warpPerspective<short2>*/, warpPerspective<short3>, warpPerspective<short4>},
      {0 /*warpPerspective<int>*/, 0 /*warpPerspective<int2>*/, 0 /*warpPerspective<int3>*/,
       0 /*warpPerspective<int4>*/},
      {warpPerspective<float>, 0 /*warpPerspective<float2>*/, warpPerspective<float3>, warpPerspective<float4>}};

  const func_t func = funcs[data_type][channels - 1];
#endif

  PerspectiveTransform transform(transMatrix);

  if (interpolation) {
    cuda::math::Matrix<float, 3, 3> tempMatrixForInverse;

    tempMatrixForInverse[0][0] = (float)(transMatrix[0]);
    tempMatrixForInverse[0][1] = (float)(transMatrix[1]);
    tempMatrixForInverse[0][2] = (float)(transMatrix[2]);
    tempMatrixForInverse[1][0] = (float)(transMatrix[3]);
    tempMatrixForInverse[1][1] = (float)(transMatrix[4]);
    tempMatrixForInverse[1][2] = (float)(transMatrix[5]);
    tempMatrixForInverse[2][0] = (float)(transMatrix[6]);
    tempMatrixForInverse[2][1] = (float)(transMatrix[7]);
    tempMatrixForInverse[2][2] = (float)(transMatrix[8]);

    math::inv_inplace(tempMatrixForInverse);

    transform.xform[0] = tempMatrixForInverse[0][0];
    transform.xform[1] = tempMatrixForInverse[0][1];
    transform.xform[2] = tempMatrixForInverse[0][2];
    transform.xform[3] = tempMatrixForInverse[1][0];
    transform.xform[4] = tempMatrixForInverse[1][1];
    transform.xform[5] = tempMatrixForInverse[1][2];
    transform.xform[6] = tempMatrixForInverse[2][0];
    transform.xform[7] = tempMatrixForInverse[2][1];
    transform.xform[8] = tempMatrixForInverse[2][2];
  }

  // func(src, dst, transform, interpolation, borderMode, borderValue, stream);
  warpPerspective<uchar3>(src, dst, transform, interpolation, borderMode, borderValue, stream);

  return ErrorCode::SUCCESS;
}

}  // namespace cuda
}  // namespace lcv
}  // namespace lad
