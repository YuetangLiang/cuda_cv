/******************************************************************************
 *****************************************************************************/
#include "avm_stitch.h"
#include "lcv_cuda.h"

namespace lad {
namespace lcv {

AvmStitch::AvmStitch(/* args */) { ; }

AvmStitch::~AvmStitch() { ; }

void AvmStitch::Merge(const unsigned char* src1, const unsigned char* src2, const float* weights, unsigned char* dst,
                      int left, int up, int src_width, int src_height, int dst_width, cudaStream_t stream) {
  cuda::AvmStitch avm_stitch;
  avm_stitch.Merge(src1, src2, weights, dst, left, up, src_width, src_height, dst_width, stream);

  return;
}

void AvmStitch::Copy(const unsigned char* src, unsigned char* dst, int left, int up, int copy_width, int copy_height,
                     int dst_width, cudaStream_t stream) {
  cuda::AvmStitch avm_stitch;
  avm_stitch.Copy(src, dst, left, up, copy_width, copy_height, dst_width, stream);

  return;
}

}  // namespace lcv
}  // namespace lad