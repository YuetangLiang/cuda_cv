/******************************************************************************
 *****************************************************************************/

#include "warp.h"
#include <stdint.h>
#include "gtest/gtest.h"
#include "utils/time.h"

TEST(WarpPerspective, Operator) {
  FILE *in;
  int length;
  // 1. 打开图片文件
  in = fopen("../../test/resources/bgr_2880_1860_warp_ori.rgb", "rb");
  if (!in) {
    return;
  }
  cudaStream_t stream;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  printf("fopen ok\n");
  // 2. 计算图片长度
  fseek(in, 0L, SEEK_END);
  length = ftell(in);
  fseek(in, 0L, SEEK_SET);
  // 3. 创建内存缓存区
  uint8_t *buffer = new uint8_t[length];
  // 4. 读取图片
  fread(buffer, sizeof(uint8_t), length, in);
  if (in) {
    printf("in not null, picture read successfully.\n");
  }
  fclose(in);
  // 到此，图片已经成功的被读取到内存（buffer）中

  printf("length = %d\n", length);

  lad::lcv::WarpPerspective warp;
  lad::lcv::ImageData src;
  lad::lcv::ImageData dst;
  uint8_t *src_ptr = nullptr;
  uint8_t *dst_ptr = nullptr;

  cudaMalloc(&src_ptr, length);
  cudaMalloc(&dst_ptr, length);
  printf("src_ptr %p\n", src_ptr);
  src.format = lad::lcv::kBGR_8U;
  src.width = 2880;
  src.height = 1860;
  src.row_stride = 2880;
  cudaMemcpy((void *)src_ptr, buffer, length, cudaMemcpyHostToDevice);
  src.base_ptr = src_ptr;

  dst.format = lad::lcv::kBGR_8U;
  dst.width = 2880;
  dst.height = 1860;
  dst.row_stride = 2880;
  dst.base_ptr = dst_ptr;

  /* const float xform[9] = {
    0.4705075885018677, 0.109657846049481, -178.2455103440204,
    0.00153851522006792, 0.5302831028229942, -167.6817093045246,
    -3.129252699350168e-06, 0.0001772050521398554, 1
}; */

/*   const float xform[9] = {0.4705075885018677,     0.309657846049481,     -200.2455103440204,
                          0.00153851522006792,    0.7302831028229942,    -160.6817093045246,
                          -3.129252699350168e-06, 0.0001772050521398554, 1}; */

  const float xform[9] = {0.996600978683475,     0.1182040112496254,    -125.6150237041926,
                           -0.006767036783381765, 1.053820548892408,     -63.34798599158658,
                           -1.06319498835945e-06, 5.260880922951616e-05, 1};


  float4 borderValue{10.0, 10.0, 10.0, 10.0};

  std::uint64_t time1{lad::lcv::cuda::system_now()};
  int result = warp.Operator(src, dst, xform, lad::lcv::INTERP_NEAREST, lad::lcv::BORDER_CONSTANT, borderValue, 0);
  std::uint64_t time2{lad::lcv::cuda::system_now()};
  printf("time1 = %ld, time2 = %ld, time2 - time1 = %ld\n", time1, time2, (time2 - time1));
  printf("result %d \n", result);
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  uint8_t *h_dst = (uint8_t *)malloc(length * sizeof(uint8_t));
  cudaMemcpy((void *)h_dst, dst.base_ptr, length, cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  FILE *file = fopen("../../test/resources/bgr_2880_1860_warp.rgb", "w");
  if (file == NULL) {  // 打开失败直接返回
    return;
  }

  fwrite((void *)h_dst, 1, length, file);  // 写入操作
  fclose(file);

  delete[] buffer;
  cudaFree(src_ptr);
  cudaFree(dst_ptr);
  free(h_dst);
}
