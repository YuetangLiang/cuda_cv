/******************************************************************************
 *****************************************************************************/

#include <stdio.h>
#include <iostream>
#include "cvt_color.h"
#include "gtest/gtest.h"
#include "utils/time.h"

TEST(CvtColorNV12ToYU12, Operator) {
  FILE *in;
  int length;

  // 1. 打开图片文件
  in = fopen("../../test/resources/NV12ToYU12.yuv", "rb");
  if (!in) {
    return;
  }
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
  printf("picture process end\n");

  lad::lcv::ImageData src;
  lad::lcv::ImageData dst;
  uint8_t *src_ptr = nullptr;
  uint8_t *dst_ptr = nullptr;

  cudaMalloc(&src_ptr, length);
  cudaMalloc(&dst_ptr, length);
  printf("src_ptr %p\n", src_ptr);
  src.format = lad::lcv::kNV12_8U;
  src.width = 3840;
  src.height = 2160;
  src.row_stride = 3840;

  std::uint64_t time_h2d1{lad::lcv::cuda::system_now()};
  cudaMemcpy((void *)src_ptr, buffer, length, cudaMemcpyHostToDevice);
  std::uint64_t time_h2d2{lad::lcv::cuda::system_now()};
  printf("time_h2d1 = %ld, time_h2d2 = %ld, time_h2d2 - time_h2d1 = %ld\n", time_h2d1, time_h2d2,
         (time_h2d2 - time_h2d1));
  src.base_ptr = src_ptr;

  dst.format = lad::lcv::kIYUV_8U;
  dst.width = 3840;
  dst.height = 2160;
  dst.row_stride = 3840;
  dst.base_ptr = dst_ptr;

  cudaStream_t stream;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  std::uint64_t time1{lad::lcv::cuda::system_now()};
  lad::lcv::CvtColor cvt_color;
  int result = (int)cvt_color.Operator(src, dst, lad::lcv::COLOR_NV122YU12, stream);
  std::uint64_t time2{lad::lcv::cuda::system_now()};
  printf("time1 = %ld, time2 = %ld, time2 - time1 = %ld\n", time1, time2, (time2 - time1));
  printf("result %d \n", result);
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  uint8_t *h_dst = (uint8_t *)malloc(length * sizeof(uint8_t));
  std::uint64_t time_d2h1{lad::lcv::cuda::system_now()};
  cudaMemcpy((void *)h_dst, dst.base_ptr, length, cudaMemcpyDeviceToHost);
  std::uint64_t time_d2h2{lad::lcv::cuda::system_now()};
  printf("time_d2h1 = %ld, time_d2h2 = %ld, time_d2h2 - time_d2h1 = %ld\n", time_d2h1, time_d2h2,
         (time_d2h2 - time_d2h1));
  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  FILE *file = fopen("../../test/resources/NV12ToYU12_result.yuv", "w");
  if (file == NULL) {  // 打开失败直接返回
    delete[] buffer;
    cudaFree(src_ptr);
    cudaFree(dst_ptr);
    free(h_dst);
    return;
  }

  fwrite((void *)h_dst, 1, length, file);  // 写入操作
  fclose(file);

  delete[] buffer;
  cudaFree(src_ptr);
  cudaFree(dst_ptr);
  free(h_dst);
}
