/******************************************************************************
 *****************************************************************************/

#include "cvt_color.h"
#include <stdio.h>
#include <iostream>
#include "utils/time.h"
#include "gtest/gtest.h"

TEST(CvtColor, Operator) {
  FILE *in;
  int length;

  // 1. 打开图片文件
  in = fopen("../../test/resources/yuv420sp_NV12_3840_2160.yuv", "rb");
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
  // delete[] buffer;

  printf("length = %d\n", length);
  printf("picture process end\n");
  // return 0;

  lad::lcv::CvtColor cvt_color;

  lad::lcv::ImageData src;
  lad::lcv::ImageData dst;
  uint8_t *src_ptr = nullptr;
  uint8_t *dst_ptr = nullptr;

  cudaMalloc(&src_ptr, length);
  cudaMalloc(&dst_ptr, 2160 * 3840 * 3);
  printf("src_ptr %p\n", src_ptr);
  src.format = lad::lcv::kNV12_8U;
  src.width = 3840;
  //src.height = 2160 * 3 / 2;
  src.height = 2160;
  src.row_stride = 3840;
  cudaMemcpy((void *)src_ptr, buffer, length, cudaMemcpyHostToDevice);
  src.base_ptr = src_ptr;
  // for (int i = 0; i < length; i++) {
  // printf("src %u %u\n", src_ptr[i], buffer[i]);
  //}

  dst.format = lad::lcv::kBGR_8U;
  dst.width = 3840;
  dst.height = 2160;
  dst.row_stride = 3840;
  dst.base_ptr = dst_ptr;

  // int result = (int)cvt_color.Operator(src, dst, lad::lcv::NVCV_COLOR_YUV2RGB_NV12, stream);
  std::uint64_t time1{lad::lcv::cuda::system_now()};
  int result = (int)cvt_color.Operator(src, dst, lad::lcv::COLOR_YUV2BGR_NV12, stream);
  std::uint64_t time2{lad::lcv::cuda::system_now()};
  printf("time1 = %ld, time2 = %ld, time2 - time1 = %ld\n", time1, time2, (time2 - time1));
  printf("result %d \n", result);
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  uint8_t *h_dst = (uint8_t *)malloc(2160 * 3840 * 3 * sizeof(uint8_t));
  cudaMemcpy((void *)h_dst, dst.base_ptr, 2160 * 3840 * 3, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < length; i++) {
  // printf("src i %u %u\n", i, h_dst[i]);
  //}
  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  FILE *file = fopen("../../test/resources/bgr_3840_2160.rgb", "w");
  if (file == NULL) {  // 打开失败直接返回
    return;
  }

  fwrite((void *)h_dst, 1, 2160 * 3840 * 3, file);  // 写入操作
  fclose(file);

  delete[] buffer;
  cudaFree(src_ptr);
  cudaFree(dst_ptr);
  free(h_dst);
}
