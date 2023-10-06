/******************************************************************************
 *****************************************************************************/

#include "resize.h"
#include <stdint.h>
#include "utils/time.h"
#include "gtest/gtest.h"

TEST(Resize, Operator) {
  FILE *in;
  int length;
  // 1. 打开图片文件
  in = fopen("../../test/resources/bgr_3840_2160_ori.rgb", "rb");
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

  lad::lcv::Resize resize;
  lad::lcv::ImageData src;
  lad::lcv::ImageData dst;
  uint8_t *src_ptr = nullptr;
  uint8_t *dst_ptr = nullptr;

  cudaMalloc(&src_ptr, length);
  cudaMalloc(&dst_ptr, length / 4);
  printf("src_ptr %p\n", src_ptr);
  src.format = lad::lcv::kBGR_8U;
  src.width = 3840;
  src.height = 2160;
  src.row_stride = 3840;
  cudaMemcpy((void *)src_ptr, buffer, length, cudaMemcpyHostToDevice);
  src.base_ptr = src_ptr;

  dst.format = lad::lcv::kBGR_8U;
  dst.width = 1920;
  dst.height = 1080;
  dst.row_stride = 1920;
  dst.base_ptr = dst_ptr;

  std::uint64_t time1{lad::lcv::cuda::system_now()};
  int result = resize.Operator(src, dst, lad::lcv::INTERP_NEAREST, 0);
  std::uint64_t time2{lad::lcv::cuda::system_now()};
  printf("time1 = %ld, time2 = %ld, time2 - time1 = %ld\n", time1, time2, (time2 - time1));
  printf("result %d \n", result);
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  uint8_t *h_dst = (uint8_t *)malloc(length / 4 * sizeof(uint8_t));
  cudaMemcpy((void *)h_dst, dst.base_ptr, length / 4, cudaMemcpyDeviceToHost);
  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  FILE *file = fopen("../../test/resources/bgr_1920_1080.rgb", "w");
  if (file == NULL) {  // 打开失败直接返回
    return;
  }

  fwrite((void *)h_dst, 1, length / 4, file);  // 写入操作
  fclose(file);

  delete[] buffer;
  cudaFree(src_ptr);
  cudaFree(dst_ptr);
  free(h_dst);
}
