/******************************************************************************
 *****************************************************************************/

#include <stdint.h>
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"
#include "utils/time.h"
#include "warp.h"

TEST(WarpPerspectiveJPEG, Operator) {
  // 1. 读取图片
  cv::Mat frame = cv::imread("../../test/resources/bgr_2880_1860_warp_ori.jpeg");

  // 2. 申请资源
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  uint8_t* src_ptr = nullptr;
  uint8_t* dst_ptr = nullptr;
  size_t length = frame.rows * frame.cols * frame.channels() * sizeof(uchar);

  cudaMalloc(&src_ptr, length);
  cudaMalloc(&dst_ptr, length);

  if (nullptr == src_ptr || nullptr == dst_ptr) {
    std::cout << "内存申请失败" << std::endl;
    return;
  }

  // 3. 输入结构
  lad::lcv::ImageData src;
  src.format = lad::lcv::kBGR_8U;
  src.width = frame.cols;
  src.height = frame.rows;
  src.row_stride = frame.cols;
  std::uint64_t time_copy1{lad::lcv::cuda::system_now()};
  cudaMemcpy(reinterpret_cast<void*>(src_ptr), frame.data, length, cudaMemcpyHostToDevice);
  // cudaMemcpy(reinterpret_cast<void*>(src_ptr), frame.ptr<uint8_t*>(), length, cudaMemcpyHostToDevice);
  std::uint64_t time_copy2{lad::lcv::cuda::system_now()};
  printf("time_copy1 = %ld, time_copy2 = %ld, time_copy2 - time_copy1 = %ld\n", time_copy1, time_copy2, (time_copy2 - time_copy1));
  src.base_ptr = src_ptr;

  // 4. 输出结构
  lad::lcv::ImageData dst;
  dst.format = lad::lcv::kBGR_8U;
  dst.width = frame.cols;
  dst.height = frame.rows;
  dst.row_stride = frame.cols;
  dst.base_ptr = dst_ptr;

  // 5. 转换
  lad::lcv::WarpPerspective warp;

  const float xform[9] = {0.996600978683475,     0.1182040112496254,    -125.6150237041926,
                          -0.006767036783381765, 1.053820548892408,     -63.34798599158658,
                          -1.06319498835945e-06, 5.260880922951616e-05, 1};

  float4 borderValue{10.0, 10.0, 10.0, 10.0};

  std::uint64_t time1{lad::lcv::cuda::system_now()};
  int result =
      warp.Operator(src, dst, (const float*)xform, lad::lcv::INTERP_NEAREST, lad::lcv::BORDER_CONSTANT, borderValue, 0);
  std::uint64_t time2{lad::lcv::cuda::system_now()};
  printf("time1 = %ld, time2 = %ld, time2 - time1 = %ld\n", time1, time2, (time2 - time1));

  // ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  cudaStreamSynchronize(stream);

  uint8_t* h_dst = reinterpret_cast<uint8_t*>(malloc(length * sizeof(uint8_t)));
  cudaMemcpy(reinterpret_cast<void*>(h_dst), dst.base_ptr, length, cudaMemcpyDeviceToHost);
  // ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  cudaStreamDestroy(stream);

  // 5. 可视化
  cv::Mat img_result(dst.height, dst.width, CV_8UC3, reinterpret_cast<void*>(h_dst));
  cv::imwrite("../../test/resources/bgr_2880_1860_warp.jpeg", img_result);

  // 6. 释放资源
  cudaFree(src_ptr);
  cudaFree(dst_ptr);
  free(h_dst);
}
