# SciCudaOperator
Our source code (including libraries starting with the v2.x versions, and all versions), and ALL source code will be licensed under the [Elastic License v2 (ELv2)](https://www.elastic.co/licensing/elastic-license).

We chose ELv2 because of its permissiveness and simplicity. We're following the well-paved path of other great infrastructure projects like Elasticsearch and MongoDB that have implemented similar source code licenses to preserve their communities. Our community and customers still have no-charge and open access to use, modify, redistribute, and collaborate on the code. ELv2 also protects our continued investment in developing freely available libraries and developer tools by restricting cloud service providers from offering it as a service.

## sample
```cpp
  FILE *in;
  int length;
  in = fopen("../../test/resources/bgr_3840_2160_ori.rgb", "rb");
  cudaStream_t stream;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  printf("fopen ok\n");
  fseek(in, 0L, SEEK_END);
  length = ftell(in);
  fseek(in, 0L, SEEK_SET);
  uint8_t *buffer = new uint8_t[length];
  fread(buffer, sizeof(uint8_t), length, in);
  fclose(in);
  Resize resize;
  ImageData src;
  ImageData dst;
  uint8_t *src_ptr = nullptr;
  uint8_t *dst_ptr = nullptr;
  cudaMalloc(&src_ptr, length);
  cudaMalloc(&dst_ptr, length / 4);
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
  uint8_t *h_dst = (uint8_t *)malloc(length / 4 * sizeof(uint8_t));
  cudaMemcpy((void *)h_dst, dst.base_ptr, length / 4, cudaMemcpyDeviceToHost);
  FILE *file = fopen("../../test/resources/bgr_1920_1080.rgb", "w");
  fwrite((void *)h_dst, 1, length / 4, file);
  fclose(file);
  delete[] buffer;
  cudaFree(src_ptr);
  cudaFree(dst_ptr);
  free(h_dst);
```
