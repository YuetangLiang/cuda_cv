
set(TARGET_NAME lcv_cuda)

enable_language(CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${TARGET_NAME} STATIC
    resize.cu
    cvt_color.cu
    cylindrical_projector.cu
    warp.cu
    undistort.cu
    remap.cu
    avm_stitch.cu
)

message("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/cuda)

target_link_libraries(${TARGET_NAME}
    cudart
)