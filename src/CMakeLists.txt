set(TARGET_NAME lcv)

add_library(${TARGET_NAME} SHARED
    resize.cpp
    cvt_color.cpp
    cylindrical_projector.cpp
    warp.cpp
    undistort.cpp
    remap.cpp
    avm_stitch.cpp
)

target_link_libraries(${TARGET_NAME}
  lcv_cuda
)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src/cuda)

add_subdirectory(cuda)

if (ENABLE_PYBIND)
    set(PY_TARGET_NAME py${PROJECT_NAME})
    add_library(${PY_TARGET_NAME} SHARED pylcv.cc)
    set_target_properties(${PY_TARGET_NAME} PROPERTIES PREFIX "" SUFFIX "${PYLINFER_SUFFIX}")
    target_link_libraries(${PY_TARGET_NAME}
        PUBLIC
        ${CONAN_TARGETS}
        ${TARGET_NAME}
        -lcudart)
    target_include_directories(${PY_TARGET_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    install(
        TARGETS ${PY_TARGET_NAME} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
endif(ENABLE_PYBIND)

set(out_headers
    cvt_color.h
    resize.h
    cylindrical_projector.h
    warp.h
    undistort.h
    remap.h
    avm_stitch.h
    image_data.h
    cuda/types.h
)
install(FILES ${out_headers}
    DESTINATION include/lcv
)

# add_executable(test_warp warp_jpeg_test2.cpp)
# target_link_libraries(test_warp
#         PUBLIC
#         ${CONAN_TARGETS}
#         lcv_cuda
#         ${TARGET_NAME}
#         -lcudart)

install(
  TARGETS ${TARGET_NAME} 
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

set_target_warnings(
    TARGET_NAME ${TARGET_NAME} # 要添加警告编译参数的目标名,可以有多个
    WARNINGS_AS_ERRORS # 警告视为错误
)
