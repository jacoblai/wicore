# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(WiCore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# 查找TensorRT
set(TensorRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT root directory")
if(NOT EXISTS ${TensorRT_ROOT})
    set(TensorRT_ROOT "/opt/TensorRT")
endif()

find_library(TensorRT_LIBRARY nvinfer HINTS ${TensorRT_ROOT}/lib)
find_library(TensorRT_PLUGIN_LIBRARY nvinfer_plugin HINTS ${TensorRT_ROOT}/lib)
find_library(TensorRT_ONNX_PARSER nvonnxparser HINTS ${TensorRT_ROOT}/lib)

if(NOT TensorRT_LIBRARY)
    message(FATAL_ERROR "TensorRT library not found. Please set TensorRT_ROOT.")
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 查找pkg-config依赖
find_package(PkgConfig REQUIRED)

# 查找jsoncpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# 查找evhtp
pkg_check_modules(EVHTP REQUIRED evhtp)

# 查找SentencePiece
find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)

# 查找libevent
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBEVENT REQUIRED libevent)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${EVHTP_INCLUDE_DIRS}
    ${SENTENCEPIECE_INCLUDE_DIRS}
    ${LIBEVENT_INCLUDE_DIRS}
)

# 源文件
set(SOURCES
    src/main.cpp
    src/wicore_engine.cpp
    src/hmt_memory_manager.cpp
    src/multimodal_processor.cpp
    src/tensorrt_inference_engine.cpp
    src/batch_scheduler.cpp
    src/web_server.cpp
)

# CUDA内核文件（暂时为空，后续添加）
set(CUDA_SOURCES
    # src/cuda_kernels.cu
)

# 创建可执行文件
add_executable(wicore_server ${SOURCES} ${CUDA_SOURCES})

# 设置目标属性
set_target_properties(wicore_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# CUDA架构设置
set_property(TARGET wicore_server PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

# 链接库
target_link_libraries(wicore_server
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${TensorRT_LIBRARY}
    ${TensorRT_PLUGIN_LIBRARY}
    ${TensorRT_ONNX_PARSER}
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
    ${EVHTP_LIBRARIES}
    ${SENTENCEPIECE_LIBRARIES}
    ${LIBEVENT_LIBRARIES}
    pthread
    stdc++fs  # for std::filesystem
)

# 编译选项
target_compile_options(wicore_server PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -Wall>
)

# 链接选项
target_link_options(wicore_server PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wl,--as-needed>
)

# 调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(wicore_server PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0 -DDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G -O0 -DDEBUG>
    )
endif()

# 安装规则
install(TARGETS wicore_server
    RUNTIME DESTINATION bin
)

# 配置文件安装
install(FILES config_template.json
    DESTINATION etc/wicore
    RENAME config.json
)

# 文档安装
install(FILES README_WICORE_CPP.md
    DESTINATION share/doc/wicore
)

# 创建测试目标（可选）
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    # 添加测试相关的配置
endif()

# 打印配置信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "TensorRT root: ${TensorRT_ROOT}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA architectures: 75;80;86;89") 