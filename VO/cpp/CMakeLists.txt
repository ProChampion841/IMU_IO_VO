cmake_minimum_required(VERSION 3.10)
project(vo_stream CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime (C++ API): point ONNXRUNTIME_DIR at an unpacked release
# (include/ + lib/), e.g. onnxruntime-linux-aarch64-<ver> on the Jetson:
#   cmake -S . -B build -DONNXRUNTIME_DIR=/opt/onnxruntime
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root (include/, lib/)")
if(NOT ONNXRUNTIME_DIR)
  message(FATAL_ERROR "set -DONNXRUNTIME_DIR=<unpacked onnxruntime release>")
endif()
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_DIR}/include ${ONNXRUNTIME_DIR}/include/onnxruntime NO_DEFAULT_PATH REQUIRED)
find_library(ORT_LIB onnxruntime
  PATHS ${ONNXRUNTIME_DIR}/lib ${ONNXRUNTIME_DIR}/lib64 NO_DEFAULT_PATH REQUIRED)
get_filename_component(ORT_LIB_DIR ${ORT_LIB} DIRECTORY)

# OpenCV: undistortion (initUndistortRectifyMap + remap, as the Python side
# does) and image decoding. initUndistortRectifyMap is in calib3d up to
# OpenCV 4 and in the 3d module from OpenCV 5.
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS 5)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs 3d)
endif()

# Bit-for-bit parity with the Python runtime needs plain IEEE arithmetic: no
# fused multiply-add contraction (GCC contracts by default where the target
# has FMA - aarch64 always does) and no fast-math.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(VO_FP_FLAGS -ffp-contract=off -fno-fast-math)
elseif(MSVC)
  set(VO_FP_FLAGS /fp:precise)
endif()

find_package(Threads REQUIRED)

# The runtime: link this into the application.
add_library(vo_stream STATIC src/vo_stream.cpp src/vo_image.cpp src/vo_json.cpp)
target_include_directories(vo_stream PUBLIC include PRIVATE src ${ORT_INCLUDE_DIR})
target_compile_options(vo_stream PRIVATE ${VO_FP_FLAGS}
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>)
target_link_libraries(vo_stream PUBLIC ${ORT_LIB} ${OpenCV_LIBS} Threads::Threads)
target_include_directories(vo_stream SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

# Replay a recorded flight folder (flight.csv + images/) like
# tools/onnx_inference.py does - the integration template and parity check.
add_executable(vo_replay tools/vo_replay.cpp)
target_include_directories(vo_replay PRIVATE src)
target_compile_options(vo_replay PRIVATE ${VO_FP_FLAGS})
target_link_libraries(vo_replay PRIVATE vo_stream)

# Image decode / conversion / resize / preprocessing dumps, for parity tests.
add_executable(vo_preprocess tools/vo_preprocess.cpp)
target_compile_options(vo_preprocess PRIVATE ${VO_FP_FLAGS})
target_link_libraries(vo_preprocess PRIVATE vo_stream)

set_target_properties(vo_replay vo_preprocess PROPERTIES BUILD_RPATH ${ORT_LIB_DIR})
message(STATUS "ONNX Runtime: ${ORT_LIB}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
