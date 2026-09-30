cmake_minimum_required(VERSION 3.10)
project(imuvo_ekf CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The filter: link this into the Jetson application.
add_library(imuvo_ekf STATIC src/imuvo_ekf.cpp)
target_include_directories(imuvo_ekf PUBLIC include)
target_compile_options(imuvo_ekf PRIVATE -Wall -Wextra)

# Replay a message log (testing and integration template).
add_executable(ekf_replay tools/ekf_replay.cpp)
target_link_libraries(ekf_replay PRIVATE imuvo_ekf)

# ---------------------------------------------------------------- optional: IMU model
# The learned IMU correction needs ONNX Runtime (C++ API).  Point ONNXRUNTIME_DIR at an
# unpacked release (include/ + lib/), e.g. onnxruntime-linux-aarch64-<ver> on the Jetson:
#   cmake -S . -B build -DONNXRUNTIME_DIR=/opt/onnxruntime
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime root (include/, lib/)")
if(ONNXRUNTIME_DIR)
  find_library(ORT_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib NO_DEFAULT_PATH REQUIRED)
  add_library(imuvo_imu_model STATIC src/imuvo_imu_model.cpp)
  target_include_directories(imuvo_imu_model PUBLIC include ${ONNXRUNTIME_DIR}/include)
  target_link_libraries(imuvo_imu_model PUBLIC imuvo_ekf ${ORT_LIB})
  add_executable(imu_correct_replay tools/imu_correct_replay.cpp)
  target_link_libraries(imu_correct_replay PRIVATE imuvo_imu_model)
  set_target_properties(imu_correct_replay PROPERTIES BUILD_RPATH ${ONNXRUNTIME_DIR}/lib)
  message(STATUS "IMU model support: ON (${ORT_LIB})")
else()
  message(STATUS "IMU model support: OFF (set -DONNXRUNTIME_DIR=... to build it)")
endif()
