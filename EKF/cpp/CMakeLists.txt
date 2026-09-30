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
