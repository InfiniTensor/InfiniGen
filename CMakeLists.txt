cmake_minimum_required(VERSION 3.17)
project(InfiniTensor C CXX)

option(USE_CUDA "Support CUDA GPU" OFF)
option(USE_BANG "Support BANG MLU" OFF)

option(BUILD_TEST "Build test code" ON)
option(BUILD_ASAN "Build code whith ASAN" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

################################################################################
# ASAN Check
################################################################################
if(BUILD_ASAN)
  set(CMAKE_ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_ASAN_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ASAN_FLAGS}")
endif()

include_directories(include)
include_directories(tests)

include_directories(3rd-party/fmt/include)
# add_subdirectory(3rd-party/fmt)

file(GLOB_RECURSE SOURCE source/core/*.cpp source/operators/*.cpp source/kernels/*.cpp source/micros/*.cpp source/graphs/*.cpp)
add_library(Thread SHARED ${SOURCE})

add_compile_options(-Wall)

# 将 fmt 库链接到 Thread 库
target_link_directories(Thread PRIVATE 3rd-party/fmt/build)
target_link_libraries(Thread PRIVATE fmt)

if(BUILD_TEST)
  file(GLOB_RECURSE TEST tests/*.cpp)
  foreach(file ${TEST})
    get_filename_component(filename ${file} NAME_WE)
    add_executable(${filename} ${file})
    target_link_libraries(${filename} Thread)
    add_test(NAME ${filename} COMMAND ${filename})
  endforeach()
endif()



