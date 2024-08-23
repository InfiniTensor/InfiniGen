cmake_minimum_required(VERSION 3.12)
project(InfiniGen C CXX)

option(USE_CUDA "Support CUDA GPU" OFF)
option(USE_BANG "Support BANG MLU" OFF)

option(BUILD_TEST "Build test code" ON)
option(BUILD_ASAN "Build code whith ASAN" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Configuring for Debug build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    add_compile_definitions(DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Configuring for Release build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    add_compile_definitions(NDEBUG)
else()
    message("Build type not specified. Configuring for RelWithDebInfo build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")
endif()

################################################################################
# ASAN Check
################################################################################
if(BUILD_ASAN)
  set(CMAKE_ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_ASAN_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ASAN_FLAGS}")
endif()

include_directories(include)

include_directories(3rd-party/fmt/include)
add_subdirectory(3rd-party/fmt)
add_definitions(-D FMT_HEADER_ONLY)

file(GLOB_RECURSE SOURCE source/core/*.cpp source/micros/*/*.cpp)
add_library(infiniGen SHARED ${SOURCE})

if(BUILD_TEST)
  file(GLOB_RECURSE TEST test/*.cpp)
  foreach(file ${TEST})
    include_directories(test)
    get_filename_component(filename ${file} NAME_WE)
    add_executable(${filename} ${file})
    target_link_libraries(${filename} infiniGen)
    add_test(NAME ${filename} COMMAND ${filename})
  endforeach()
endif()


