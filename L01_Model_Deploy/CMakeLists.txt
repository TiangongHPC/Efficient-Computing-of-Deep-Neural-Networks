cmake_minimum_required(VERSION 3.10)

set(VIAIR_SEGMENTATION_MAJOR_VERSION 1)
set(VIAIR_SEGMENTATION_MINOR_VERSION 2)
set(VIAIR_SEGMENTATION_PATCH_VERSION 0)

set(VIAIR_SEGMENTATION_VERSION "${VIAIR_SEGMENTATION_MAJOR_VERSION}.\
${VIAIR_SEGMENTATION_MINOR_VERSION}.\
${VIAIR_SEGMENTATION_PATCH_VERSION}")

project(viair-segmentation-sdk LANGUAGES CUDA CXX)

option(VIAIR_SEGMENTATION_EXAMPLE "Build The Example of ViAir Segmentation Usage" ON)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fuse-ld=gold")
endif()

if(DEBUG)
    add_definitions(-DDEBUG)
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "set build type to debug" FORCE)
    endif()
else()
    if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "set build type to release" FORCE)
    endif()
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

set(CMAKE_CUDA_ARCHITECTURES 72)

message(STATUS ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
message(STATUS "ViAir Segmentation Build INFO:")
message(STATUS "Version: ${VIAIR_SEGMENTATION_VERSION}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Example: ${VIAIR_SEGMENTATION_EXAMPLE}")
message(STATUS "Debug: ${DEBUG}")
message(STATUS ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

include_directories(include)

set(OPENCV_DIR ${PROJECT_SOURCE_DIR}/thirdParty/opencv)
set(TENSORRT_DIR /home/leozheng/TensorRT-8.0.1.6)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out)

find_package(OpenCV REQUIRED)

add_subdirectory(src)

if(VIAIR_SEGMENTATION_EXAMPLE)
    add_subdirectory(tutorials)
endif()
