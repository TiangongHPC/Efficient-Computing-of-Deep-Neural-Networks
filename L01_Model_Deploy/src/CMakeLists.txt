# file(GLOB_RECURSE srcs *.cc *.cpp *.h)
file(GLOB_RECURSE srcs *.cu *.cc *.cpp *.h)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

cuda_add_library(tiangongClassification SHARED ${srcs})

target_include_directories(tiangongClassification PRIVATE ${OPENCV_INCLUDE_DIR}
                           ${TENSORRT_INCLUDE_DIR})

target_link_libraries(tiangongClassification
    ${OpenCV_LIBRARIES}
    ${TensorRT_LIBRARIES}
)
