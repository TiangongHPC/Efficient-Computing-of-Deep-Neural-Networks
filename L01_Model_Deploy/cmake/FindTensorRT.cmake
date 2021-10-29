find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_DIR}/include)

find_library(TENSORRT_LIBRARY_INFER
    NAMES nvinfer
    PATHS ${TENSORRT_DIR}/lib)

find_library(TENSORRT_LIBRARY_ONNXPARSER
    NAMES nvonnxparser
    PATHS ${TENSORRT_DIR}/lib)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT DEFAULT_MSG
    TENSORRT_INCLUDE_DIR 
    TENSORRT_LIBRARY_INFER TENSORRT_LIBRARY_ONNXPARSER)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
  set(TensorRT_LIBRARIES ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
  message(STATUS "Found TensorRT: (include: ${TensorRT_INCLUDE_DIRS}, library: ${TensorRT_LIBRARIES})")
endif()
