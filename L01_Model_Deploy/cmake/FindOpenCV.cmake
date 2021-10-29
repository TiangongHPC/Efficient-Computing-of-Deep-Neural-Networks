message("FOUNDING OPENCV")

find_path(OPENCV_INCLUDE_DIR
    NAMES opencv2/opencv.hpp
    PATHS ${OPENCV_DIR}/include)

find_library(OPENCV_LIBRARY_CORE
    NAMES opencv_core
    PATHS ${OPENCV_DIR}/lib)

find_library(OPENCV_LIBRARY_IMGCODECS
    NAMES opencv_imgcodecs
    PATHS ${OPENCV_DIR}/lib)

find_library(OPENCV_LIBRARY_IMGPROC
    NAMES opencv_imgproc
    PATHS ${OPENCV_DIR}/lib)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCV DEFAULT_MSG
    OPENCV_INCLUDE_DIR 
    OPENCV_LIBRARY_CORE
    OPENCV_LIBRARY_IMGCODECS
    OPENCV_LIBRARY_IMGPROC)

if(OpenCV_FOUND)
  set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIR})
  set(OpenCV_LIBRARIES ${OPENCV_LIBRARY_CORE} ${OPENCV_LIBRARY_IMGCODECS} ${OPENCV_LIBRARY_IMGPROC})
  message(STATUS "Found OpenCV: (include: ${OpenCV_INCLUDE_DIRS}, library: ${OpenCV_LIBRARIES})")
endif()
