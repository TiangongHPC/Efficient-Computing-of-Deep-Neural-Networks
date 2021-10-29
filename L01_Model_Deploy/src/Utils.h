//!
//! \file utils.h
//! \author Haisheng Zheng (leo@tiangong.edu.cn)
//! \version 1.0.0
//! \date 2021-10-19
//!

#ifndef UTILS_H_
#define UTILS_H_

#define CHECK_CUDA(e)                                           \
  {                                                             \
    if (e != cudaSuccess) {                                     \
      printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                            \
      exit(0);                                                  \
    }                                                           \
  }

struct NvInferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
};

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    using namespace nvinfer1;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cout << "[Internal] " + std::string(msg);
        break;
      case Severity::kERROR:
        std::cout << "[Error]" << msg << std::endl;
        break;
      default:
        break;
    }
  }
  ~Logger() {}
};

#endif  // PROJECT_UTILS_H