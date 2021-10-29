//!
//! \file Classification.h
//! \author leo.zheng (leo.zheng@smartmore.com)
//! \version 1.0.0
//! \date 2021-06-30
//! \copyright Copyright (c) 2021
//!

#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

namespace tiangong {
namespace hpc {
struct ClassificationRequest {
  cv::Mat image;
  /* data */
};

struct ClassificationResponse {
  /* data */
};

class ClassificationModule {
 public:
  ClassificationModule();
  ~ClassificationModule();
  void init(const std::string &modelPath);
  void run(const tiangong::ClassificationRequest &req,
           tiangong::ClassificationResponse &rsp);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace hpc
}  // namespace tiangong
#endif  // CLASSIFICATION_H_
