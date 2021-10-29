//!
//! \file Demo.cc
//! \author Haisheng Zheng (leo@tiangong.edu.cn)
//! \version 1.0.0
//! \date 2021-10-19
//!

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "Classification.h"

int main(int argc, char *argv[]) {
  std::string modelPath = argv[1];
  std::string imagePath = argv[2];

  std::unique_ptr<tiangong::hpc::ClassificationModule> mModel{nullptr};
  mModel = std::make_unique<tiangong::ClassificationModule>();

  mModel->init(modelPath);
  cv::Mat image = cv::imread(imagePath);

  tiangong::hpc::ClassificationRequest req;
  tiangong::hpc::ClassificationResponse rsp;

  req.image = image;

  auto start = std::chrono::high_resolution_clock::now();

  //  for(auto i = 0; i < 1000; i++)
  mModel->run(req, rsp);

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "[Info] Latency: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                       .count() /
                   1000.0
            << "ms\n";
}
