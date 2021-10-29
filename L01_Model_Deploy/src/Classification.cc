//!
//! \file Classification.cc
//! \author Haisheng Zheng (leo@tiangong.edu.cn)
//! \version 1.0.0
//! \date 2021-10-19
//!

#include "Classification.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "Utils.h"

namespace tiangong {
namespace hpc {
class ClassificationModule::Impl {
 public:
  ~Impl() {}

  void init(const std::string &modelPath) {
    std::vector<char> modelData = loadFile(modelPath);

    // init classification model.
    initEngine(modelData);

    // get node shape.
    getNodeShape();

    // allocate memory
    allocateMemory();
  }

  void run(const tiangong::ClassificationRequest &req,
           tiangong::ClassificationResponse &rsp) {
    preProcess(req.image);
    inferContext_->enqueueV2((void **)inferBuffers_, stream_, nullptr);
    postProcess(rsp);
  }

 private:
  inline std::vector<char> loadFile(const std::string inFileName) {
    std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);
    if (!iFile) {
      return std::vector<char>();
    }
    iFile.seekg(0, std::ifstream::end);
    size_t fsize = iFile.tellg();
    iFile.seekg(0, std::ifstream::beg);
    std::vector<char> content(fsize);
    iFile.read(content.data(), fsize);
    iFile.close();
    return content;
  }

  void initEngine(const std::vector<char> &modelData) {
    inferRuntime_.reset(nvinfer1::createInferRuntime(logger));
    inferEngine_.reset(inferRntime_->deserializeCudaEngine(
        modelData.data(), modelData.size(), nullptr));
    inferContext_.reset(inferEngine_->createExecutionContext());
  }

  void getNodeShape() {
    // input
    nvinfer1::Dims dims = inferEngine_->getBindingDimensions(0);
    inputC_ = std::abs(dims.d[1]);
    inputH_ = std::abs(dims.d[2]);
    inputW_ = std::abs(dims.d[3]);

    // output
    dims = inferEngine_->getBindingDimensions(1);
    if (dims.nbDims == 2) {
      outputH_ = std::abs(dims.d[0]);
      outputW_ = std::abs(dims.d[1]);
    } else {
      outputC_ = std::abs(dims.d[1]);
      outputH_ = std::abs(dims.d[2]);
      outputW_ = std::abs(dims.d[3]);
    }
  }

  void allocateMemory() {
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamCreate(&stream_));

    // alloc gpu memory for tensorrt engine input & output
    CHECK_CUDA(cudaMalloc((void **)&inferInputBuffer_,
                          sizeof(float) * inputH_ * inputW_ * inputC_));
    inferBuffers_[0] = inferInputBuffer_;

    CHECK_CUDA(cudaMalloc((void **)&inferOutputBuffer_,
                          sizeof(float) * outputH_ * outputW_ * outputC_));
    inferBuffers_[1] = inferOutputBuffer_;

    // alloc host memory for copy inference result from gpu memory.
    inferOutputHost_.reset(new float[outputW_ * outputH_ * outputC_]);
  }

  void preProcess(const cv::Mat &inputImage) {}

  void postProcess(tiangong::ClassificationResponse &rsp) {
    CHECK_CUDA(cudaMemcpyAsync(inferOutputHost_.get(), inferOutputBuffer_,
                               sizeof(float) * outputW_ * outputH_,
                               cudaMemcpyDeviceToHost, stream_));

    cudaStreamSynchronize(stream_);
  }

  // storage labels.
  std::vector<std::string> clsLabels_;

  // the object of TensorRT.
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter> inferRuntime_ = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> inferEngine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter> inferContext_ =
      nullptr;

  // cuda stream.
  cudaStream_t stream_ = nullptr;

  // Host memory.
  std::unique_ptr<float[]> inferOutputHost_;

  // device memory.
  float *inferInputBuffer_ = nullptr;
  float *inferOutputBuffer_ = nullptr;
  float *inferBuffers_[2];

  // model info
  uint16_t inputH_;
  uint16_t inputW_;
  uint16_t inputC_;
  uint16_t outputH_;
  uint16_t outputW_;
  uint16_t outputC_;
};

ClassificationModule::ClassificationModule()
    : impl_(std::make_shared<Impl>()){};
ClassificationModule::~ClassificationModule() = default;

// --------------------------------- Interface ---------------------------------
void ClassificationModule::init(const std::string &modelPath) {
  return impl_->init(modelPath);
}

void ClassificationModule::run(const tiangong::ClassificationRequest &req,
                               tiangong::ClassificationResponse &rsp) {
  return impl_->run(req, rsp);
}
}  // namespace hpc
}  // namespace tiangong
