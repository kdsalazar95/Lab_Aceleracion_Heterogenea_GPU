#pragma once

#include <memory>

#include <opencv2/core.hpp>

class CudaFrameInverter {
public:
    CudaFrameInverter(int width, int height);
    ~CudaFrameInverter();

    CudaFrameInverter(const CudaFrameInverter&) = delete;
    CudaFrameInverter& operator=(const CudaFrameInverter&) = delete;

    cv::Mat invert(const cv::Mat& input_bgra, float& kernel_ms);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
