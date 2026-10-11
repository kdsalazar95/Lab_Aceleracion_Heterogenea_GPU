#pragma once

#include <opencv2/core.hpp>

cv::Mat invert_rgba_cuda(const cv::Mat& input_bgra, float& kernel_ms);
