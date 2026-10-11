#include "invert_cuda.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void invert_rgba_kernel(const uchar4* input, uchar4* output, int total_pixels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) {
        return;
    }

    const uchar4 pixel = input[idx];
    output[idx] = make_uchar4(
        static_cast<unsigned char>(255 - pixel.x),
        static_cast<unsigned char>(255 - pixel.y),
        static_cast<unsigned char>(255 - pixel.z),
        pixel.w);
}

}  // namespace

cv::Mat invert_rgba_cuda(const cv::Mat& input_bgra, float& kernel_ms) {
    CV_Assert(input_bgra.type() == CV_8UC4);
    CV_Assert(input_bgra.isContinuous());

    const int total_pixels = input_bgra.rows * input_bgra.cols;
    const std::size_t image_bytes = static_cast<std::size_t>(total_pixels) * sizeof(uchar4);

    uchar4* device_input = nullptr;
    uchar4* device_output = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    try {
        check_cuda(cudaMalloc(&device_input, image_bytes), "cudaMalloc input");
        check_cuda(cudaMalloc(&device_output, image_bytes), "cudaMalloc output");
        check_cuda(cudaMemcpy(device_input, input_bgra.ptr<uchar4>(), image_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");
        check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&kernel_stop), "cudaEventCreate stop");

        constexpr int threads_per_block = 256;
        const int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

        check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord start");
        invert_rgba_kernel<<<blocks, threads_per_block>>>(device_input, device_output, total_pixels);
        check_cuda(cudaGetLastError(), "invert_rgba_kernel launch");
        check_cuda(cudaEventRecord(kernel_stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(kernel_stop), "cudaEventSynchronize stop");
        check_cuda(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop), "cudaEventElapsedTime");

        cv::Mat output_bgra(input_bgra.size(), input_bgra.type());
        check_cuda(cudaMemcpy(output_bgra.ptr<uchar4>(), device_output, image_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host");

        check_cuda(cudaEventDestroy(kernel_start), "cudaEventDestroy start");
        check_cuda(cudaEventDestroy(kernel_stop), "cudaEventDestroy stop");
        check_cuda(cudaFree(device_input), "cudaFree input");
        check_cuda(cudaFree(device_output), "cudaFree output");

        return output_bgra;
    } catch (...) {
        if (kernel_start != nullptr) {
            cudaEventDestroy(kernel_start);
        }
        if (kernel_stop != nullptr) {
            cudaEventDestroy(kernel_stop);
        }
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        throw;
    }
}
