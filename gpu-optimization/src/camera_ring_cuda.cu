#include "camera_ring_cuda.hpp"

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

struct CudaFrameInverter::Impl {
    int width = 0;
    int height = 0;
    int total_pixels = 0;
    std::size_t image_bytes = 0;
    uchar4* device_input = nullptr;
    uchar4* device_output = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    Impl(int frame_width, int frame_height)
        : width(frame_width),
          height(frame_height),
          total_pixels(frame_width * frame_height),
          image_bytes(static_cast<std::size_t>(total_pixels) * sizeof(uchar4)) {
        check_cuda(cudaMalloc(&device_input, image_bytes), "cudaMalloc input");
        check_cuda(cudaMalloc(&device_output, image_bytes), "cudaMalloc output");
        check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&kernel_stop), "cudaEventCreate stop");
    }

    ~Impl() {
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
    }
};

CudaFrameInverter::CudaFrameInverter(int width, int height) : impl_(std::make_unique<Impl>(width, height)) {}

CudaFrameInverter::~CudaFrameInverter() = default;

cv::Mat CudaFrameInverter::invert(const cv::Mat& input_bgra, float& kernel_ms) {
    CV_Assert(input_bgra.type() == CV_8UC4);
    CV_Assert(input_bgra.isContinuous());

    if (input_bgra.cols != impl_->width || input_bgra.rows != impl_->height) {
        throw std::runtime_error("Frame size changed after CUDA buffers were created");
    }

    check_cuda(cudaMemcpy(impl_->device_input,
                          input_bgra.ptr<uchar4>(),
                          impl_->image_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");

    constexpr int threads_per_block = 256;
    const int blocks = (impl_->total_pixels + threads_per_block - 1) / threads_per_block;

    check_cuda(cudaEventRecord(impl_->kernel_start), "cudaEventRecord start");
    invert_rgba_kernel<<<blocks, threads_per_block>>>(impl_->device_input, impl_->device_output, impl_->total_pixels);
    check_cuda(cudaGetLastError(), "invert_rgba_kernel launch");
    check_cuda(cudaEventRecord(impl_->kernel_stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(impl_->kernel_stop), "cudaEventSynchronize stop");
    check_cuda(cudaEventElapsedTime(&kernel_ms, impl_->kernel_start, impl_->kernel_stop), "cudaEventElapsedTime");

    cv::Mat output_bgra(input_bgra.size(), input_bgra.type());
    check_cuda(cudaMemcpy(output_bgra.ptr<uchar4>(),
                          impl_->device_output,
                          impl_->image_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy device to host");

    return output_bgra;
}
