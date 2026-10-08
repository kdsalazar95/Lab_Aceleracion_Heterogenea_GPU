#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "invert_cuda.hpp"

namespace {

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <input_image> <output_image>\n";
}

cv::Mat read_as_bgra(const std::string& input_path) {
    const cv::Mat image = cv::imread(input_path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("Could not read input image: " + input_path);
    }

    cv::Mat bgra;
    switch (image.channels()) {
        case 1:
            cv::cvtColor(image, bgra, cv::COLOR_GRAY2BGRA);
            break;
        case 3:
            cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
            break;
        case 4:
            bgra = image;
            break;
        default:
            throw std::runtime_error("Unsupported number of channels: " + std::to_string(image.channels()));
    }

    if (bgra.depth() != CV_8U) {
        cv::Mat converted;
        bgra.convertTo(converted, CV_8UC4);
        return converted.isContinuous() ? converted : converted.clone();
    }

    return bgra.isContinuous() ? bgra : bgra.clone();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];

        const cv::Mat input_bgra = read_as_bgra(input_path);

        float kernel_ms = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        const cv::Mat output_bgra = invert_rgba_cuda(input_bgra, kernel_ms);
        const auto end = std::chrono::steady_clock::now();

        if (!cv::imwrite(output_path, output_bgra)) {
            throw std::runtime_error("Could not write output image: " + output_path);
        }

        const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        const double megapixels =
            static_cast<double>(input_bgra.rows) * static_cast<double>(input_bgra.cols) / 1'000'000.0;

        std::cout << "Input: " << input_path << '\n';
        std::cout << "Output: " << output_path << '\n';
        std::cout << "Size: " << input_bgra.cols << " x " << input_bgra.rows << '\n';
        std::cout << "CUDA kernel time: " << kernel_ms << " ms\n";
        std::cout << "Total CUDA path time: " << total_ms << " ms\n";
        std::cout << "Kernel throughput: " << megapixels / (static_cast<double>(kernel_ms) / 1000.0)
                  << " MPixels/s\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
