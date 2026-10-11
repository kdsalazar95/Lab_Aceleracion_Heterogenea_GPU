#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "invert_cuda.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kCameraWidth = 1920;
constexpr int kCameraHeight = 1080;
const auto kStatsRefreshInterval = std::chrono::milliseconds(500);

struct DisplayStats {
    double fps = 0.0;
    double kernel_ms = 0.0;
    double total_cuda_ms = 0.0;
};

class StatsAverager {
public:
    void add_sample(double fps, double kernel_ms, double total_cuda_ms, Clock::time_point now) {
        if (!has_display_stats_) {
            displayed_.fps = fps;
            displayed_.kernel_ms = kernel_ms;
            displayed_.total_cuda_ms = total_cuda_ms;
            has_display_stats_ = true;
            last_update_ = now;
        }

        fps_sum_ += fps;
        kernel_sum_ += kernel_ms;
        total_cuda_sum_ += total_cuda_ms;
        ++samples_;

        if (now - last_update_ >= kStatsRefreshInterval && samples_ > 0) {
            displayed_.fps = fps_sum_ / static_cast<double>(samples_);
            displayed_.kernel_ms = kernel_sum_ / static_cast<double>(samples_);
            displayed_.total_cuda_ms = total_cuda_sum_ / static_cast<double>(samples_);

            fps_sum_ = 0.0;
            kernel_sum_ = 0.0;
            total_cuda_sum_ = 0.0;
            samples_ = 0;
            last_update_ = now;
        }
    }

    const DisplayStats& displayed() const {
        return displayed_;
    }

private:
    DisplayStats displayed_;
    bool has_display_stats_ = false;
    Clock::time_point last_update_ = Clock::now();
    double fps_sum_ = 0.0;
    double kernel_sum_ = 0.0;
    double total_cuda_sum_ = 0.0;
    int samples_ = 0;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [camera_index]\n";
}

int parse_camera_index(int argc, char** argv) {
    if (argc > 2) {
        print_usage(argv[0]);
        throw std::runtime_error("Too many arguments");
    }

    if (argc == 1) {
        return 0;
    }

    try {
        return std::stoi(argv[1]);
    } catch (const std::exception&) {
        print_usage(argv[0]);
        throw std::runtime_error("Camera index must be an integer");
    }
}

std::string fourcc_to_string(int fourcc) {
    std::string value(4, ' ');
    value[0] = static_cast<char>(fourcc & 0xFF);
    value[1] = static_cast<char>((fourcc >> 8) & 0xFF);
    value[2] = static_cast<char>((fourcc >> 16) & 0xFF);
    value[3] = static_cast<char>((fourcc >> 24) & 0xFF);
    return value;
}

void configure_camera_720p_mjpeg(cv::VideoCapture& camera) {
    const int camera_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    camera.set(cv::CAP_PROP_FOURCC, camera_fourcc);
    camera.set(cv::CAP_PROP_FRAME_WIDTH, kCameraWidth);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, kCameraHeight);

    const int actual_width = static_cast<int>(camera.get(cv::CAP_PROP_FRAME_WIDTH));
    const int actual_height = static_cast<int>(camera.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int actual_fourcc = static_cast<int>(camera.get(cv::CAP_PROP_FOURCC));
    std::cout << "Requested camera format: MJPG\n";
    std::cout << "Actual camera format: " << fourcc_to_string(actual_fourcc) << '\n';
    std::cout << "Requested camera size: " << kCameraWidth << " x " << kCameraHeight << '\n';
    std::cout << "Actual camera size: " << actual_width << " x " << actual_height << '\n';
}

cv::Mat to_bgra(const cv::Mat& frame) {
    cv::Mat bgra;
    switch (frame.channels()) {
        case 1:
            cv::cvtColor(frame, bgra, cv::COLOR_GRAY2BGRA);
            break;
        case 3:
            cv::cvtColor(frame, bgra, cv::COLOR_BGR2BGRA);
            break;
        case 4:
            bgra = frame;
            break;
        default:
            throw std::runtime_error("Unsupported camera frame channels: " + std::to_string(frame.channels()));
    }

    if (bgra.depth() != CV_8U) {
        cv::Mat converted;
        bgra.convertTo(converted, CV_8UC4);
        return converted.isContinuous() ? converted : converted.clone();
    }

    return bgra.isContinuous() ? bgra : bgra.clone();
}

void draw_stats(cv::Mat& frame, const DisplayStats& stats) {
    cv::putText(frame,
                "FPS: " + std::to_string(static_cast<int>(stats.fps + 0.5)),
                cv::Point(16, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);
    cv::putText(frame,
                "CUDA kernel: " + std::to_string(stats.kernel_ms) + " ms",
                cv::Point(16, 64),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);
    cv::putText(frame,
                "CUDA total: " + std::to_string(stats.total_cuda_ms) + " ms",
                cv::Point(16, 96),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int camera_index = parse_camera_index(argc, argv);

        cv::VideoCapture camera("v4l2src device=/dev/video2 ! image/jpeg,width=1920,height=1080 ! jpegdec ! "
                                "videoconvert ! video/x-raw,format=BGR ! appsink");
        if (!camera.isOpened()) {
            throw std::runtime_error("Could not open camera index: " + std::to_string(camera_index));
        }
        configure_camera_720p_mjpeg(camera);

        std::cout << "Camera index: " << camera_index << '\n';
        std::cout << "Press q or Esc to exit\n";

        const std::string window_name = "Live CUDA Color Inversion";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

        cv::Mat frame;
        auto previous_time = std::chrono::steady_clock::now();
        StatsAverager stats_averager;

        while (true) {
            camera >> frame;
            if (frame.empty()) {
                throw std::runtime_error("Captured an empty frame");
            }

            const auto current_time = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(current_time - previous_time).count();
            previous_time = current_time;

            double fps = 0.0;
            if (elapsed > 0.0) {
                fps = 1.0 / elapsed;
            }

            const cv::Mat input_bgra = to_bgra(frame);
            float kernel_ms = 0.0f;
            const auto cuda_start = std::chrono::steady_clock::now();
            cv::Mat inverted = invert_rgba_cuda(input_bgra, kernel_ms);
            const auto cuda_end = std::chrono::steady_clock::now();
            const double total_cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();

            stats_averager.add_sample(fps, static_cast<double>(kernel_ms), total_cuda_ms, cuda_end);
            draw_stats(inverted, stats_averager.displayed());
            cv::imshow(window_name, inverted);

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
