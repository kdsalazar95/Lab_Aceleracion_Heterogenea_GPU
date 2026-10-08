#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "camera_ring_cuda.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kRingCapacity = 4;
constexpr int kCameraWidth = 1920;
constexpr int kCameraHeight = 1080;
const auto kStatsRefreshInterval = std::chrono::milliseconds(500);

struct DisplayStats {
    double fps = 0.0;
    double kernel_ms = 0.0;
    double input_copy_ms = 0.0;
    double cuda_total_ms = 0.0;
    double output_copy_ms = 0.0;
};

class StatsAverager {
public:
    void add_sample(double fps,
                    double kernel_ms,
                    double input_copy_ms,
                    double cuda_total_ms,
                    double output_copy_ms,
                    Clock::time_point now) {
        if (!has_display_stats_) {
            displayed_.fps = fps;
            displayed_.kernel_ms = kernel_ms;
            displayed_.input_copy_ms = input_copy_ms;
            displayed_.cuda_total_ms = cuda_total_ms;
            displayed_.output_copy_ms = output_copy_ms;
            has_display_stats_ = true;
            last_update_ = now;
        }

        fps_sum_ += fps;
        kernel_sum_ += kernel_ms;
        input_copy_sum_ += input_copy_ms;
        cuda_total_sum_ += cuda_total_ms;
        output_copy_sum_ += output_copy_ms;
        ++samples_;

        if (now - last_update_ >= kStatsRefreshInterval && samples_ > 0) {
            displayed_.fps = fps_sum_ / static_cast<double>(samples_);
            displayed_.kernel_ms = kernel_sum_ / static_cast<double>(samples_);
            displayed_.input_copy_ms = input_copy_sum_ / static_cast<double>(samples_);
            displayed_.cuda_total_ms = cuda_total_sum_ / static_cast<double>(samples_);
            displayed_.output_copy_ms = output_copy_sum_ / static_cast<double>(samples_);

            fps_sum_ = 0.0;
            kernel_sum_ = 0.0;
            input_copy_sum_ = 0.0;
            cuda_total_sum_ = 0.0;
            output_copy_sum_ = 0.0;
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
    double input_copy_sum_ = 0.0;
    double cuda_total_sum_ = 0.0;
    double output_copy_sum_ = 0.0;
    int samples_ = 0;
};

struct CapturedFrame {
    cv::Mat bgra;
    int sequence = 0;
    double input_copy_ms = 0.0;
};

struct ProcessedFrame {
    cv::Mat bgra;
    int sequence = 0;
    double input_copy_ms = 0.0;
    double cuda_total_ms = 0.0;
    float kernel_ms = 0.0f;
};

template <typename T>
class RingQueue {
public:
    explicit RingQueue(std::size_t capacity) : buffer_(capacity) {
        if (capacity == 0) {
            throw std::runtime_error("RingQueue capacity must be greater than zero");
        }
    }

    bool push(T item, const std::atomic<bool>& stop_requested) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&] { return count_ < buffer_.size() || closed_ || stop_requested.load(); });
        if (closed_ || stop_requested.load()) {
            return false;
        }

        buffer_[tail_] = std::move(item);
        tail_ = (tail_ + 1) % buffer_.size();
        ++count_;
        not_empty_.notify_one();
        return true;
    }

    bool pop(T& item, const std::atomic<bool>& stop_requested) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return count_ > 0 || closed_ || stop_requested.load(); });
        if (count_ == 0) {
            return false;
        }

        item = std::move(buffer_[head_]);
        head_ = (head_ + 1) % buffer_.size();
        --count_;
        not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::vector<T> buffer_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t count_ = 0;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
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
    const double actual_fps = camera.get(cv::CAP_PROP_FPS);
    std::cout << "Requested camera format: MJPG\n";
    std::cout << "Actual camera format: " << fourcc_to_string(actual_fourcc) << '\n';
    std::cout << "Actual camera FPS: " << actual_fps << '\n';
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

    return bgra.isContinuous() ? bgra.clone() : bgra.clone();
}

std::string slowest_stage(double input_copy_ms, double cuda_total_ms, double output_copy_ms) {
    std::string name = "input copy";
    double value = input_copy_ms;

    if (cuda_total_ms > value) {
        name = "cuda";
        value = cuda_total_ms;
    }
    if (output_copy_ms > value) {
        name = "output copy";
        value = output_copy_ms;
    }

    return name + " " + std::to_string(value) + " ms";
}

void draw_stats(cv::Mat& frame,
                const DisplayStats& stats,
                int sequence) {
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
                "CUDA total: " + std::to_string(stats.cuda_total_ms) + " ms",
                cv::Point(16, 96),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);
    cv::putText(frame,
                "Slowest: " + slowest_stage(stats.input_copy_ms, stats.cuda_total_ms, stats.output_copy_ms),
                cv::Point(16, 128),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);
    cv::putText(frame,
                "Frame: " + std::to_string(sequence),
                cv::Point(16, 160),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(255, 255, 255),
                2,
                cv::LINE_AA);
}

void record_exception(std::exception_ptr error,
                      std::exception_ptr& first_error,
                      std::mutex& error_mutex,
                      std::atomic<bool>& stop_requested) {
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
            first_error = error;
        }
    }
    stop_requested.store(true);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int camera_index = parse_camera_index(argc, argv);

        RingQueue<CapturedFrame> captured_queue(kRingCapacity);
        RingQueue<ProcessedFrame> processed_queue(kRingCapacity);
        std::atomic<bool> stop_requested(false);
        std::exception_ptr first_error = nullptr;
        std::mutex error_mutex;

        std::thread capture_thread([&] {
            try {
                cv::VideoCapture camera("v4l2src device=/dev/video2 ! image/jpeg,width=1920,height=1080 ! jpegdec ! "
                                "videoconvert ! video/x-raw,format=BGR ! appsink");
                if (!camera.isOpened()) {
                    throw std::runtime_error("Could not open camera index: " + std::to_string(camera_index));
                }
                configure_camera_720p_mjpeg(camera);

                int sequence = 0;
                while (!stop_requested.load()) {
                    cv::Mat frame;
                    camera >> frame;
                    if (frame.empty()) {
                        throw std::runtime_error("Captured an empty frame");
                    }

                    const auto input_copy_start = std::chrono::steady_clock::now();
                    CapturedFrame captured;
                    captured.bgra = to_bgra(frame);
                    captured.sequence = sequence++;
                    const auto input_copy_end = std::chrono::steady_clock::now();
                    captured.input_copy_ms =
                        std::chrono::duration<double, std::milli>(input_copy_end - input_copy_start).count();

                    if (!captured_queue.push(std::move(captured), stop_requested)) {
                        break;
                    }
                }
            } catch (...) {
                record_exception(std::current_exception(), first_error, error_mutex, stop_requested);
            }
            captured_queue.close();
        });

        std::thread cuda_thread([&] {
            try {
                std::unique_ptr<CudaFrameInverter> inverter;
                int width = 0;
                int height = 0;

                CapturedFrame captured;
                while (captured_queue.pop(captured, stop_requested)) {
                    if (inverter == nullptr || captured.bgra.cols != width || captured.bgra.rows != height) {
                        width = captured.bgra.cols;
                        height = captured.bgra.rows;
                        inverter = std::make_unique<CudaFrameInverter>(width, height);
                    }

                    ProcessedFrame processed;
                    processed.sequence = captured.sequence;
                    processed.input_copy_ms = captured.input_copy_ms;
                    const auto cuda_start = std::chrono::steady_clock::now();
                    processed.bgra = inverter->invert(captured.bgra, processed.kernel_ms);
                    const auto cuda_end = std::chrono::steady_clock::now();
                    processed.cuda_total_ms =
                        std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();

                    if (!processed_queue.push(std::move(processed), stop_requested)) {
                        break;
                    }
                }
            } catch (...) {
                record_exception(std::current_exception(), first_error, error_mutex, stop_requested);
            }
            processed_queue.close();
        });

        std::thread display_thread([&] {
            try {
                const std::string window_name = "Live CUDA Ring Color Inversion";
                cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

                auto previous_time = std::chrono::steady_clock::now();
                StatsAverager stats_averager;

                ProcessedFrame processed;
                while (processed_queue.pop(processed, stop_requested)) {
                    const auto current_time = std::chrono::steady_clock::now();
                    const double elapsed = std::chrono::duration<double>(current_time - previous_time).count();
                    previous_time = current_time;

                    double fps = 0.0;
                    if (elapsed > 0.0) {
                        fps = 1.0 / elapsed;
                    }

                    const auto output_copy_start = std::chrono::steady_clock::now();
                    cv::Mat display_frame = processed.bgra.clone();
                    const auto output_copy_end = std::chrono::steady_clock::now();
                    const double output_copy_ms =
                        std::chrono::duration<double, std::milli>(output_copy_end - output_copy_start).count();

                    stats_averager.add_sample(fps,
                                              static_cast<double>(processed.kernel_ms),
                                              processed.input_copy_ms,
                                              processed.cuda_total_ms,
                                              output_copy_ms,
                                              output_copy_end);

                    draw_stats(display_frame, stats_averager.displayed(), processed.sequence);
                    cv::imshow(window_name, display_frame);

                    const int key = cv::waitKey(1);

                    if (key == 27 || key == 'q' || key == 'Q') {
                        stop_requested.store(true);
                        captured_queue.close();
                        processed_queue.close();
                        break;
                    }
                }

                cv::destroyWindow(window_name);
            } catch (...) {
                record_exception(std::current_exception(), first_error, error_mutex, stop_requested);
            }
            processed_queue.close();
        });

        std::cout << "Camera index: " << camera_index << '\n';
        std::cout << "Ring capacity: " << kRingCapacity << '\n';
        std::cout << "Press q or Esc to exit\n";

        capture_thread.join();
        cuda_thread.join();
        display_thread.join();

        if (first_error != nullptr) {
            std::rethrow_exception(first_error);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
