// GPU v0 (linea base): porteo directo a CUDA de simulate_step() y
// render_frame(), sin optimizaciones. Mantiene el mismo flujo que la version
// CPU (src/main.cpp): las mallas viven en el host y en cada paso se copian a
// la GPU y de vuelta. El texto del cuadro y la escritura del video siguen en CPU.
//
// Uso:
//   ./build/drop_simulation_gpu_v0                 # genera el video y mide
//   ./build/drop_simulation_gpu_v0 --validate      # ademas compara contra CPU
//   ./build/drop_simulation_gpu_v0 --output <ruta.mp4> --csv <errores.csv>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "../../common/config.hpp"
#include "../../common/cpu_reference.hpp"
#include "kernels.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    bool validate = false;
    std::string output = "output/drop_simulation_gpu_v0.mp4";
    std::string csv = "validation/v0_errors.csv";
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--validate") {
            opts.validate = true;
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            opts.csv = argv[++i];
        } else {
            throw std::runtime_error("Argumento no reconocido: " + arg +
                                     "\nUso: " + argv[0] + " [--validate] [--output ruta.mp4] [--csv errores.csv]");
        }
    }
    return opts;
}

void ensure_parent_dir(const std::string& path) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
}

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Acumula el error de la malla y de la imagen GPU respecto a la CPU.
struct Validator {
    const Config& cfg;
    std::vector<float> previous, current, next;
    std::vector<unsigned char> gray;
    std::ofstream csv;

    double max_abs_grid = 0.0;
    double sum_mean_abs_grid = 0.0;
    int max_abs_pixel = 0;
    long long total_pixels_diff = 0;
    int frames = 0;

    Validator(const Config& c, const std::vector<float>& prev0, const std::vector<float>& cur0, const std::string& csv_path)
        : cfg(c), previous(prev0), current(cur0), next(cur0.size(), 0.0f), gray(cur0.size(), 0) {
        ensure_parent_dir(csv_path);
        csv.open(csv_path);
        if (!csv) {
            throw std::runtime_error("No se pudo abrir el CSV de validacion: " + csv_path);
        }
        csv << "frame,mean_abs_err,max_abs_err,max_pixel_diff,pixels_diff\n";
    }

    void advance() {
        for (int step = 0; step < cfg.steps_per_frame; ++step) {
            cpu_ref::simulate_step(previous, current, next, cfg);
            previous.swap(current);
            current.swap(next);
        }
    }

    void compare(int frame, const std::vector<float>& gpu_grid, const cv::Mat& gpu_image) {
        double sum = 0.0;
        double max_abs = 0.0;
        for (std::size_t i = 0; i < current.size(); ++i) {
            const double err = std::fabs(static_cast<double>(gpu_grid[i]) - static_cast<double>(current[i]));
            sum += err;
            max_abs = std::max(max_abs, err);
        }
        const double mean_abs = sum / static_cast<double>(current.size());

        cpu_ref::render_gray(current, cfg, gray);
        int max_pix = 0;
        long long pix_diff = 0;
        const unsigned char* img = gpu_image.ptr<unsigned char>();
        for (std::size_t i = 0; i < gray.size(); ++i) {
            const int d = std::abs(static_cast<int>(img[3 * i]) - static_cast<int>(gray[i]));
            max_pix = std::max(max_pix, d);
            pix_diff += (d != 0);
        }

        csv << frame << ',' << std::scientific << std::setprecision(6) << mean_abs << ',' << max_abs
            << std::defaultfloat << ',' << max_pix << ',' << pix_diff << '\n';

        max_abs_grid = std::max(max_abs_grid, max_abs);
        sum_mean_abs_grid += mean_abs;
        max_abs_pixel = std::max(max_abs_pixel, max_pix);
        total_pixels_diff += pix_diff;
        ++frames;
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        Config cfg;
        cfg.output = opts.output;

        const int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);

        ensure_parent_dir(cfg.output);

        std::vector<float> previous(cells, 0.0f);
        std::vector<float> current(cells, 0.0f);
        std::vector<float> next(cells, 0.0f);

        // La perturbacion inicial se ejecuta una sola vez: se deja en CPU.
        cpu_ref::add_drop(current, previous, cfg);

        const cv::Vec3f light = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));
        const SimParams params{cfg.width,
                               cfg.height,
                               cfg.wave_speed * cfg.wave_speed,
                               cfg.damping,
                               cfg.edge_damping,
                               light[0],
                               light[1],
                               light[2]};

        std::cout << "GPU: " << gpu_device_description() << '\n';
        GpuSimulator gpu(params);

        std::unique_ptr<Validator> validator;
        if (opts.validate) {
            validator = std::make_unique<Validator>(cfg, previous, current, opts.csv);
            std::cout << "Modo validacion activo: los tiempos NO son representativos.\n";
        }

        cv::VideoWriter writer(cfg.output,
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               static_cast<double>(cfg.fps),
                               cv::Size(cfg.width, cfg.height));
        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el archivo de salida: " + cfg.output);
        }

        cv::Mat image(cfg.height, cfg.width, CV_8UC3);
        double video_ms = 0.0;

        const auto start = Clock::now();

        for (int frame = 0; frame < total_frames; ++frame) {
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                gpu.step(previous.data(), current.data(), next.data());
                previous.swap(current);
                current.swap(next);
            }

            gpu.render(current.data(), image.ptr<unsigned char>());

            if (validator) {
                validator->advance();
                validator->compare(frame, current, image);
            }

            const auto video_start = Clock::now();
            cv::putText(image,
                        "GPU v0 float32 | frame " + std::to_string(frame),
                        cv::Point(18, 32),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.65,
                        cv::Scalar(235, 235, 235),
                        1,
                        cv::LINE_AA);
            writer.write(image);
            video_ms += ms_since(video_start);

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << total_frames << '\n';
            }
        }

        const double elapsed = ms_since(start) / 1000.0;
        const double simulated_steps = static_cast<double>(total_frames) * cfg.steps_per_frame;
        const GpuTimings& t = gpu.timings();

        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo: " << elapsed << " s\n";
        std::cout << "Pasos simulados: " << simulated_steps << '\n';
        std::cout << "Rendimiento: " << simulated_steps / elapsed << " pasos/s\n";
        std::cout << "--- Desglose (cudaEvent, acumulado) ---\n";
        std::cout << "Kernel simulate_step: " << t.simulate_kernel_ms << " ms\n";
        std::cout << "Kernel render:        " << t.render_kernel_ms << " ms\n";
        std::cout << "Copias H2D:           " << t.h2d_ms << " ms (" << t.h2d_bytes / (1024.0 * 1024.0) << " MiB)\n";
        std::cout << "Copias D2H:           " << t.d2h_ms << " ms (" << t.d2h_bytes / (1024.0 * 1024.0) << " MiB)\n";
        std::cout << "putText + video (CPU): " << video_ms << " ms\n";

        if (validator) {
            std::cout << "--- Validacion vs CPU (" << validator->frames << " cuadros) ---\n";
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "Error abs. maximo malla:  " << validator->max_abs_grid << '\n';
            std::cout << "Error abs. medio malla:   " << validator->sum_mean_abs_grid / validator->frames << '\n';
            std::cout << std::defaultfloat;
            std::cout << "Dif. maxima de pixel:     " << validator->max_abs_pixel << " niveles de gris\n";
            std::cout << "Pixeles distintos (total): " << validator->total_pixels_diff << '\n';
            std::cout << "Detalle por cuadro: " << opts.csv << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
