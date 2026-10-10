// =============================================================================
// drop_simulation.cpp - Simulación de una gota sobre un estanque
// Laboratorio Semana 9 - Computación Heterogénea
//
// El programa representa la superficie del agua mediante una malla 2D y calcula
// cómo se propagan las ondas usando una ecuación de onda amortiguada. Al final,
// genera un video en escala de grises con la evolución de la superficie.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "grid_dump.hpp"

namespace {

// -----------------------------------------------------------------------------
// Config: reúne los parámetros de la simulación y del video.
// Incluye el tamaño de la malla, la duración, los FPS, la velocidad de onda,
// el amortiguamiento y las características de la gota inicial.
// -----------------------------------------------------------------------------
struct Config {
    int width = 640;
    int height = 640;
    double seconds = 30.0;
    int fps = 30;
    int steps_per_frame = 1;
    float wave_speed = 0.45f;
    float damping = 0.006f;
    float edge_damping = 0.035f;
    float drop_radius = 18.0f;
    float drop_strength = 1.0f;
    std::string output = "output/drop_simulation.mp4";
};

// Convierte las coordenadas (x, y) de la malla en una posición del vector.
// La malla se guarda en un vector 1D, fila por fila.
// -----------------------------------------------------------------------------
int index_of(int x, int y, int width) {
    return y * width + x;
}

// add_drop: crea la perturbación inicial que representa la gota.
// Añade una forma gaussiana en el centro de la malla. También modifica la altura
// previa para que la superficie comience con movimiento.
// -----------------------------------------------------------------------------
void add_drop(std::vector<float>& current, std::vector<float>& previous, const Config& cfg) {
    const float cx = 0.5f * static_cast<float>(cfg.width - 1);
    const float cy = 0.5f * static_cast<float>(cfg.height - 1);
    const float sigma2 = cfg.drop_radius * cfg.drop_radius;

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r2 = dx * dx + dy * dy;
            const float pulse = cfg.drop_strength * std::exp(-r2 / (2.0f * sigma2));
            const int idx = index_of(x, y, cfg.width);
            current[idx] += pulse;
            previous[idx] -= 0.35f * pulse;
        }
    }
}

// border_absorption: calcula cuánto amortiguamiento se aplica en una celda.
// Cerca de los bordes aumenta el amortiguamiento para reducir el rebote de las
// ondas contra el límite de la malla.
// -----------------------------------------------------------------------------
float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

// simulate_step: calcula el siguiente estado de la superficie.
// Para cada celda interior, usa sus cuatro vecinas y los dos estados anteriores
// para actualizar la altura. El amortiguamiento reduce gradualmente el movimiento.
// Los bordes permanecen en cero.
// -----------------------------------------------------------------------------
void simulate_step(const std::vector<float>& previous,
                   const std::vector<float>& current,
                   std::vector<float>& next,
                   const Config& cfg) {
    const float c2 = cfg.wave_speed * cfg.wave_speed;

    std::fill(next.begin(), next.end(), 0.0f);

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const int idx = index_of(x, y, cfg.width);
            const float laplacian =
                current[idx - 1] + current[idx + 1] +
                current[idx - cfg.width] + current[idx + cfg.width] -
                4.0f * current[idx];
            const float velocity = current[idx] - previous[idx];
            const float local_damping = border_absorption(x, y, cfg);

            next[idx] = 2.0f * current[idx] - previous[idx] +
                        c2 * laplacian -
                        local_damping * velocity;
        }
    }
}

// render_frame: convierte las alturas de la malla en una imagen.
// Estima la inclinación de la superficie y aplica iluminación para que las ondas
// se distingan en escala de grises. También añade el número del cuadro al video.
// -----------------------------------------------------------------------------
cv::Mat render_frame(const std::vector<float>& height, const Config& cfg, int frame_number) {
    cv::Mat image(cfg.height, cfg.width, CV_8UC3);

    const cv::Vec3f light_dir = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));

    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cfg.width - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(cfg.height - 1, y + 1);

            const float dx = height[index_of(xm, y, cfg.width)] - height[index_of(xp, y, cfg.width)];
            const float dy = height[index_of(x, ym, cfg.width)] - height[index_of(x, yp, cfg.width)];
            const cv::Vec3f normal = cv::normalize(cv::Vec3f(2.8f * dx, 2.8f * dy, 1.0f));

            const float diffuse = std::max(0.0f, normal.dot(light_dir));
            const float wave = std::clamp(0.5f + 1.8f * height[index_of(x, y, cfg.width)], 0.0f, 1.0f);
            const float specular = std::pow(std::max(0.0f, diffuse), 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;

            const auto gray = static_cast<unsigned char>(std::clamp(intensity, 0.0f, 255.0f));
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
        }
    }

    cv::putText(image,
                "CPU float32 | frame " + std::to_string(frame_number),
                cv::Point(18, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(235, 235, 235),
                1,
                cv::LINE_AA);

    return image;
}

}  // namespace

// main: controla la ejecución del programa.
// Configura la simulación, reserva las tres mallas, crea la gota inicial y abre
// el archivo de video. Después actualiza la superficie y guarda cada cuadro.
// Al terminar, muestra el tiempo, los pasos simulados y el rendimiento.
// -----------------------------------------------------------------------------
int main() {
    try {
        const Config cfg;
        const int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);

        std::filesystem::path output_path(cfg.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        std::vector<float> previous(cells, 0.0f);
        std::vector<float> current(cells, 0.0f);
        std::vector<float> next(cells, 0.0f);

        add_drop(current, previous, cfg);

        cv::VideoWriter writer(
            cfg.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps),
            cv::Size(cfg.width, cfg.height));

        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el archivo de salida: " + cfg.output);
        }

        const auto start = std::chrono::steady_clock::now();

        for (int frame = 0; frame < total_frames; ++frame) {
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                simulate_step(previous, current, next, cfg);
                previous.swap(current);
                current.swap(next);
            }

            // Guarda la malla solo cuando se activa la opción de validación -DDUMP_GRID.
            dump_grid_frame("cpu", frame, total_frames, current.data(), cells);

            writer.write(render_frame(current, cfg, frame));

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << total_frames << '\n';
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        const double simulated_steps = static_cast<double>(total_frames) * cfg.steps_per_frame;

        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo: " << elapsed << " s\n";
        std::cout << "Pasos simulados: " << simulated_steps << '\n';
        std::cout << "Rendimiento: " << simulated_steps / elapsed << " pasos/s\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}