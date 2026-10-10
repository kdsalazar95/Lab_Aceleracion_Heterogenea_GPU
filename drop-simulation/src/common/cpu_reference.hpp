#pragma once

// Copia fiel de las funciones de src/main.cpp (version CPU original).
// Las versiones GPU la usan para dos cosas:
//   1. add_drop(): la perturbacion inicial se calcula una sola vez en CPU.
//   2. Modo --validate: avanzar la simulacion CPU en paralelo con la GPU y
//      medir el error de la malla y de la imagen en cada cuadro.

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "config.hpp"

namespace cpu_ref {

inline int index_of(int x, int y, int width) {
    return y * width + x;
}

inline void add_drop(std::vector<float>& current, std::vector<float>& previous, const Config& cfg) {
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

inline float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

inline void simulate_step(const std::vector<float>& previous,
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

// Igual a render_frame() de la version CPU, pero sin putText: devuelve solo
// la intensidad gris de cada pixel para poder compararla contra la GPU.
inline void render_gray(const std::vector<float>& height, const Config& cfg, std::vector<unsigned char>& gray_out) {
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

            gray_out[index_of(x, y, cfg.width)] =
                static_cast<unsigned char>(std::clamp(intensity, 0.0f, 255.0f));
        }
    }
}

}  // namespace cpu_ref
