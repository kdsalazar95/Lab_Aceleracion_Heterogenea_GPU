#pragma once

// Interfaz C++ pura hacia la parte CUDA de la version GPU v1.
// main.cpp se compila con g++ (C++17 + OpenCV) y solo ve esta interfaz;
// todo lo que depende de CUDA vive en kernels.cu y se compila con nvcc.

#include <memory>
#include <string>

// Parametros planos (POD) que se pasan por valor a los kernels.
struct SimParams {
    int width;
    int height;
    float c2;            // wave_speed^2
    float damping;
    float edge_damping;
    float light_x;       // Direccion de luz ya normalizada
    float light_y;
    float light_z;
};

// Tiempos acumulados medidos con cudaEvent durante toda la ejecucion.
struct GpuTimings {
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;
    double simulate_kernel_ms = 0.0;
    double render_kernel_ms = 0.0;
    long long h2d_bytes = 0;
    long long d2h_bytes = 0;
};

// GPU v1 (Optimizacion 1: menos comunicacion host-device).
//   - Las mallas previous/current/next viven en la GPU toda la ejecucion.
//   - upload() las sube una vez al inicio; step() solo lanza el kernel y rota
//     punteros en el device.
//   - render() usa la malla del device y solo copia la imagen BGR al host.
class GpuSimulator {
public:
    explicit GpuSimulator(const SimParams& params);
    ~GpuSimulator();

    GpuSimulator(const GpuSimulator&) = delete;
    GpuSimulator& operator=(const GpuSimulator&) = delete;

    void upload(const float* previous, const float* current);
    void step();
    void render(unsigned char* bgr_out);
    void download_current(float* current);

    const GpuTimings& timings() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string gpu_device_description();
