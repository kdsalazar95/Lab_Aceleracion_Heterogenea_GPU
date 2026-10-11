#pragma once

// Interfaz C++ pura hacia la parte CUDA de la version GPU v0.
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

// GPU v0 (linea base): version deliberadamente ingenua.
//   - step(): copia previous y current host->device, lanza el kernel de la
//     ecuacion de onda y copia next device->host. Las mallas viven en el host.
//   - render(): copia la malla host->device, lanza el kernel de render y copia
//     la imagen BGR device->host.
// Todas las copias son sincronas y con memoria paginable. Esta comunicacion
// host-device en cada paso es justamente lo que ataca la Optimizacion 1.
class GpuSimulator {
public:
    explicit GpuSimulator(const SimParams& params);
    ~GpuSimulator();

    GpuSimulator(const GpuSimulator&) = delete;
    GpuSimulator& operator=(const GpuSimulator&) = delete;

    void step(const float* previous, const float* current, float* next);
    void render(const float* height, unsigned char* bgr_out);

    const GpuTimings& timings() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string gpu_device_description();
