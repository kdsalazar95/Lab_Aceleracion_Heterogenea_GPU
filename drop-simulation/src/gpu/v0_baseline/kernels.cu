#include "kernels.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

// Bloques de 16x16 = 256 hilos: un hilo por celda de la malla / pixel.
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Equivalente a border_absorption() de la CPU: amortiguamiento extra en una
// banda de 32 celdas junto a los bordes para que las ondas no reboten.
__device__ float border_absorption(int x, int y, const SimParams p) {
    const int band = 32;
    const int dist = min(min(x, y), min(p.width - 1 - x, p.height - 1 - y));
    if (dist >= band) {
        return p.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return p.damping + p.edge_damping * t * t;
}

// Un paso de la ecuacion de onda 2D amortiguada (diferencias finitas,
// Laplaciano de 5 puntos). Cada hilo calcula una celda de next.
// Las celdas del borde se fuerzan a 0, igual que el std::fill de la CPU.
__global__ void simulate_step_kernel(const float* __restrict__ previous,
                                     const float* __restrict__ current,
                                     float* __restrict__ next,
                                     const SimParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) {
        return;
    }

    const int idx = y * p.width + x;
    if (x == 0 || y == 0 || x == p.width - 1 || y == p.height - 1) {
        next[idx] = 0.0f;
        return;
    }

    const float center = current[idx];
    const float laplacian =
        current[idx - 1] + current[idx + 1] +
        current[idx - p.width] + current[idx + p.width] -
        4.0f * center;
    const float velocity = center - previous[idx];
    const float local_damping = border_absorption(x, y, p);

    next[idx] = 2.0f * center - previous[idx] + p.c2 * laplacian - local_damping * velocity;
}

// Equivalente a render_frame() de la CPU (sin el texto): sombreado difuso +
// especular a partir de la normal de la superficie. Cada hilo escribe un pixel
// BGR con el mismo valor gris en los tres canales.
__global__ void render_kernel(const float* __restrict__ height,
                              uchar3* __restrict__ image,
                              const SimParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) {
        return;
    }

    // Vecinos con clamp en los bordes, igual que std::max/std::min en la CPU.
    const int xm = max(0, x - 1);
    const int xp = min(p.width - 1, x + 1);
    const int ym = max(0, y - 1);
    const int yp = min(p.height - 1, y + 1);

    const float dx = height[y * p.width + xm] - height[y * p.width + xp];
    const float dy = height[ym * p.width + x] - height[yp * p.width + x];

    const float nx = 2.8f * dx;
    const float ny = 2.8f * dy;
    const float nz = 1.0f;
    const float inv_len = 1.0f / sqrtf(nx * nx + ny * ny + nz * nz);

    const float diffuse =
        fmaxf(0.0f, (nx * p.light_x + ny * p.light_y + nz * p.light_z) * inv_len);
    const float wave = fminf(fmaxf(0.5f + 1.8f * height[y * p.width + x], 0.0f), 1.0f);
    const float specular = powf(diffuse, 24.0f);

    float intensity = 35.0f + 120.0f * wave;
    intensity *= 0.60f + 0.65f * diffuse;
    intensity += 130.0f * specular;

    const unsigned char gray = static_cast<unsigned char>(fminf(fmaxf(intensity, 0.0f), 255.0f));
    image[y * p.width + x] = make_uchar3(gray, gray, gray);
}

}  // namespace

struct GpuSimulator::Impl {
    SimParams params{};
    std::size_t grid_bytes = 0;
    std::size_t image_bytes = 0;

    float* d_previous = nullptr;
    float* d_current = nullptr;
    float* d_next = nullptr;
    float* d_height = nullptr;
    uchar3* d_image = nullptr;

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;

    GpuTimings timings;

    dim3 block{kBlockX, kBlockY};
    dim3 grid{1, 1};

    // Mide con cudaEvent el tiempo de una operacion y lo acumula en total_ms.
    template <typename Op>
    void timed(double& total_ms, const char* name, Op op) {
        check_cuda(cudaEventRecord(ev_start), "cudaEventRecord start");
        op();
        check_cuda(cudaGetLastError(), name);
        check_cuda(cudaEventRecord(ev_stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(ev_stop), "cudaEventSynchronize stop");
        float ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&ms, ev_start, ev_stop), "cudaEventElapsedTime");
        total_ms += ms;
    }

    void release() {
        if (ev_start != nullptr) cudaEventDestroy(ev_start);
        if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
        cudaFree(d_previous);
        cudaFree(d_current);
        cudaFree(d_next);
        cudaFree(d_height);
        cudaFree(d_image);
    }
};

GpuSimulator::GpuSimulator(const SimParams& params) : impl_(new Impl) {
    Impl& s = *impl_;
    s.params = params;

    const std::size_t cells = static_cast<std::size_t>(params.width) * static_cast<std::size_t>(params.height);
    s.grid_bytes = cells * sizeof(float);
    s.image_bytes = cells * sizeof(uchar3);
    s.grid = dim3((params.width + kBlockX - 1) / kBlockX, (params.height + kBlockY - 1) / kBlockY);

    try {
        check_cuda(cudaMalloc(&s.d_previous, s.grid_bytes), "cudaMalloc previous");
        check_cuda(cudaMalloc(&s.d_current, s.grid_bytes), "cudaMalloc current");
        check_cuda(cudaMalloc(&s.d_next, s.grid_bytes), "cudaMalloc next");
        check_cuda(cudaMalloc(&s.d_height, s.grid_bytes), "cudaMalloc height");
        check_cuda(cudaMalloc(&s.d_image, s.image_bytes), "cudaMalloc image");
        check_cuda(cudaEventCreate(&s.ev_start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&s.ev_stop), "cudaEventCreate stop");
    } catch (...) {
        s.release();
        throw;
    }
}

GpuSimulator::~GpuSimulator() {
    impl_->release();
}

void GpuSimulator::step(const float* previous, const float* current, float* next) {
    Impl& s = *impl_;

    s.timed(s.timings.h2d_ms, "cudaMemcpy H2D previous/current", [&] {
        check_cuda(cudaMemcpy(s.d_previous, previous, s.grid_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D previous");
        check_cuda(cudaMemcpy(s.d_current, current, s.grid_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D current");
    });
    s.timings.h2d_bytes += 2 * static_cast<long long>(s.grid_bytes);

    s.timed(s.timings.simulate_kernel_ms, "simulate_step_kernel launch", [&] {
        simulate_step_kernel<<<s.grid, s.block>>>(s.d_previous, s.d_current, s.d_next, s.params);
    });

    s.timed(s.timings.d2h_ms, "cudaMemcpy D2H next", [&] {
        check_cuda(cudaMemcpy(next, s.d_next, s.grid_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H next");
    });
    s.timings.d2h_bytes += static_cast<long long>(s.grid_bytes);
}

void GpuSimulator::render(const float* height, unsigned char* bgr_out) {
    Impl& s = *impl_;

    s.timed(s.timings.h2d_ms, "cudaMemcpy H2D height", [&] {
        check_cuda(cudaMemcpy(s.d_height, height, s.grid_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D height");
    });
    s.timings.h2d_bytes += static_cast<long long>(s.grid_bytes);

    s.timed(s.timings.render_kernel_ms, "render_kernel launch", [&] {
        render_kernel<<<s.grid, s.block>>>(s.d_height, s.d_image, s.params);
    });

    s.timed(s.timings.d2h_ms, "cudaMemcpy D2H image", [&] {
        check_cuda(cudaMemcpy(bgr_out, s.d_image, s.image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H image");
    });
    s.timings.d2h_bytes += static_cast<long long>(s.image_bytes);
}

const GpuTimings& GpuSimulator::timings() const {
    return impl_->timings;
}

std::string gpu_device_description() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::ostringstream out;
    out << prop.name << " (sm_" << prop.major << prop.minor << ", " << prop.multiProcessorCount << " SM, "
        << prop.totalGlobalMem / (1024 * 1024) << " MiB)";
    return out.str();
}
