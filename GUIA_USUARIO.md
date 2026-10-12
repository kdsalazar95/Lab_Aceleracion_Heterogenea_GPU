# Guía de usuario

## 1. Requerimientos del sistema

| Requisito         | Detalle                                                                |
| ----------------- | ---------------------------------------------------------------------- |
| Sistema operativo | Linux (probado en Ubuntu 24.04)                                        |
| GPU               | NVIDIA con soporte CUDA (probado en RTX 4060 Ti; también Jetson Nano) |
| Compilador        | g++ con soporte de C++17                                               |
| CUDA Toolkit      | 12.x                                                                   |
| Meson y Ninja     | Para compilar el proyecto                                              |
| OpenCV            | 4.x                                                                    |
| FFmpeg            | Opcional, solo para armar el video final                               |
| Nsight Systems    | Opcional, solo para perfilar                                           |

## 2. Preparación

Instalar las dependencias:

```bash
sudo apt update
sudo apt install build-essential git pkg-config meson ninja-build libopencv-dev ffmpeg
```

Instalar el CUDA Toolkit desde la página de NVIDIA: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). En la Jetson Nano ya viene incluido con JetPack.

Verificar que todo quedó instalado:

```bash
g++ --version
meson --version
ninja --version
pkg-config --modversion opencv4
nvcc --version
nvidia-smi
```

Descargar el repositorio:

```bash
git clone https://github.com/kdsalazar95/Lab_Aceleracion_Heterogenea_GPU.git
cd Lab_Aceleracion_Heterogenea_GPU/drop-simulation
```

## 3. Compilación y ejecución

Compilar todas las versiones (CPU y GPU):

```bash
scripts/build_all.sh               # RTX 4060 Ti (por defecto)
GPU_ARCH=53 scripts/build_all.sh   # Jetson Nano
```

Para otra GPU, usar su compute capability sin el punto. Se puede consultar con:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Ejecutar:

```bash
./build/drop_simulation          # CPU
./build/drop_simulation_gpu_v0   # GPU base
./build/drop_simulation_gpu_v1   # Optimización 1: menos transferencias
./build/drop_simulation_gpu_v2   # Optimización 2: memoria compartida
./build/drop_simulation_gpu_v3   # Optimización 3: fast math
```

Cada programa imprime el tiempo total y los pasos por segundo, y guarda su video en `output/`.
