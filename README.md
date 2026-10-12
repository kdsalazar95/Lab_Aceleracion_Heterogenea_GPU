# Computación Heterogénea 2026 — Grupo 1

## Laboratorio Semana 9: Aceleracion Heterogenea con GPU usando CUDA


### Profesor

| Rol | Nombre |
|---|---|
| Profesor | Luis Gerardo León Vega |

### Integrantes

| Nombre | Carnet | Rol |
|---|---|---|
| Ronald Duarte Barrantes | 2021004089 | Líder de equipo |
| Khaterine Salazar Martinez | 2014160591 | Desarrollador |
| Fabian Chacon Solano | 2018135154 | Desarrollador |
| Keylor Muñoz Soto. | 2020100689| Revisor |

---

## Descripción

Simulación de la caída de una gota sobre un estanque (ecuación de onda 2D amortiguada, malla de 640 × 640, 900 cuadros) escrita en C++ con OpenCV y portada a CUDA. El repositorio tiene la versión CPU de referencia, la versión GPU base y tres optimizaciones.

| Versión | Carpeta | Qué cambia |
|---|---|---|
| CPU base | `drop-simulation/src/main.cpp` | Referencia en CPU |
| GPU v0 | `src/gpu/v0_baseline/` | Porteo directo de `simulate_step` y `render_frame`, con copias host-device en cada paso |
| GPU v1 | `src/gpu/v1_reduced_transfers/` | Las mallas se quedan en la GPU y solo se copia la imagen por cuadro |
| GPU v2 | `src/gpu/v2_shared_memory/` | `simulate_step` usa memoria compartida y accesos coalesced |
| GPU v3 | `src/gpu/v3_fast_math_precision/` | Kernels de v0 compilados con `--use_fast_math` |

## Requisitos

| Herramienta | Versión probada |
|---|---|
| CUDA / nvcc | 12.9 |
| gcc | 13.4 |
| Meson / Ninja | 1.12.1 / 1.13.2 |
| OpenCV | 4.13 |
| FFmpeg | Solo para `make_final_video.sh` |
| Nsight Systems | 2026.3 (opcional, para perfilar) |

Sin `nvcc`, Meson omite los targets GPU y solo compila la versión CPU.

## Compilación

```bash
cd drop-simulation
scripts/build_all.sh               # RTX 4060 Ti (sm_89)
GPU_ARCH=53 scripts/build_all.sh   # Jetson Nano
```

El script equivale a esto:

```bash
LDFLAGS="-L$CONDA_PREFIX/lib" meson setup build -Dgpu_arch=89
meson compile -C build
```

`LDFLAGS` solo hace falta con un entorno conda, porque `nvcc` no acepta los flags de enlace que conda define por defecto.

## Ejecución

```bash
./build/drop_simulation              # CPU, genera output/drop_simulation.mp4
./build/drop_simulation_gpu_v1       # GPU, genera output/drop_simulation_gpu_v1.mp4
./build/drop_simulation_gpu_v1 --validate   # compara contra CPU y genera validation/v1_errors.csv
```

Cada versión GPU acepta `--validate`, `--output <ruta.mp4>` y `--csv <errores.csv>`. En modo `--validate` los tiempos no son representativos.

## Scripts

Se corren desde `drop-simulation/`.

| Script | Qué hace | Salida |
|---|---|---|
| `scripts/build_all.sh` | Configura y compila todas las versiones | `build/` |
| `scripts/run_benchmarks.sh [N]` | Corre cada versión N veces (3 por defecto) | `results/benchmarks.csv` |
| `scripts/validate.sh` | Valida v0 a v3 contra la CPU | `validation/vN_errors.csv` |
| `scripts/make_final_video.sh` | Arma el video final y uno comparativo con todas las versiones | `output/drop_simulation_final.mp4`, `output/drop_simulation_comparativo.mp4` |

## Estructura

```text
.
├── README.md
├── Reporte.md                      # Reporte final del laboratorio
├── README_Lab9_A_B.md              # Ejercicios A, B y Optimización 1
├── EJ_C - Perfilado y rendimiento  en CPU.md   # Lectura, perfilado CPU y Optimización 2
├── diagrama_flujo.drawio.png
└── drop-simulation/
    ├── meson.build / meson_options.txt
    ├── src/
    │   ├── main.cpp                # CPU base
    │   ├── drop_simulation_comentado_revisado.cpp
    │   ├── common/                 # config.hpp, cpu_reference.hpp
    │   └── gpu/                    # v0_baseline, v1_reduced_transfers, v2_shared_memory, v3_fast_math_precision
    ├── scripts/
    ├── docs/ronald_gpu.md          # GPU v0 y Optimización 3
    ├── profiling/gpu/              # Reportes de Nsight Systems por versión
    ├── validation/                 # Error por cuadro de cada versión vs CPU
    ├── results/                    # benchmarks.csv
    └── output/                     # Videos
```

## Resultados

Equipo principal (i5-12400F + RTX 4060 Ti), 900 pasos, promedio de 3 corridas. El detalle está en [Reporte.md](Reporte.md).

| Versión | Tiempo (s) | Pasos/s | Speedup vs CPU | Error máximo vs CPU |
|---|---:|---:|---:|---:|
| CPU base | 12.963 | 69.43 | 1.00x | — |
| GPU v0 | 2.411 | 373.52 | 5.38x | 6.58e-05 |
| GPU v1 | 1.321 | 681.18 | **9.81x** | 6.58e-05 |
| GPU v2\* | 2.736 | 329.27 | 5.32x (estimado) | 6.58e-05 |
| GPU v3 | 2.264 | 397.67 | 5.73x | 6.58e-05 |

\* La v2 se midió en otro equipo (RTX 4060 Laptop) y su speedup se estimó contra la v0 de ese equipo.
