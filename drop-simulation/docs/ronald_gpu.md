# Ronald — GPU v0 (Ejercicios D/E) y Optimización 3

## Equipo de pruebas

Como no se contó con la Jetson Nano, todas las pruebas de CPU y GPU se hicieron en la misma computadora, para que el speedup compare ambas versiones en igualdad de condiciones.

**Hardware**

| | CPU | GPU |
|---|---|---|
| Modelo | Intel Core i5-12400F | NVIDIA GeForce RTX 4060 Ti |
| Núcleos | 6 núcleos / 12 hilos | 34 SM (Ada, compute capability 8.9) |
| Memoria | 16 GB RAM (WSL2: 15 GB) | 16 GB VRAM, 288 GB/s |
| Caché | L2 7.5 MB, L3 18 MB | L2 32 MB |
| Bus | — | PCIe 4.0 x8 |

**Software**

| Herramienta | Versión |
|---|---|
| SO | Ubuntu 24.04.1 (WSL2) |
| Driver NVIDIA | 617.14 |
| CUDA / nvcc | 12.9.86 |
| gcc | 13.4.0 |
| Meson / Ninja | 1.12.1 / 1.13.2 |
| OpenCV | 4.13.0 |
| Nsight Systems | 2026.3.2 |

## Compilación

```bash
LDFLAGS="-L$CONDA_PREFIX/lib" meson setup build -Dgpu_arch=89
meson compile -C build
```

| Cambio | Motivo |
|---|---|
| Se renombró la opción `cuda_arch` a `gpu_arch` en `meson_options.txt` y `meson.build` | Meson 1.12 reserva `cuda_arch` |
| `-Dgpu_arch=89` | RTX 4060 Ti (la Jetson usa 53) |
| `LDFLAGS` manual | `nvcc` no acepta los flags de enlace por defecto de conda |

## Porteo a CUDA (v0)

Se portaron `simulate_step` y `render_frame` porque se ejecutan en cada cuadro sobre las 409 600 celdas de la malla. `add_drop` se dejó en CPU porque solo corre una vez al inicio.

| Aspecto | Diseño |
|---|---|
| Kernels | `simulate_step_kernel` y `render_kernel`, un hilo por celda/píxel |
| Bloques e hilos | Bloques de 16×16 (256 hilos), grilla de 40×40 bloques |
| Bordes | `simulate_step` pone el borde en 0; `render` usa clamp en los vecinos |
| Memoria GPU | 5 buffers con `cudaMalloc` (previous, current, next, height, imagen) |
| Copias | `cudaMemcpy` síncrono en cada paso y en cada render |
| Video | `putText` y `VideoWriter` siguen en CPU |

## Resultados de tiempo

**CPU base** (`./build/drop_simulation`, 900 pasos)

| Corrida | Tiempo (s) | Pasos/s |
|---|---|---|
| 1 | 12.9067 | 69.73 |
| 2 | 13.0793 | 68.81 |
| 3 | 12.9036 | 69.75 |
| **Promedio** | **12.963** | **69.43** |

**GPU v0** (`./build/drop_simulation_gpu_v0`, 900 pasos)

| Corrida | Tiempo (s) | Pasos/s |
|---|---|---|
| 1 | 2.3731 | 379.26 |
| 2 | 2.3769 | 378.64 |
| 3 | 2.4816 | 362.67 |
| **Promedio** | **2.411** | **373.52** |

**Resumen**

| Versión | Tiempo (s) | Pasos/s | Speedup vs CPU |
|---|---|---|---|
| CPU base | 12.963 | 69.43 | 1.00x |
| GPU v0 | 2.411 | 373.52 | **5.38x** |

## Desglose del tiempo de la v0

Promedio de las 3 corridas (`cudaEvent`, acumulado en 900 cuadros).

| Parte | Tiempo (ms) | % del total |
|---|---|---|
| Kernel `simulate_step` | 37.17 | 1.5 % |
| Kernel `render` | 38.09 | 1.6 % |
| Copias H2D (4218.75 MiB) | 592.41 | 24.6 % |
| Copias D2H (2460.94 MiB) | 428.07 | 17.8 % |
| `putText` + video (CPU) | 1037.14 | 43.0 % |
| Resto | ~277.6 | ~11.5 % |

La mayor parte del tiempo se va en las copias entre CPU y GPU (~42 %) y en escribir el video (~43 %). Los kernels apenas representan un 3 % del total, así que el cálculo en la GPU no es lo que limita a esta versión.

## Validación v0 vs CPU

Se ejecutó `./build/drop_simulation_gpu_v0 --validate`, que guarda el error de cada cuadro en `validation/v0_errors.csv`.

| Métrica | Valor |
|---|---|
| Error abs. máximo malla | 6.58e-05 (cuadro 125) |
| Error abs. medio malla | 1.76e-06 |
| Dif. máxima de píxel | 1 nivel de gris |
| Píxeles distintos | 61 329 de 368.6 M (0.017 %) |

| Cuadro | Error medio | Error máximo | Dif. máx. píxel | Píxeles distintos |
|---|---|---|---|---|
| 0 | 2.1e-12 | 1.2e-07 | 0 | 0 |
| 100 | 2.4e-07 | 5.9e-05 | 1 | 16 |
| 450 | 1.4e-06 | 2.3e-05 | 1 | 72 |
| 899 | 4.5e-06 | 1.9e-05 | 1 | 153 |

Las diferencias se deben al redondeo de float32: la GPU combina multiplicaciones y sumas en una sola operación (FMA) y sus funciones `powf` y `sqrtf` no redondean igual que las de la CPU. El error crece un poco con los cuadros, pero nunca pasa de 1e-4 y en la imagen como mucho cambia un nivel de gris, por lo que la v0 se considera equivalente a la CPU.

## Perfilado nsys v0

```bash
~/micromamba/envs/nsys/bin/nsys profile -f true -o profiling/gpu/v0/v0 --stats=true ./build/drop_simulation_gpu_v0
```

**Kernels (`cuda_gpu_kern_sum`)**

| Kernel | Total (ms) | Llamadas | Promedio (µs) | % |
|---|---|---|---|---|
| `render_kernel` | 8.715 | 900 | 9.68 | 62.2 |
| `simulate_step_kernel` | 5.298 | 900 | 5.89 | 37.8 |

**Copias (`cuda_gpu_mem_time_sum`)**

| Operación | Total (ms) | Copias | Total (MB) | Ancho de banda |
|---|---|---|---|---|
| Host-to-Device | 379.72 | 2700 | 4423.68 | 11.6 GB/s |
| Device-to-Host | 210.07 | 1800 | 2580.48 | 12.3 GB/s |

**API de CUDA (`cuda_api_sum`)**

| Función | Total (ms) | Llamadas | % |
|---|---|---|---|
| `cudaMemcpy` | 937.14 | 4500 | 62.1 |
| `cudaEventSynchronize` | 294.74 | 5400 | 19.5 |
| `cudaMalloc` | 129.77 | 5 | 8.6 |
| `cudaEventRecord` | 101.27 | 10800 | 6.7 |
| `cudaLaunchKernel` | 44.00 | 1800 | 2.9 |

**Conclusiones**

- Las copias suman 589.8 ms en la GPU, mientras que los dos kernels suman solo 14.0 ms. El cuello de botella es la transferencia por PCIe en cada paso, no el cálculo.
- Los tiempos de kernel que imprime el programa con `cudaEvent` (~40 ms) son mayores que los reales de nsys (5 a 9 ms), porque incluyen el costo de lanzar y sincronizar cada llamada. Para el reporte se usan los valores de nsys.
- `render_kernel` es el kernel más costoso, ya que usa `powf`, `sqrtf` y una división por píxel. Por eso es el kernel donde se espera ver el efecto de la Optimización 3.

## Optimización 3 — fast math / precisión

Según nsys, `render_kernel` es el kernel más costoso por usar `powf`, `sqrtf` y divisiones. La hipótesis es que fast math lo acelera a cambio de una pequeña pérdida de precisión.

La v3 usa los mismos kernels que la v0 (`src/gpu/v3_fast_math_precision/`), compilados con `--use_fast_math` en un target nuevo de `meson.build`.

**GPU v3** (`./build/drop_simulation_gpu_v3`, 900 pasos)

| Corrida | Tiempo (s) | Pasos/s |
|---|---|---|
| 1 | 2.2582 | 398.55 |
| 2 | 2.2332 | 403.01 |
| 3 | 2.2992 | 391.44 |
| **Promedio** | **2.264** | **397.67** |

**Desglose v0 vs v3** (promedio, `cudaEvent`)

| Parte | v0 (ms) | v3 (ms) |
|---|---|---|
| Kernel `simulate_step` | 37.17 | 35.94 |
| Kernel `render` | 38.09 | 35.39 |
| Copias H2D | 592.41 | 582.81 |
| Copias D2H | 428.07 | 413.31 |
| `putText` + video (CPU) | 1037.14 | 971.74 |

**Resumen**

| Versión | Tiempo (s) | Pasos/s | Speedup vs CPU | Speedup vs v0 |
|---|---|---|---|---|
| CPU base | 12.963 | 69.43 | 1.00x | — |
| GPU v0 | 2.411 | 373.52 | 5.38x | 1.00x |
| GPU v3 | 2.264 | 397.67 | 5.73x | 1.06x |

La v3 tarda en promedio 0.15 s menos que la v0, pero casi toda esa diferencia está en el video y en las copias, que fast math no modifica. Los kernels solo bajan unos 4 ms en total, por lo que la mejora real de la optimización se mide con los tiempos de kernel de nsys.

### Perfilado nsys v3

```bash
~/micromamba/envs/nsys/bin/nsys profile -f true -o profiling/gpu/v3/v3 --stats=true ./build/drop_simulation_gpu_v3
```

**Kernels v0 vs v3 (`cuda_gpu_kern_sum`)**

| Kernel | v0 promedio (µs) | v3 promedio (µs) | v0 total (ms) | v3 total (ms) | Speedup |
|---|---|---|---|---|---|
| `render_kernel` | 9.68 | 7.76 | 8.715 | 6.980 | **1.25x** |
| `simulate_step_kernel` | 5.89 | 5.62 | 5.298 | 5.059 | 1.05x |
| **Total kernels** | | | **14.013** | **12.039** | **1.16x** |

**Copias v0 vs v3 (`cuda_gpu_mem_time_sum`)**

| Operación | v0 (ms) | v3 (ms) | Total (MB) |
|---|---|---|---|
| Host-to-Device | 379.72 | 362.32 | 4423.68 |
| Device-to-Host | 210.07 | 200.76 | 2580.48 |

**API de CUDA v3 (`cuda_api_sum`)**

| Función | Total (ms) | Llamadas | % |
|---|---|---|---|
| `cudaMemcpy` | 911.11 | 4500 | 63.0 |
| `cudaEventSynchronize` | 283.18 | 5400 | 19.6 |
| `cudaMalloc` | 110.82 | 5 | 7.7 |
| `cudaEventRecord` | 96.92 | 10800 | 6.7 |
| `cudaLaunchKernel` | 44.18 | 1800 | 3.1 |

`render_kernel` baja de 9.68 a 7.76 µs (20 % menos), porque fast math reemplaza `powf`, `sqrtf` y la división por versiones aproximadas del hardware. `simulate_step_kernel` casi no cambia, ya que solo hace sumas y multiplicaciones. Las copias son iguales en volumen y su pequeña diferencia es variación entre corridas.

### Validación v3 vs CPU

Se ejecutó `./build/drop_simulation_gpu_v3 --validate`, que guarda el error de cada cuadro en `validation/v3_errors.csv`.

| Métrica | v0 | v3 |
|---|---|---|
| Error abs. máximo malla | 6.580e-05 | 6.580e-05 |
| Error abs. medio malla | 1.762e-06 | 1.755e-06 |
| Dif. máxima de píxel | 1 nivel de gris | 1 nivel de gris |
| Píxeles distintos | 61 329 (0.017 %) | 61 223 (0.017 %) |

| Cuadro | Error medio | Error máximo | Dif. máx. píxel | Píxeles distintos |
|---|---|---|---|---|
| 0 | 2.1e-12 | 1.2e-07 | 0 | 0 |
| 100 | 2.4e-07 | 5.9e-05 | 1 | 16 |
| 450 | 1.4e-06 | 2.3e-05 | 1 | 81 |
| 899 | 4.4e-06 | 1.7e-05 | 1 | 141 |

La precisión prácticamente no cambia. La malla depende de `simulate_step_kernel`, que no usa funciones afectadas por fast math, y en la imagen las aproximaciones de `__powf` y `rsqrtf` quedan por debajo de un nivel de gris. Fast math no tuvo costo de precisión visible en este programa.

### Conclusión de la Optimización 3

| Métrica | v0 | v3 | Cambio |
|---|---|---|---|
| Tiempo total (s) | 2.411 | 2.264 | 1.06x |
| Tiempo de kernels nsys (ms) | 14.01 | 12.04 | 1.16x |
| `render_kernel` (µs) | 9.68 | 7.76 | 1.25x |
| Error máximo malla | 6.58e-05 | 6.58e-05 | igual |

Fast math acelera el kernel de render un 25 % sin perder precisión, pero el efecto en el tiempo total es pequeño porque los kernels son solo un 3 % de la ejecución. Para que esta optimización se note en el total, primero hay que reducir las copias host-device (Optimización 1).
