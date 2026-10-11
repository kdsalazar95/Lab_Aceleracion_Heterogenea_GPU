# Reporte de Optimizaciones — Aceleración Heterogénea con GPU

**Instituto Tecnológico de Costa Rica**
**Curso:** EL5859 Computación Heterogénea — II Semestre 2026
**Profesor:** Luis G. León-Vega, Ph.D.
**Evaluación:** Laboratorio Semana 9 — Ejercicio G

---

## Índice

1. [Entorno de pruebas](#1-entorno-de-pruebas)
2. [Descripción del programa base](#2-descripción-del-programa-base)
3. [Diagrama de flujo](#3-diagrama-de-flujo)
4. [Explicación de cada función del programa](#4-explicación-de-cada-función-del-programa)
5. [Ideas de la lectura aplicadas al laboratorio](#5-ideas-de-la-lectura-aplicadas-al-laboratorio)
6. [Resultados del perfilado CPU](#6-resultados-del-perfilado-cpu)
7. [Justificación de las funciones portadas a GPU](#7-justificación-de-las-funciones-portadas-a-gpu)
8. [Diseño de los kernels CUDA](#8-diseño-de-los-kernels-cuda)
9. [Estrategia de validación CPU vs GPU](#9-estrategia-de-validación-cpu-vs-gpu)
10. [Resultados de cada optimización](#10-resultados-de-cada-optimización)
11. [Tabla resumen](#11-tabla-resumen)
12. [Análisis de rendimiento](#12-análisis-de-rendimiento)
13. [Conclusiones](#13-conclusiones)
14. [Instrucciones de compilación y ejecución](#14-instrucciones-de-compilación-y-ejecución)
15. [Anexos](#15-anexos)
16. [Referencias](#16-referencias)

---

## 1. Entorno de pruebas

No se contó con la NVIDIA Jetson Nano. Para que los speedups comparen versiones en igualdad de condiciones, la CPU base, la GPU v0, la Optimización 1 y la Optimización 3 se midieron en el mismo equipo (**equipo principal**). La Optimización 2 se midió en otro equipo (**equipo secundario**), y por eso se compara contra una v0 medida en ese mismo equipo. Los Ejercicios A y C también se corrieron en equipos distintos; sus tiempos se reportan por separado y no se usan para calcular speedups.

**Equipo principal (CPU base, GPU v0, v1 y v3)**

| | CPU | GPU |
|---|---|---|
| Modelo | Intel Core i5-12400F | NVIDIA GeForce RTX 4060 Ti |
| Núcleos / SM | 6 núcleos / 12 hilos | 34 SM (Ada, compute capability 8.9) |
| Memoria | 16 GB RAM (WSL2: 15 GB) | 16 GB VRAM, 288 GB/s |
| Caché | L2 7.5 MB, L3 18 MB | L2 32 MB |
| Bus | — | PCIe 4.0 x8 |

| Herramienta | Versión |
|---|---|
| Sistema operativo | Ubuntu 24.04.1 (WSL2) |
| Compilador (gcc) | 13.4.0 |
| CUDA / nvcc | 12.9.86 |
| Driver NVIDIA | 617.14 |
| Meson / Ninja | 1.12.1 / 1.13.2 |
| OpenCV | 4.13.0 |
| Nsight Systems | 2026.3.2 |

**Otros equipos usados**

| Uso | Equipo |
|---|---|
| Ejercicio A (CPU base inicial) | Intel Core i3-10100F, 15 GiB RAM, Pop!_OS 24.04, g++ 13.3.0, Meson 1.3.2, Ninja 1.11.1, OpenCV 4.6.0 |
| Optimización 2 | NVIDIA GeForce RTX 4060 Laptop GPU |

**Limitaciones del entorno**

- Sin Jetson Nano: el código se compila con `-Dgpu_arch=89` para la RTX 4060 Ti (la Jetson usaría `53`).
- En Meson 1.12 el nombre `cuda_arch` está reservado, así que la opción se renombró a `gpu_arch` en `meson_options.txt` y `meson.build`.
- Se usó Nsight Systems (`nsys`). No se usaron Nsight Compute ni nvprof.

Este detalle importa porque, como explica León-Vega (2026), cuando la tarjeta gráfica está separada del procesador, cada uno tiene su propia memoria y la información tiene que pasar de una a otra por una conexión mucho más lenta que la memoria interna de la tarjeta; la diferencia puede ser de más de diez veces (pp. 137, 142). La tarjeta que se usó, además, tiene esa conexión reducida a la mitad de su ancho (8 carriles en lugar de 16), así que mover datos le cuesta todavía más. En la Jetson Nano la situación cambia: el procesador y la parte gráfica comparten la misma memoria, lo que abarata el intercambio, aunque el autor aclara que no lo vuelve gratuito (León-Vega, 2026, pp. 142–143).

---

## 2. Descripción del programa base

El programa simula la caída de una gota sobre un estanque y guarda el resultado en un video en escala de grises. La superficie del agua se representa con una malla de 640 × 640 alturas: cada celda es una posición de la superficie y un píxel del video.

Al inicio, `add_drop` crea una perturbación en el centro de la malla. En cada paso, la altura de cada celda se calcula a partir de sus cuatro vecinas y de las alturas de los dos instantes anteriores, de modo que la perturbación se propaga como ondas. Cerca de los bordes el amortiguamiento aumenta para reducir los rebotes. Después, `render_frame` convierte las alturas en una imagen sombreada y el programa la escribe en el video.

**Modelo físico:** ecuación de onda bidimensional amortiguada

```text
∂²h/∂t² = c² · ∇²h − γ · ∂h/∂t
```

resuelta con diferencias finitas explícitas y un stencil de 5 puntos:

```text
siguiente = 2 · actual − anterior
            + c² · (izquierda + derecha + arriba + abajo − 4 · actual)
            − amortiguamiento_local · (actual − anterior)
```

La velocidad de onda (`0.45` celdas/paso) está por debajo del límite de estabilidad aproximado de `0.707` para este esquema en 2D.

**Parámetros de la simulación**

| Parámetro | Valor |
|---|---|
| Resolución | 640 × 640 |
| Duración (s) | 30 |
| FPS | 30 |
| Pasos por cuadro | 1 |
| Pasos simulados totales | 900 |
| `wave_speed` | 0.45 |
| `damping` | 0.006 |
| `edge_damping` | 0.035 |
| `drop_radius` / `drop_strength` | 18 / 1.0 |

**Resultados de ejecución CPU base**

| Métrica | Ejercicio A (i3-10100F) | Equipo principal (i5-12400F) |
|---|---:|---:|
| Tiempo total (s) | 15.74 ± 0.17 | 12.963 |
| Pasos simulados | 900 | 900 |
| Pasos/s | 57.2 ± 0.6 | 69.43 |
| Número de corridas | 5 (mín. 15.62 s, máx. 16.01 s) | 3 (12.90 – 13.08 s) |

La resolución, la duración y la tasa de cuadros se comprobaron con `ffprobe` sobre `output/drop_simulation.mp4`. Para calcular speedups se usa la CPU base del equipo principal (12.963 s).

Este tipo de programa es un buen candidato para la GPU. León-Vega (2026) señala que la tarjeta gráfica rinde mejor cuando un problema se puede dividir en muchas tareas pequeñas que hacen lo mismo con datos distintos, y menciona las simulaciones numéricas como un ejemplo típico (p. 130). Aquí, cada una de las 409 600 celdas se calcula con la misma fórmula. El autor también advierte que no basta con que el trabajo se pueda repartir: tiene que haber suficiente para mantener ocupada a toda la tarjeta (León-Vega, 2026, pp. 131–133). En este caso, cada paso de la simulación genera 1600 grupos de 256 tareas, que es trabajo de sobra.

---

## 3. Diagrama de flujo

![Diagrama de flujo](diagrama_flujo.drawio.png)

El programa inicializa los parámetros, reserva las tres mallas (`previous`, `current` y `next`), aplica la perturbación inicial y abre el archivo de video. Luego recorre los 900 cuadros. En cada uno actualiza la malla, intercambia los buffers con `swap` (sin copiar su contenido), renderiza el cuadro y lo escribe en el video. Al terminar, detiene el cronómetro e imprime el tiempo, los pasos simulados y los pasos por segundo.

Reservar las mallas una sola vez y después solo intercambiar sus papeles coincide con un consejo del capítulo: es mejor apartar la memoria al principio y reutilizarla en cada repetición que pedirla y liberarla una y otra vez (León-Vega, 2026, p. 138).

---

## 4. Explicación de cada función del programa

| Función / estructura | Propósito | Papel en la simulación |
|---|---|---|
| `Config` | Agrupa los parámetros de la malla, el video, la velocidad de onda, el amortiguamiento y la gota. | Fuente única de configuración para todo el programa. |
| `index_of` | Convierte las coordenadas `(x, y)` en el índice del vector que almacena la malla. | Acceso lineal a la malla, fila por fila. |
| `add_drop` | Crea la perturbación inicial en el centro de la superficie. | Condición inicial; se ejecuta una sola vez. |
| `border_absorption` | Aumenta el amortiguamiento cerca de los bordes. | Evita que las ondas reboten en los límites de la malla. |
| `simulate_step` | Calcula las nuevas alturas con la ecuación de onda amortiguada. | Núcleo numérico; recorre las 409 600 celdas en cada paso. |
| `render_frame` | Calcula normales, iluminación (`powf`, `sqrt`) y escribe el número de cuadro. | Convierte la malla en imagen; recorre todos los píxeles en cada cuadro. |
| `main` | Inicializa los datos, ejecuta el ciclo de simulación, genera el video y calcula las métricas. | Orquesta el flujo completo y mide el rendimiento. |

Las dos funciones más pesadas no se comportan igual. León-Vega (2026) propone distinguir entre las tareas que se ven frenadas por la cantidad de cálculos y las que se ven frenadas por la velocidad con la que llegan los datos, y sugiere comparar cuántas operaciones se hacen por cada dato que se lee (p. 131). Con ese criterio, `simulate_step` hace muy pocas cuentas por cada dato, así que depende sobre todo de la memoria. `render_frame`, en cambio, usa operaciones más costosas, como potencias y raíces, y depende más del cálculo.

---

## 5. Ideas de la lectura aplicadas al laboratorio

### 5.1. Muchas tareas iguales e independientes

En cada paso se repite el mismo cálculo, por separado, para cada una de las 409 600 celdas de la malla. León-Vega (2026) explica que la GPU aprovecha justamente este tipo de trabajo: cada tarea pequeña se le asigna a un hilo distinto, y hace falta que haya muchas para que la tarjeta no se quede sin nada que hacer (pp. 130, 133). Por eso se llevaron a la GPU `simulate_step` y `render_frame`, que recorren toda la malla en cada cuadro, mientras que `add_drop` se quedó en el procesador, ya que solo se usa una vez al inicio.

### 5.2. La forma de leer los datos

Según León-Vega (2026), la velocidad de la GPU no depende solo de cuántas cuentas hace, sino también de cómo lee los datos: rinde mejor cuando tareas vecinas leen posiciones vecinas de la memoria (pp. 134–135). Con esa idea se hizo la Optimización 2, que copia cada vecindario de la malla a una memoria más rápida, compartida por cada grupo de hilos, antes de calcular. El resultado no fue el esperado, y el mismo autor ayuda a entender por qué: esa memoria compartida solo compensa cuando los mismos datos se usan muchas veces (León-Vega, 2026, p. 141). En esta simulación cada dato se reutiliza poco, así que copiarlo y esperar a que todo el grupo terminara de copiar probablemente costó más de lo que ahorró.

### 5.3. El costo de mover datos entre el procesador y la tarjeta

León-Vega (2026) advierte que pasar información entre el procesador y la tarjeta gráfica puede borrar la ganancia de calcular en paralelo, por lo que recomienda dejar los datos dentro de la tarjeta todo el tiempo posible (pp. 137–138, 142). Las mediciones de la versión base lo confirman: en la v0, los cálculos en la GPU ocuparon cerca del 3 % del tiempo, mientras que el ir y venir de datos ocupó cerca del 42 %. La Optimización 1 deja las mallas en la tarjeta durante toda la simulación, redujo en un 84 % la información transferida y fue la mejora más grande del laboratorio, sin cambiar los cálculos.

### 5.4. Precisión numérica contra velocidad

La opción `--use_fast_math` cambia algunas funciones matemáticas, como las potencias, las raíces y las divisiones, por versiones más rápidas pero un poco menos exactas. León-Vega (2026) lo plantea como un intercambio: la versión rápida solo debe usarse si el error que introduce es aceptable para el problema (pp. 160–161). En esta simulación, los cálculos de la malla no usan esas funciones, y en la imagen la diferencia queda por debajo de un tono de gris. Por eso el dibujo de cada cuadro se aceleró 1.25 veces sin que se notara ninguna pérdida de precisión.

---

## 6. Resultados del perfilado CPU

**Herramientas usadas:** perf y Valgrind Callgrind.

**Comandos ejecutados**

```bash
perf record -g ./build/drop_simulation
perf report

valgrind --tool=callgrind ./build/drop_simulation
callgrind_annotate callgrind.out.<pid>
```

**Métricas de ejecución**

| Métrica | Resultado |
|---|---:|
| Tiempo CPU sin instrumentación | 11.4182 s |
| Tiempo con Callgrind | 330.224 s |
| Referencias a instrucciones (Ir) | 122 782 185 856 |

**Distribución por función**

| Función | % ciclos (perf) | % Ir (Callgrind) | Observaciones |
|---|---:|---:|---|
| `main` (código propio, incluye `simulate_step` y `render_frame` integradas) | — | 43.41 % | El compilador integró las funciones dentro de `main` |
| `__powf_fma` | 37.34 % | 19.19 % | Iluminación dentro de `render_frame` |
| `cv::VideoWriter::write` | — | 13.68 % | Codificación del video |
| `sws_scale` | — | 8.41 % | Conversión de formato de FFmpeg |
| `main` (inclusivo) | — | 99.51 % | No se suma con las demás filas |

**Conclusiones del perfilado**

- **Funciones más críticas:** `render_frame` (con `__powf_fma` como mayor consumidor individual) y `simulate_step`. Ambas recorren la malla completa en cada cuadro.
- **Función que conviene portar primero:** `render_frame`, por el peso de `powf`. Le sigue `simulate_step`, que además debe estar en GPU para no copiar la malla en cada paso.
- **Funciones que no conviene portar:** `add_drop`, porque se ejecuta una sola vez, y `cv::VideoWriter::write`, porque es codificación de video y no forma parte de la simulación. Esta última es la parte que no se acelera.
- **Limitaciones:** por la integración de funciones (inlining), no se obtuvieron porcentajes individuales confiables para `render_frame` y `simulate_step`. El 37.34 % corresponde solo a `__powf_fma`. El tiempo de Callgrind no sirve para calcular speedups.

Este orden de trabajo sigue lo que plantea León-Vega (2026): antes de optimizar cualquier detalle, hay que comprobar que el problema realmente se presta para la GPU (p. 129). El autor propone hacerse tres preguntas: cuánto trabajo se puede hacer al mismo tiempo, cuánto le toca a cada tarea y cuántos datos hay que mover para hacerlo (León-Vega, 2026, p. 131). El perfilado ayudó a responderlas y mostró que el cálculo de la iluminación y la actualización de la malla eran las partes que más convenía llevar a la tarjeta.

Detalle completo: [EJ_C - Perfilado y rendimiento en CPU.md](EJ_C%20-%20Perfilado%20y%20rendimiento%20%20en%20CPU.md)

---

## 7. Justificación de las funciones portadas a GPU

| Función portada | Evidencia del perfilado | Por qué es paralelizable | Dependencias / riesgos |
|---|---|---|---|
| `render_frame` → `render_kernel` | `__powf_fma` concentra el 37.34 % de los ciclos (perf) | Cada píxel se calcula de forma independiente a partir de sus vecinos | `powf` y `sqrtf` en GPU no redondean igual que en CPU; `putText` sigue en CPU |
| `simulate_step` → `simulate_step_kernel` | Es parte del 43.41 % de Ir propio de `main`; recorre las 409 600 celdas por paso | Stencil de 5 puntos; cada celda solo lee el estado anterior | Necesita rotar tres buffers; las FMA cambian el redondeo |

**Funciones que se mantuvieron en CPU:** `add_drop` (se ejecuta una vez), `putText` y `cv::VideoWriter::write` (dependen de OpenCV y FFmpeg en el host).

Se llevaron a la GPU las funciones que repiten la misma operación sobre todos los píxeles, que es el tipo de trabajo que León-Vega (2026) describe como ideal para estas tarjetas (p. 130). `add_drop` se quedó en el procesador porque se ejecuta una sola vez y, como explica el autor, en tareas tan pequeñas el costo de preparar el trabajo y mover los datos puede ser mayor que la ganancia (León-Vega, 2026, pp. 131–132).

---

## 8. Diseño de los kernels CUDA

### 8.1. Gestión de memoria

| Aspecto | GPU v0 | GPU v1 |
|---|---|---|
| Buffers (`cudaMalloc`) | 5: `previous`, `current`, `next`, `height`, imagen | 4: `render` lee directamente `d_current` |
| H2D por cuadro | `previous`, `current` y `height` (3 × 1.56 MiB) | Ninguna; solo la copia inicial |
| D2H por cuadro | `next` e imagen (1.56 + 1.17 MiB) | Solo la imagen (1.17 MiB) |
| Rotación de mallas | `swap` de vectores en el host | `std::swap` de punteros en el device |
| Copias | `cudaMemcpy` síncrono | `cudaMemcpy` síncrono |

La v1 aplica lo que León-Vega (2026) describe como mantener los datos dentro de la tarjeta: se suben una vez, se trabajan muchas veces ahí y solo se baja el resultado que de verdad se necesita (pp. 137–138).

### 8.2. Kernels

| Kernel | Entrada | Salida | Configuración (grid / block) | Manejo de bordes |
|---|---|---|---|---|
| `simulate_step_kernel` | `previous`, `current` | `next` | Grid 40 × 40, block 16 × 16 (v2: block 32 × 8 con tile `[8+2][32+2]` en shared) | Las celdas del borde se fijan en 0 |
| `render_kernel` | `height` / `current` | Imagen BGR 640 × 640 | Grid 40 × 40, block 16 × 16 | Los vecinos se acotan (clamp) a la malla |

Cada grupo de 256 hilos está muy por debajo del máximo de 1024 que permite la tarjeta (León-Vega, 2026, p. 122), y en total se forman 1600 grupos, más que suficientes para mantenerla ocupada (León-Vega, 2026, pp. 132–133). Además, hilos vecinos leen celdas vecinas, lo que, según el mismo autor, permite traer los datos en menos viajes a la memoria (León-Vega, 2026, pp. 134–135).

### 8.3. Integración con el flujo de video

Después de `render_kernel`, la imagen se copia al host con `cudaMemcpy` D2H en un `cv::Mat`. Ahí `putText` agrega el número de cuadro y `cv::VideoWriter` lo escribe en `output/drop_simulation.mp4`. El ciclo y las métricas de `main` se mantienen igual que en la versión CPU. León-Vega (2026) insiste en mirar el recorrido completo de los datos y no cada cálculo por separado (p. 138); siguiendo esa idea, lo único que vuelve al procesador es la imagen que hace falta para escribir el video.

---

## 9. Estrategia de validación CPU vs GPU

**Método:** se siguió el camino que propone León-Vega (2026, p. 165): partir de una versión de referencia, definir cuánto error se acepta, medir el error de cada cambio y medir cuánto mejora la velocidad. Cada versión GPU se ejecuta con `--validate`. En este modo, cada cuadro se compara con la referencia CPU ([cpu_reference.hpp](drop-simulation/src/common/cpu_reference.hpp)) en dos niveles: la malla de alturas (error absoluto) y la imagen renderizada (diferencia por píxel). Los tiempos del modo de validación no se usan para medir rendimiento.

**Métricas:** error absoluto medio y máximo de la malla, diferencia máxima de píxel y píxeles distintos acumulados en los 900 cuadros.

**Tolerancia aceptada:** error absoluto de la malla < 1e-4 y diferencia de píxel ≤ 1 nivel de gris, que es lo esperable con redondeo de `float32`.

| Versión | Error medio | Error máximo | Dif. máx. píxel | Píxeles distintos | ¿Dentro de tolerancia? | Archivo |
|---|---|---|---|---|---|---|
| GPU v0 | 1.762e-06 | 6.580e-05 (cuadro 125) | 1 | 61 329 (0.017 %) | Sí | [v0_errors.csv](drop-simulation/validation/v0_errors.csv) |
| GPU v1 | 1.762e-06 | 6.580e-05 | 1 | 61 329 (0.017 %) | Sí | [v1_errors.csv](drop-simulation/validation/v1_errors.csv) |
| GPU v2 | 1.762e-06 | 6.580e-05 | 1 | 61 329 | Sí | — |
| GPU v3 | 1.755e-06 | 6.580e-05 | 1 | 61 223 (0.017 %) | Sí | [v3_errors.csv](drop-simulation/validation/v3_errors.csv) |

**Evolución del error por cuadro (v0)**

| Cuadro | Error medio | Error máximo | Dif. máx. píxel | Píxeles distintos |
|---|---|---|---|---|
| 0 | 2.1e-12 | 1.2e-07 | 0 | 0 |
| 100 | 2.4e-07 | 5.9e-05 | 1 | 16 |
| 450 | 1.4e-06 | 2.3e-05 | 1 | 72 |
| 899 | 4.5e-06 | 1.9e-05 | 1 | 153 |

**Explicación de las diferencias:** se deben al redondeo de `float32`. La GPU junta una multiplicación y una suma en un solo paso (FMA), y León-Vega (2026) explica que eso puede cambiar levemente el resultado frente a hacerlas por separado (pp. 158–159). Además, sus funciones `powf` y `sqrtf` no redondean igual que las de la CPU. El error crece un poco con los cuadros, pero nunca pasa de 1e-4, y en la imagen cambia como mucho un nivel de gris. `v1_errors.csv` es idéntico byte a byte a `v0_errors.csv`, como se esperaba, porque los kernels son los mismos. En v3 se repitió la validación por el cambio de representación numérica (fast math).

---

## 10. Resultados de cada optimización

> **Nota:** las tres optimizaciones se desarrollaron en paralelo y **cada una parte de la GPU v0**, no de la versión anterior. Por eso, en las tablas, la columna "speedup vs GPU anterior" se calcula contra v0.

### 10.0. GPU base (v0)

| Métrica | Valor |
|---|---|
| Tiempo total (s) | 2.411 (3 corridas: 2.373 / 2.377 / 2.482) |
| Tiempo de kernels nsys (ms) | 14.013 (`render` 8.715 + `simulate_step` 5.298) |
| Tiempo H→D nsys (ms) | 379.72 (2700 copias, 4423.68 MB) |
| Tiempo D→H nsys (ms) | 210.07 (1800 copias, 2580.48 MB) |
| Pasos/s | 373.52 |
| Speedup vs CPU | 5.38x |
| Error máximo vs CPU | 6.58e-05 |

**Comando de perfilado**

```bash
nsys profile -f true -o profiling/gpu/v0/v0 --stats=true ./build/drop_simulation_gpu_v0
```

**Desglose del tiempo (`cudaEvent`, acumulado en 900 cuadros)**

| Parte | Tiempo (ms) | % del total |
|---|---:|---:|
| Kernel `simulate_step` | 37.17 | 1.5 % |
| Kernel `render` | 38.09 | 1.6 % |
| Copias H2D | 592.41 | 24.6 % |
| Copias D2H | 428.07 | 17.8 % |
| `putText` + video (CPU) | 1037.14 | 43.0 % |
| Resto | ~277.6 | ~11.5 % |

**API de CUDA (`cuda_api_sum`)**

| Función | Total (ms) | Llamadas | % |
|---|---:|---:|---:|
| `cudaMemcpy` | 937.14 | 4500 | 62.1 |
| `cudaEventSynchronize` | 294.74 | 5400 | 19.5 |
| `cudaMalloc` | 129.77 | 5 | 8.6 |
| `cudaEventRecord` | 101.27 | 10800 | 6.7 |
| `cudaLaunchKernel` | 44.00 | 1800 | 2.9 |

**Cuello de botella identificado:** las copias suman 589.8 ms en la GPU, mientras que los kernels suman solo 14.0 ms. En esta versión el tiempo lo limitan la transferencia por PCIe en cada paso y la escritura del video, no el cálculo. Los tiempos de kernel medidos con `cudaEvent` (~37 ms) son mayores que los de nsys porque incluyen el lanzamiento y la sincronización. Para el análisis se usan los valores de nsys.

León-Vega (2026) descompone el tiempo de la GPU en cuatro partes: preparar el trabajo, mover los datos, calcular y esperar a que todo termine (pp. 131–132). En la v0, mover datos pesó muchísimo más que calcular, tal como anticipa el autor cuando señala que la conexión con el procesador es mucho más lenta que la memoria propia de la tarjeta (León-Vega, 2026, p. 137).

---

### 10.1. Optimización 1 — Reducción de transferencias host-device (v1)

**Qué problema intenta resolver:** en v0, las copias por PCIe ocupaban alrededor del 42 % del tiempo total (4500 `cudaMemcpy`, 7 GB transferidos), mientras que los kernels ocupaban alrededor del 3 %.

**Qué cambio se realizó en el código:** las mallas `previous`, `current` y `next` se quedan en la GPU durante toda la ejecución y solo se copian una vez al inicio. La rotación se hace con `std::swap` de punteros en el device. `render_kernel` lee directamente `d_current`, así que se elimina el buffer `height`. Por cuadro solo baja la imagen. Con esto se siguen dos consejos de León-Vega (2026): hacer el menor número posible de viajes de datos entre el procesador y la tarjeta (p. 142) y mantener los datos dentro de la tarjeta mientras se trabaja con ellos (pp. 137–138).

**Evidencia de perfilado que justifica el cambio:** según `cuda_gpu_mem_time_sum` de v0, las copias suman 589.79 ms frente a 14.01 ms de kernels, y `cudaMemcpy` es el 62.1 % del tiempo de la API de CUDA.

**Resultados:**

| Métrica | GPU v0 | GPU v1 | Cambio |
|---|---:|---:|---:|
| Tiempo total (s) | 2.411 | 1.321 | −45.2 % |
| Tiempo de kernels nsys (ms) | 14.013 | 13.199 | 1.06x |
| Tiempo H→D nsys (ms) | 379.72 | 0.25 | −99.9 % |
| Tiempo D→H nsys (ms) | 210.07 | 84.31 | −59.9 % |
| Copias realizadas | 4500 | 902 | −80 % |
| Volumen transferido (MB) | 7004.16 | 1109.20 | −84.2 % |
| Pasos/s | 373.52 | 681.18 | +82 % |
| Speedup vs CPU | 5.38x | 9.81x | |
| Speedup vs GPU anterior (v0) | 1.00x | 1.82x | |
| Error máximo vs CPU | 6.58e-05 | 6.58e-05 | igual |

**Desglose v0 vs v1 (`cudaEvent`)**

| Parte | v0 (ms) | v1 (ms) | v1 % del total |
|---|---:|---:|---:|
| Kernel `simulate_step` | 37.17 | 50.86 | 3.8 % |
| Kernel `render` | 38.09 | 14.43 | 1.1 % |
| Copias H2D | 592.41 | 0.61 | 0.05 % |
| Copias D2H | 428.07 | 199.29 | 15.1 % |
| `putText` + video (CPU) | 1037.14 | 947.41 | 71.7 % |
| Resto | ~277.6 | ~108.8 | ~8.2 % |

**Cuánto mejoró el rendimiento:** la v1 es 1.82 veces más rápida que la v0 y alcanza 9.81x respecto a la CPU. El tiempo de copias en la GPU baja de 589.8 ms a 84.6 ms, unas 7 veces menos. `cudaMemcpy` todavía tarda 212.7 ms en el host, aunque la copia en la GPU dura 84.3 ms, porque se usa memoria paginable. Con memoria pinned (`cudaMallocHost`) ese tiempo podría bajar: León-Vega (2026) explica que, al reservar la memoria de esa forma, el sistema operativo no la cambia de lugar y la tarjeta puede copiarla de manera más directa (p. 145). El nuevo cuello de botella es `putText` y la escritura del video en CPU (~72 %).

**Impacto sobre la precisión numérica:** ninguno. `v1_errors.csv` es idéntico a `v0_errors.csv`, porque los kernels no cambiaron.

---

### 10.2. Optimización 2 — Memoria compartida (v2)

**Qué problema intenta resolver:** cada hilo de `simulate_step_kernel` lee cinco valores de memoria global (el centro y sus cuatro vecinos), y los vecinos se repiten entre hilos adyacentes. La idea es cargar el vecindario una sola vez por bloque.

**Qué cambio se realizó en el código:** cada bloque carga un tile `__shared__ float tile[kBlockY + 2][kBlockX + 2]` con halos y sincroniza con `__syncthreads()` antes de calcular el stencil. Se probaron bloques de 16 × 16 y 32 × 8 (256 hilos en ambos casos). Con 32 × 8, los 32 hilos de un warp recorren una fila consecutiva, lo que favorece los accesos coalesced. En el código quedó 32 × 8.

**Evidencia de perfilado que justifica el cambio:** el stencil hace 5 lecturas globales por celda, y León-Vega (2026) sugiere usar la memoria compartida cuando varios hilos de un mismo grupo necesitan los mismos datos (p. 140).

**Resultados** (equipo secundario: RTX 4060 Laptop GPU, promedio de 3 corridas sin `--validate`):

| Métrica | GPU v0 (mismo equipo) | v2 16 × 16 | v2 32 × 8 | Cambio 32 × 8 vs v0 |
|---|---:|---:|---:|---:|
| Tiempo total (s) | 2.709 | 2.755 | 2.736 | +1.0 % |
| Kernel `simulate_step` (ms, `cudaEvent`) | 26.35 | 32.54 | 31.02 | +17.7 % |
| Tiempo H→D / D→H | — | — | — | igual que v0 (mismas copias) |
| Pasos/s | 332.26 | 326.97 | 329.27 | −0.9 % |
| Speedup vs CPU (estimado\*) | 5.38x | 5.29x | 5.32x | |
| Speedup vs GPU anterior (v0) | 1.00x | 0.98x | 0.99x | kernel: 0.85x |
| Error máximo vs CPU | 6.58e-05 | 6.58e-05 | 6.58e-05 | igual |

\* La CPU base no se midió en el equipo secundario. Para estimarla se supone que la v0 logra el mismo speedup sobre CPU en ambos equipos (12.963 s / 2.411 s = 5.377x). Con esa suposición, la CPU base estimada en el equipo secundario es 2.709 s × 5.377 ≈ 14.57 s, y el speedup de cada configuración es:

```text
Speedup_v2 = (T_CPU / T_v0)_principal × (T_v0 / T_v2)_secundario
16 × 16:  5.377 × (2.709 / 2.755) = 5.377 × 0.983 = 5.29x
32 × 8:   5.377 × (2.709 / 2.736) = 5.377 × 0.990 = 5.32x
```

La suposición es razonable porque la v2 conserva el mismo flujo de copias y de video que la v0, que dominan el tiempo total. Aun así, es una estimación: para un valor medido hay que correr la CPU base en el equipo secundario.

**Cuánto mejoró o empeoró el rendimiento:** empeoró. El kernel de 32 × 8 es 17.7 % más lento que v0 (speedup 0.85x), aunque es 4.7 % más rápido que la configuración 16 × 16. El tiempo total casi no cambia, porque v2 conserva las copias de v0 por cada paso y estas dominan la ejecución. Una explicación probable es que la caché L1/L2 de Ada ya reutilizaba los vecinos y que la carga de halos más `__syncthreads()` agregan más trabajo del que ahorran. Para confirmarlo haría falta Nsight Compute.

Este resultado coincide con dos advertencias de León-Vega (2026). La primera es que copiar datos a la memoria compartida solo compensa si se van a usar muchas veces (p. 141). La segunda es que hacer que un grupo de hilos se espere entre sí tiene un costo, porque los más rápidos quedan detenidos hasta que terminan los más lentos (pp. 147–148, 151). En esta simulación cada dato se reutiliza poco, así que, probablemente, la copia extra y la espera costaron más de lo que se ahorró.

**Impacto sobre la precisión numérica:** ninguno. Ambas configuraciones dan el mismo error que v0.

---

### 10.3. Optimización 3 — Fast math (v3)

**Qué problema intenta resolver:** según nsys, `render_kernel` es el kernel más costoso (62.2 % del tiempo de kernels) porque usa `powf`, `sqrtf` y una división por píxel.

**Qué cambio se realizó en el código:** los kernels son los mismos que en v0, compilados con `--use_fast_math` en un target nuevo de `meson.build`. León-Vega (2026) explica que esta opción cambia las funciones matemáticas exactas por versiones más rápidas y un poco menos precisas, que aprovechan partes de la tarjeta pensadas para ese tipo de cálculos (pp. 159–161):

```meson
cuda_args: gpu_cuda_args + ['--use_fast_math'],
```

**Evidencia de perfilado que justifica el cambio:** en `cuda_gpu_kern_sum` de v0, `render_kernel` suma 8.715 ms (9.68 µs por llamada) frente a 5.298 ms de `simulate_step_kernel`.

**Resultados:**

| Métrica | GPU v0 | GPU v3 | Cambio |
|---|---:|---:|---:|
| Tiempo total (s) | 2.411 | 2.264 | −6.1 % |
| Tiempo de kernels nsys (ms) | 14.013 | 12.039 | 1.16x |
| `render_kernel` promedio (µs) | 9.68 | 7.76 | 1.25x |
| `simulate_step_kernel` promedio (µs) | 5.89 | 5.62 | 1.05x |
| Tiempo H→D nsys (ms) | 379.72 | 362.32 | igual volumen |
| Tiempo D→H nsys (ms) | 210.07 | 200.76 | igual volumen |
| Pasos/s | 373.52 | 397.67 | +6.5 % |
| Speedup vs CPU | 5.38x | 5.73x | |
| Speedup vs GPU anterior (v0) | 1.00x | 1.06x | |
| Error máximo vs CPU | 6.58e-05 | 6.58e-05 | igual |

**Cuánto mejoró el rendimiento:** `render_kernel` baja de 9.68 µs a 7.76 µs (20 % menos), porque fast math usa versiones aproximadas del hardware de `powf`, `sqrtf` y la división. `simulate_step_kernel` casi no cambia porque solo hace sumas y multiplicaciones. En el tiempo total la mejora es de solo 0.15 s, y la mayor parte de esa diferencia está en el video y las copias, que fast math no modifica, por lo que es variación entre corridas. Los kernels representan solo el 3 % de la ejecución.

**Impacto sobre la precisión numérica:** se repitió la validación por el cambio en la representación numérica. El error máximo de la malla no cambia (6.58e-05), el error medio baja ligeramente a 1.755e-06 y los píxeles distintos bajan de 61 329 a 61 223. Fast math no tuvo costo de precisión visible. Repetir la validación responde a una advertencia de León-Vega (2026): como esta opción afecta a todo el programa, solo debe aceptarse después de revisar cómo cambia la precisión del resultado final (p. 161).

---

## 11. Tabla resumen

Equipo principal (i5-12400F + RTX 4060 Ti), 900 pasos, promedio de 3 corridas.

| Versión | Tiempo total | Pasos/s | Speedup | Error máximo |
|---|---:|---:|---:|---:|
| CPU base | 12.963 s | 69.43 | 1.00x | — |
| GPU base (v0) | 2.411 s | 373.52 | 5.38x | 6.58e-05 |
| Optimización 1 — reducción de transferencias (v1) | 1.321 s | 681.18 | **9.81x** | 6.58e-05 |
| Optimización 2 — memoria compartida (v2)\* | 2.736 s | 329.27 | 5.32x (estimado) | 6.58e-05 |
| Optimización 3 — fast math (v3) | 2.264 s | 397.67 | 5.73x | 6.58e-05 |

\* La v2 se midió en el equipo secundario (RTX 4060 Laptop), donde la v0 tardó 2.709 s. Su speedup respecto a la CPU se estimó como 5.377 × (2.709 / 2.736) = 5.32x (ver sección 10.2).

**Desglose de tiempos GPU (nsys)**

| Versión | Kernels (ms) | H→D (ms) | D→H (ms) | Speedup vs v0 |
|---|---:|---:|---:|---:|
| GPU base (v0) | 14.013 | 379.72 | 210.07 | 1.00x |
| Optimización 1 (v1) | 13.199 | 0.25 | 84.31 | 1.82x |
| Optimización 2 (v2)\* | 31.02 (`simulate_step`, `cudaEvent`) | igual que v0 | igual que v0 | 0.99x |
| Optimización 3 (v3) | 12.039 | 362.32 | 200.76 | 1.06x |

Separar el tiempo en cálculo, subida de datos y bajada de datos sigue la forma en que León-Vega (2026) reparte el tiempo total de la GPU (pp. 131–132), y permite ver qué parte mejora cada optimización.

---

## 12. Análisis de rendimiento

- **Dónde se va el tiempo.** En v0, los kernels representan alrededor del 3 % del total, las copias alrededor del 42 % y `putText` más el video alrededor del 43 %. En otras palabras, el programa no está frenado por el cálculo, sino por el ir y venir de datos y por la parte que sigue en el procesador, algo que se entiende bien con la forma en que León-Vega (2026) explica de qué se compone el tiempo de la GPU (pp. 131–132).
- **Mayor impacto.** La Optimización 1 es la única que ataca el cuello de botella real: reduce el volumen transferido en un 84 % y logra 1.82x sobre v0 (9.81x sobre CPU) sin tocar los kernels. Esto coincide con lo que plantea León-Vega (2026): hay que optimizar pensando en el programa completo y no solo en los cálculos, porque reducir el intercambio de datos puede mejorar el tiempo total aunque los cálculos sigan igual (pp. 138, 142).
- **Optimizaciones que no mejoraron.** La memoria compartida (v2) hizo el kernel un 17.7 % más lento. Un stencil de 5 puntos reutiliza pocos datos, probablemente la caché de la GPU ya aprovechaba esa poca reutilización, y copiar los bordes extra y esperar a todo el grupo agrega trabajo; León-Vega (2026) advierte sobre ambos costos (pp. 141, 151). Fast math (v3) sí acelera el render en un 25 %, pero su efecto en el total es pequeño (1.06x).
- **Ley de Amdahl.** Si los kernels son el 3 % del tiempo, aunque se aceleraran infinitamente el tiempo total solo bajaría un 3 %. Por eso v2 y v3 casi no se notan en el total. Después de v1, el 72 % del tiempo es `putText` y `VideoWriter` en CPU, así que el límite teórico de cualquier mejora en la GPU es de alrededor de 1.4x adicional.
- **Precisión contra velocidad.** Ninguna optimización cambió el error máximo (6.58e-05) ni la diferencia de píxel (1 nivel de gris). Fast math fue "gratis" en precisión porque la malla no usa funciones aproximadas. León-Vega (2026) resume este criterio en una frase sencilla: no usar más precisión de la necesaria (p. 166).
- **Optimizaciones independientes.** Las tres optimizaciones parten de v0, así que no se midió su efecto combinado. Como v3 acelera los kernels y v1 elimina las copias, se espera que combinarlas sea complementario.
- **Jetson Nano.** Como explica León-Vega (2026), en un equipo como la Jetson el procesador y la GPU comparten la misma memoria, así que mover datos es más barato que por PCIe, aunque sigue teniendo un costo (pp. 142–143). En cambio, su GPU (128 núcleos Maxwell) es mucho más lenta. Se esperaría que los kernels pesen más en el total y que fast math tenga un efecto mayor.

---

## 13. Conclusiones

1. Portar `simulate_step` y `render_frame` a CUDA dio 5.38x sobre CPU, aunque los kernels solo ocupan alrededor del 3 % del tiempo de la GPU v0. El resto se va en copias y en escribir el video.
2. Reducir la comunicación host-device (Optimización 1) fue la optimización más efectiva: 1.82x sobre v0 y 9.81x sobre CPU, con un 84 % menos de datos transferidos y el mismo resultado numérico.
3. La memoria compartida (Optimización 2) no mejoró el rendimiento: el kernel tardó un 17.7 % más que en v0 (0.85x). Agregar una técnica "de optimización" sin evidencia de que el kernel esté limitado por memoria no garantiza una mejora.
4. `--use_fast_math` (Optimización 3) aceleró `render_kernel` un 25 % sin pérdida de precisión visible, pero por la ley de Amdahl el efecto en el total fue de solo 1.06x.
5. Todas las versiones GPU se validaron contra la CPU con un error máximo de 6.58e-05 y como mucho 1 nivel de gris de diferencia.

---

## 14. Instrucciones de compilación y ejecución

```bash
cd drop-simulation
LDFLAGS="-L$CONDA_PREFIX/lib" meson setup build -Dgpu_arch=89   # Jetson Nano: -Dgpu_arch=53
meson compile -C build

# CPU base
./build/drop_simulation

# Versiones GPU
./build/drop_simulation_gpu_v0
./build/drop_simulation_gpu_v1
./build/drop_simulation_gpu_v3

# Validación contra CPU (genera validation/vN_errors.csv)
./build/drop_simulation_gpu_v0 --validate

# Perfilado
nsys profile -f true -o profiling/gpu/v0/v0 --stats=true ./build/drop_simulation_gpu_v0
```

**Salida:** `output/drop_simulation.mp4` (video de la versión final: _ruta_).

---

## 15. Anexos

- Código CPU comentado: [drop_simulation_comentado_revisado.cpp](drop-simulation/src/drop_simulation_comentado_revisado.cpp)
- Ejercicios A, B y Optimización 1: [README_Lab9_A_B.md](README_Lab9_A_B.md)
- Lectura, perfilado CPU y Optimización 2: [EJ_C - Perfilado y rendimiento en CPU.md](EJ_C%20-%20Perfilado%20y%20rendimiento%20%20en%20CPU.md)
- GPU v0, validación y Optimización 3: [ronald_gpu.md](drop-simulation/docs/ronald_gpu.md)
- Perfiles de Nsight Systems: [profiling/gpu/](drop-simulation/profiling/gpu/)
- Validación: [validation/](drop-simulation/validation/)
- Declaración de uso de IA: _según la norma del curso._

---

## 16. Referencias

León-Vega, L. G. (2026). Capítulo 3: Unidades de procesamiento gráfico.(pp. 115–166). Instituto Tecnológico de Costa Rica.