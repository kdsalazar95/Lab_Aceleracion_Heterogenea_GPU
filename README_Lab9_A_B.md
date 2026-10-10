# Laboratorio Semana 9: Ejercicios A y B

**Autor:** Keylor Muñoz  
**Curso:** EL5859 — Computación Heterogénea  
**Periodo:** II Semestre 2026

---

## Ejercicio A. Construcción y ejecución del código base

Se compiló y ejecutó el programa original `drop_simulation` para establecer una línea base de rendimiento antes de realizar optimizaciones. Esta versión implementa la simulación en CPU y no contiene código CUDA, por lo que los resultados de este ejercicio corresponden a la ejecución en CPU.

### Entorno de ejecución

| Componente | Detalle |
|---|---|
| CPU | Intel Core i3-10100F @ 3.60 GHz (turbo de hasta 4.3 GHz), 4 núcleos y 8 hilos |
| Memoria RAM | 15 GiB |
| Sistema operativo | Pop!_OS 24.04 LTS, kernel 7.1.5 |
| Compilador | g++ 13.3.0 |
| Herramientas de construcción | Meson 1.3.2 y Ninja 1.11.1 |
| OpenCV | 4.6.0 |
| Opciones de compilación | `buildtype=release` (`-O3`), C++17 |
| Perfil de energía | `Balanced`; governor `powersave` de `intel_pstate` |
| GPU instalada | NVIDIA GeForce GTX 1650; no utilizada en este ejercicio |

### Comandos ejecutados

```bash
cd drop-simulation
meson setup build
meson compile -C build
./build/drop_simulation

# Cinco corridas para calcular la media y la desviación:
bash scripts/bench_cpu.sh 5
```

### Resultados

| Métrica | Resultado |
|---|---:|
| Compilación | Exitosa, sin errores |
| Tiempo de ejecución (5 corridas) | **15.74 ± 0.17 s** |
| Tiempo mínimo | 15.62 s |
| Tiempo máximo | 16.01 s |
| Pasos simulados por corrida | 900 |
| Rendimiento | **57.2 ± 0.6 pasos/s** |
| Resolución del video | 640 × 640 píxeles |
| Duración del video | 30 s |
| Tasa de cuadros | 30 FPS |
| Archivo generado | `output/drop_simulation.mp4` |
| Datos de las corridas | `results/cpu_baseline.csv` |

La resolución, la tasa de cuadros y la duración se comprobaron con `ffprobe` sobre el video generado. Estos valores coinciden con los parámetros definidos en `Config`.

El video dura 30 segundos, mientras que la simulación tarda aproximadamente 15.74 segundos en generarlo. Por tanto, la ejecución tarda cerca de la mitad del tiempo de reproducción del video. El tiempo medio y la dispersión reportados corresponden a las cinco corridas del script de medición. Estos resultados se utilizarán como referencia para comparar las versiones optimizadas.

> **Nota:** la ejecución inicial también reportó 15.6179 s para una sola corrida. La tabla anterior presenta los resultados agregados de las cinco corridas del script de medición.

---

## Ejercicio B. Comprensión del código

El código fuente comentado se encuentra en `src/cpu/drop_simulation.cpp`. El diagrama de flujo resume las etapas principales del programa.

### ¿Qué hace el programa?

El programa simula la caída de una gota sobre un estanque y guarda el resultado en un video en escala de grises. La superficie del agua se representa mediante una malla de 640 × 640 valores de altura; cada celda corresponde a una posición de la superficie y a un píxel del video.

Al inicio, `add_drop` crea una perturbación en el centro de la malla. En cada paso de simulación, la altura de cada celda se calcula a partir de sus cuatro vecinas y de los valores de altura de los dos instantes anteriores. Así, la perturbación se propaga como ondas por la superficie. El amortiguamiento aumenta cerca de los bordes para reducir los rebotes. Después, `render_frame` convierte las alturas en una imagen y el programa la escribe en el archivo de video. Este proceso se repite hasta completar los 900 cuadros.

### Modelo físico

La simulación aproxima la **ecuación de onda bidimensional amortiguada**:

```text
∂²h/∂t² = c² · ∇²h − γ · ∂h/∂t
```

Donde:

- `h(x, y, t)` representa la altura de la superficie en una posición y un instante.
- `c² · ∇²h` representa la propagación de las ondas a través de la superficie.
- `γ · ∂h/∂t` representa el amortiguamiento, que reduce el movimiento con el tiempo.

La ecuación se resuelve mediante **diferencias finitas explícitas**. Para calcular la altura siguiente, el programa utiliza los estados anterior y actual, además de las cuatro celdas vecinas:

```text
siguiente = 2 · actual − anterior
            + c² · (izquierda + derecha + arriba + abajo − 4 · actual)
            − amortiguamiento_local · (actual − anterior)
```

La velocidad de onda configurada es `0.45` celdas por paso. Este valor está por debajo del límite aproximado de `0.707` indicado para la condición de estabilidad de este esquema en dos dimensiones.

### Funciones principales

| Elemento | Función |
|---|---|
| `Config` | Agrupa los parámetros de la malla, el video, la velocidad de onda, el amortiguamiento y la gota inicial. |
| `index_of` | Convierte las coordenadas `(x, y)` en el índice correspondiente dentro del vector que almacena la malla. |
| `add_drop` | Crea la perturbación inicial en el centro de la superficie. |
| `border_absorption` | Aumenta el amortiguamiento cerca de los bordes para reducir el rebote de las ondas. |
| `simulate_step` | Calcula las nuevas alturas de la malla mediante la ecuación de onda amortiguada. |
| `render_frame` | Convierte la malla en una imagen en escala de grises y añade el número del cuadro. |
| `main` | Inicializa los datos, ejecuta los ciclos de simulación, genera el video y calcula las métricas finales. |

### Diagrama de flujo

![Diagrama de flujo de la simulación](docs/diagrama_flujo.png)

*Nota: ajusta la ruta de la imagen si guardaste el diagrama en otra carpeta del repositorio.*

El programa inicializa los parámetros, reserva las tres mallas (`previous`, `current` y `next`), aplica la perturbación inicial y abre el archivo de video. Luego recorre los cuadros y ejecuta los pasos de simulación correspondientes. En cada paso se actualiza la malla y se intercambian los buffers mediante `swap`, evitando copiar todo su contenido. Después se renderiza y se escribe el cuadro. Al terminar los 900 cuadros, el programa detiene el cronómetro y muestra el tiempo de ejecución, los pasos simulados y el rendimiento.

### Observaciones para las siguientes etapas

A partir de la lectura del código, se identifican estos aspectos para analizar en las siguientes etapas:

- Con `steps_per_frame = 1`, se genera un cuadro de video por cada paso de simulación.
- `simulate_step` y `render_frame` recorren la malla o la imagen y son candidatas a paralelizarse en la GPU.
- La escritura del video mediante `writer.write` permanece en CPU, por lo que su costo debe tenerse en cuenta al evaluar la mejora total.
- Estas observaciones se deben contrastar con las mediciones y el perfilado de las siguientes etapas.

---

## Siguiente etapa: Optimización 1

La optimización para reducir la comunicación entre el host y el dispositivo queda pendiente para la versión base implementada en GPU.
