
# Análisis de la lectura y resultados del laboratorio

## 1. Relación de la lectura con la simulación

### 1.1. Paralelismo

En la lectura se menciona el paralelismo, que básicamente es la capacidad de hacer varias operaciones al mismo tiempo en lugar de una por una. Esto se conecta con la simulación porque en cada paso se hacen muchísimos cálculos sobre diferentes puntos de la malla.

Estos cálculos se pueden distribuir entre varios hilos de la GPU para intentar reducir el tiempo de procesamiento, aunque esto no garantiza que siempre se obtenga una mejora. Esto también se relaciona con el modelo SIMT que se explica en la lectura.

### 1.2. Uso de memoria

Otro punto importante es el manejo de memoria. No solo importa qué tan rápido sea el procesador, sino también cómo se accede a los datos.

En la lectura se mencionan los accesos *coalesced*, que permiten aprovechar mejor la memoria cuando los hilos acceden a posiciones consecutivas. También se explica el uso de *shared memory* para reutilizar datos sin tener que buscarlos constantemente en la memoria global.

Esto es importante para nuestra simulación porque se trabaja con muchos datos que se actualizan en cada paso. Por eso, mejorar el acceso a memoria podría ayudarnos a obtener un mejor rendimiento.

### 1.3. CPU y GPU

La lectura también explica que la GPU puede manejar muchos hilos al mismo tiempo, pero eso no significa que siempre sea más rápida que la CPU.

En nuestro caso, se puede aprovechar la GPU para realizar ciertos cálculos de la simulación, pero primero hay que identificar cuáles se pueden ejecutar de manera independiente.

También hay que considerar el tiempo que toma transferir información entre la CPU y la GPU, porque si se hacen demasiadas transferencias podríamos perder parte de la mejora obtenida.

Estos conceptos se relacionan directamente con las pruebas que realizamos, ya que no solo buscamos ejecutar el programa en GPU, sino también comprobar si los cambios realmente mejoran el rendimiento.

---

## 2. Ejercicio C – Perfilado del programa en CPU

Para este ejercicio se usaron **perf** y **Callgrind** para revisar qué partes del programa consumen más recursos e identificar cuáles podrían ejecutarse en GPU para mejorar el rendimiento.

La simulación trabaja con una malla de **640 × 640 píxeles** y genera **900 cuadros**.

Con Callgrind se obtuvo un tiempo de **330.224 segundos** y se registraron **122,782,185,856 referencias a instrucciones (Ir)**.

Según perf, `__powf_fma` concentró el **37.34%** de las muestras de ciclos de CPU. Esta función se utiliza en `render_frame()` para calcular la iluminación de los píxeles.

### 2.1. Resultados de Callgrind

| Función o componente | Porcentaje |
|---|---:|
| `main` (costo inclusivo) | 99.51% |
| `main` (código propio) | 43.41% |
| `__powf_fma` | 19.19% |
| `cv::VideoWriter::write` | 13.68% |
| `sws_scale` | 8.41% |

**Nota:** En Callgrind, `main` aparece varias veces porque el reporte incluye costos inclusivos y código integrado por el compilador. No debemos sumar esos porcentajes ni interpretarlos como tiempos independientes.

### 2.2. Métricas de ejecución

| Métrica | Resultado |
|---|---:|
| Tiempo CPU sin instrumentación | 11.4182 s |
| Tiempo con Callgrind | 330.224 s |
| Referencias a instrucciones (Ir) | 122,782,185,856 |
| Resolución | 640 × 640 |
| Cuadros | 900 |

Estos resultados muestran por qué no debemos utilizar el tiempo de Callgrind como referencia directa para calcular el *speedup*, ya que la herramienta agrega una sobrecarga considerable durante la ejecución.

### 2.3. Funciones que conviene pasar a GPU

- **`render_frame()`:** procesa muchos píxeles por cuadro y realiza cálculos repetitivos. Estas operaciones se pueden paralelizar en GPU para intentar reducir el tiempo de procesamiento.

- **`simulate_step()`:** actualiza muchos puntos de la simulación en cada iteración utilizando los valores de sus vecinos. También realiza operaciones repetitivas que se pueden distribuir entre varios hilos de la GPU.

### 2.4. Funciones que no conviene priorizar para GPU

- **`add_drop()`:** en la ejecución analizada solo se utiliza una vez al inicio, por lo que optimizarla tendría poco impacto sobre el tiempo total.

- **`cv::VideoWriter::write()`:** aunque consume recursos, su función principal es guardar el video y no realizar los cálculos de la simulación. Por eso no sería nuestra primera prioridad.

### 2.5. Limitaciones del perfilado

- No se puede afirmar que `render_frame()` consume exactamente el **37.34%** de los ciclos de CPU, porque ese porcentaje corresponde a las muestras asociadas a `__powf_fma`, y la función realiza otras operaciones.

- Callgrind ralentiza considerablemente la ejecución, por lo que los **330.224 segundos** no representan el tiempo normal del programa.

- Algunas funciones fueron integradas por el compilador dentro de `main`, por lo que no se pudieron obtener porcentajes individuales confiables para `render_frame()` y `simulate_step()`.

A partir de estos resultados identificamos que `render_frame()` y `simulate_step()` son buenas candidatas para ejecutarse en GPU, ya que realizan cálculos repetitivos durante la simulación.

---

## 3. Optimización 2 – CUDA con memoria compartida

Para esta parte se implementó una versión de la simulación utilizando memoria compartida (*shared memory*) en CUDA.

La idea fue reutilizar los valores de los puntos vecinos dentro de cada bloque de hilos, evitando realizar todos los accesos directamente a memoria global.

Se utilizó una estructura de datos compartidos con espacio adicional para los vecinos de los bordes, conocidos como *halos*. También se utilizó sincronización entre los hilos para asegurar que los datos estuvieran cargados antes de realizar los cálculos.

Se probaron dos configuraciones de bloques:

- **16 × 16:** 256 hilos por bloque.
- **32 × 8:** 256 hilos por bloque.

La segunda configuración permite que los 32 hilos de un *warp* recorran posiciones consecutivas de una fila, lo que puede favorecer los accesos coalescentes.

### 3.1. Resultados de rendimiento

Las pruebas se realizaron en una **NVIDIA GeForce RTX 4060 Laptop GPU**, ejecutando 900 pasos de simulación.

Los resultados corresponden al promedio de tres ejecuciones por configuración, sin activar el modo de validación.

| Métrica | GPU v0 | GPU v2 (16x16) | GPU v2 (32x8) |
|---|---:|---:|---:|
| Tiempo total | 2.709 s | 2.755 s | 2.736 s |
| Kernel `simulate_step` | 26.35 ms | 32.54 ms | 31.02 ms |
| Rendimiento (pasos/s) | 332.26 | 326.97 | 329.27 |

### 3.2. Observaciones

- La configuración 32x8 fue aproximadamente un **4.7% más rápida** que 16x16 en el kernel de simulación.
- Sin embargo, al compararla con v0, el kernel de 32x8 fue un **17.7% más lento**.
- El *speedup* del kernel respecto a v0 fue de **0.85x**, por lo que no se consiguió una aceleración.
- El tiempo total de ejecución cambió muy poco entre las versiones, aunque sí encontramos diferencias en el tiempo del kernel.

Estos resultados muestran que utilizar memoria compartida no necesariamente mejora el rendimiento.

En este caso, una posible explicación es que la carga de datos adicionales de los halos y la sincronización entre los hilos estén generando más trabajo del que se logra ahorrar. Sin embargo, necesitaríamos un perfilado más detallado para comprobarlo.

---

## 4. Validación contra CPU

También se realizó una validación de la versión v2 contra la implementación en CPU para comprobar que los resultados de la simulación fueran cercanos a los originales.

Se validaron los 900 cuadros de la simulación.

| Métrica | Resultado |
|---|---:|
| Cuadros validados | 900 |
| Error absoluto máximo | 6.580 × 10⁻⁵ |
| Error absoluto medio | 1.762 × 10⁻⁶ |
| Diferencia máxima de píxel | 1 nivel de gris |
| Píxeles distintos acumulados | 61,329 |

Las dos configuraciones de v2 presentaron las mismas métricas de error durante la validación.

Aunque existen pequeñas diferencias numéricas respecto a CPU, los errores obtenidos fueron bajos. Además, la diferencia máxima en las imágenes fue de solamente un nivel de gris.

Los 61,329 píxeles distintos corresponden al total acumulado durante los 900 cuadros, no a una sola imagen.

Esto nos indica que el cambio en la distribución de los bloques no afectó de manera importante los resultados de la simulación.

**Nota:** Los tiempos obtenidos durante la validación no se utilizaron para comparar rendimiento, porque el modo de validación agrega trabajo adicional al programa.

---


## 5. Conclusiones generales

1. A pesar de utilizar memoria compartida, no se consiguió mejorar el rendimiento del kernel respecto a la versión original v0. Esto demuestra que agregar una optimización no siempre significa que el programa vaya a ejecutarse más rápido.

2. La configuración 32x8 presentó mejores tiempos promedio que 16x16, aunque la diferencia fue pequeña y puede estar relacionada con la variabilidad de las mediciones.

3. El tiempo total de ejecución fue bastante similar entre las versiones, por lo que las diferencias observadas en el kernel no tuvieron un impacto tan grande sobre el rendimiento general del programa. Esto también demuestra la importancia de considerar las transferencias de memoria y otras operaciones que forman parte de la ejecución.
---

## 6. Recomendaciones

1. Realizar el perfilado antes de implementar optimizaciones, para identificar cuáles funciones realmente consumen más recursos y evitar trabajar en partes que tienen poco impacto sobre el rendimiento.

2. Probar otras configuraciones de bloques para comprobar si existe alguna distribución que aproveche mejor la memoria compartida.

3. Realizar más pruebas de rendimiento para obtener resultados más consistentes y determinar si las diferencias entre configuraciones representan una mejora real.
