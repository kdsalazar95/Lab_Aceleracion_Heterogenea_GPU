
## LECTURA ANALISIS ##

## Relación de la lectura con la simulación

### Paralelismo

En la lectura se menciona el paralelismo, que básicamente es la capacidad de hacer varias operaciones al mismo tiempo en lugar de una por una. Esto se conecta con la simulación porque en cada paso se hacen muchísimos cálculos sobre diferentes puntos de la malla.

Si logramos distribuir algunos de estos cálculos entre varios hilos de la GPU, podríamos reducir el tiempo de procesamiento. Esto se relaciona con el modelo SIMT que se explica en el capítulo.

### Uso de memoria

Otro punto importante es el manejo de memoria. No solo importa qué tan rápido sea el procesador, sino también cómo se accede a los datos.

En la lectura se mencionan los accesos *coalesced*, que permiten aprovechar mejor la memoria cuando los hilos acceden a posiciones consecutivas. También se explica el uso de *shared memory* para reutilizar datos sin tener que buscarlos constantemente en la memoria global.

Esto es importante para nuestra simulación porque se trabaja con muchos datos que se actualizan en cada paso. Por eso, mejorar el acceso a memoria podría ayudarnos a obtener un mejor rendimiento.

### CPU y GPU

La lectura también explica que la GPU puede manejar muchos hilos al mismo tiempo, pero eso no significa que siempre sea más rápida que la CPU.

En nuestro caso, se podría aprovechar la GPU para realizar ciertos cálculos de la simulación, pero primero hay que identificar cuáles se pueden ejecutar de manera independiente.

También hay que considerar el tiempo que toma transferir información entre la CPU y la GPU, porque si se hacen demasiadas transferencias podríamos perder parte de la mejora obtenida.

### Conclusiones

Con los resultados de *perf* y *Callgrind* pudimos identificar operaciones que tienen un costo importante dentro de la simulación, especialmente las operaciones matemáticas. Esto nos ayuda a decidir cuáles partes vale la pena intentar optimizar.

También queda claro que no siempre es necesario cambiar todo el programa para mejorar el rendimiento, sino identificar primero las partes que más consumen.

En resumen, lo más importante de la lectura es entender que el rendimiento depende del paralelismo, del uso correcto de la memoria y de cómo se distribuye el trabajo entre los procesadores.

## Ejercicio C – Perfilado del programa en CPU

Para este ejercicio se usaron **perf** y **Callgrind** para revisar qué partes del programa consumen más recursos y pensar cuáles se podrían pasar a GPU para mejorar el rendimiento.

La simulación trabaja con una malla de **640 × 640 píxeles** y genera **900 cuadros**.

Con Callgrind se obtuvo un tiempo de **330.224 segundos** y se registraron **122,782,185,856 instrucciones**.

Según perf, **`__powf_fma`** concentró el **37.34 %** de las muestras de ciclos de CPU. Esta función se usa en `render_frame()` para calcular la iluminación de los píxeles.

### Resultados de Callgrind

| Función o componente | Porcentaje |
|---|---:|
| `main` (costo inclusivo) | 99.51 % |
| `main` (código propio) | 43.41 % |
| `__powf_fma` | 19.19 % |
| `cv::VideoWriter::write` | 13.68 % |
| `sws_scale` | 8.41 % |

**Nota:** En Callgrind, `main` aparece varias veces porque el reporte incluye costos inclusivos y código integrado por el compilador. No debemos sumar esos porcentajes ni interpretarlos como tiempos independientes.

### Métricas de ejecución

| Métrica | Resultado |
|---|---:|
| Tiempo CPU sin instrumentación | 11.4182 s |
| Tiempo con Callgrind | 330.224 s |
| Instrucciones registradas | 122,782,185,856 |
| Resolución | 640 × 640 |
| Cuadros | 900 |

Esto muestra por qué no debemos usar el tiempo de Callgrind como referencia directa para calcular el *speedup*.

### Funciones que conviene pasar a GPU

- **`render_frame()`:** procesa muchos píxeles por cuadro y hace cálculos repetitivos. En GPU se pueden paralelizar y reducir el tiempo.

- **`simulate_step()`:** actualiza muchos puntos de la simulación en cada iteración usando valores de vecinos. También son operaciones repetitivas que la GPU puede manejar mejor.

### Funciones que no conviene pasar a GPU

- **`add_drop()`:** solo se ejecuta una vez al inicio, así que la mejora sería mínima.

- **`cv::VideoWriter::write()`:** aunque consume recursos, su tarea es guardar el video, no hacer cálculos de la simulación. No sería prioridad.

### Limitaciones

- No se puede decir que `render_frame()` consume exactamente el **37.34 %** porque ese número corresponde solo a las muestras de ciclos de CPU asociadas a `__powf_fma`, y la función hace más cosas.

- Callgrind ralentiza la ejecución, por lo que los **330.224 segundos** no reflejan el tiempo normal del programa.

- Algunas funciones fueron integradas por el compilador dentro de `main`, por lo que no se pudieron obtener porcentajes individuales confiables para `render_frame()` y `simulate_step()`.

### Conclusión

Las mejores funciones para pasar a GPU son **`render_frame()`** y **`simulate_step()`**, porque realizan muchos cálculos durante toda la simulación y se pueden ejecutar en paralelo.

En cambio, **`add_drop()`** y la parte de guardar el video no son prioridad.

Con esto ya tenemos una idea clara de qué partes del programa se pueden optimizar y luego comparar los resultados para ver si realmente mejora el rendimiento.
