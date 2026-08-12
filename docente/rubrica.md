# Rúbrica de evaluación — Anexo Práctico de IA

**Desarrollo de Sistemas de Inteligencia Artificial** · Trayecto F · 2.º año
ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026

Esta rúbrica se alinea con las tres categorías del programa anual —**APROBADO**,
**PROMOVIDO**, **DISTINGUIDO**— y con los contenidos mínimos de aprobación.

---

## Principio general

El anexo **no evalúa si el código corre**. Las celdas de verificación ya hacen
eso, y un estudiante puede pasarlas copiando. Lo que se evalúa es **el criterio
técnico**, que se manifiesta en tres lugares:

1. Las decisiones tomadas en los ejercicios `[I]` (qué umbral, qué
   característica, qué modelo) **y su justificación**.
2. Los textos de los ejercicios `[A]`, que no tienen respuesta única.
3. La coherencia entre lo que el estudiante afirma y lo que sus propios números
   muestran.

> El indicador más confiable de comprensión, en este material, es que el
> estudiante **reporte un resultado que no le convenía**. Quien escribe "probé un
> modelo, no le gané al baseline, y por eso me quedo con la regla" entendió el
> cuatrimestre entero.

---

## Ponderación

| Componente | Peso |
|---|---|
| Ejercicios `[B]` e `[I]` completos y ejecutados | 30 % |
| Las cinco entregas `[A]` obligatorias | 40 % |
| Coherencia entre los cuadernos y el informe F-5 | 20 % |
| Defensa oral (5 minutos, cualquier cuaderno a elección del docente) | 10 % |

Las cinco entregas `[A]` obligatorias son: **1.5** (un gráfico de afuera de la
materia, analizado con los cinco pasos), **A2.5** (reporte de calidad de datos),
**A4.4** (costo del error y métrica), **A5.4** (decisión fundamentada reglas
vs. modelo) y **A7.4** (ficha técnica del modelo).

---

## Criterios por nivel

### APROBADO — cumple los contenidos mínimos

| Dimensión | Qué se espera |
|---|---|
| **Ejecución** | Los cuadernos 0, 1 y A-0 a A-5 ejecutados de punta a punta, con salidas visibles. |
| **Ejercicios** | Todos los `[B]` y al menos el 70 % de los `[I]` en `[OK]`. |
| **Diccionario de variables** | Completo, con las cuatro definiciones (variable, unidad, rango, frecuencia) para al menos dos variables. |
| **Limpieza** | Aplicó las cuatro reglas del módulo F-2 y reporta cuántas lecturas descartó cada una. |
| **Modelo** | Entrenó al menos un modelo y reporta accuracy, precision y recall. |
| **Comparación** | Comparó reglas contra modelo con la misma métrica. |
| **Límites** | Declaró al menos una limitación de su resultado. |

**No aprueba** quien: reporta solo `accuracy`; no calcula ningún baseline; usa
partición al azar sobre datos temporales sin advertirlo; o entrega cuadernos sin
salidas.

---

### PROMOVIDO — supera los mínimos

Todo lo anterior, más:

| Dimensión | Qué se espera |
|---|---|
| **Justificación de la métrica** | Eligió precisión o sensibilidad razonando desde el **costo real** de cada error en su proyecto, no desde cuál da más alto. |
| **Umbral de éxito previo** | Declaró en A4.4 un valor mínimo aceptable **antes** de entrenar, y lo respetó al evaluar. |
| **Ingeniería de características** | Construyó al menos una característica propia (ventana móvil, pendiente, diferencia) y explica qué fenómeno captura. |
| **Visualización** | Sus figuras tienen la pregunta como título y una conclusión de una línea sobre el fenómeno, con un número. |
| **Protocolo** | Partición temporal correcta, y estadísticos de normalización calculados solo sobre entrenamiento. |
| **Documentación** | El informe F-5 se sostiene solo: alguien ajeno al equipo lo entiende. |

---

### DISTINGUIDO — autonomía y valor agregado

Todo lo anterior, más **al menos tres** de estos:

| Indicador | Cómo se reconoce |
|---|---|
| **Regla derivada del modelo** | Completó A5.3: transcribió el árbol a una función simple, la midió, y cuantificó cuánto pierde al simplificar. Bonus si la implementó en el firmware real. |
| **Un resultado negativo bien reportado** | Documenta un enfoque que probó y no funcionó, con la métrica que lo demuestra y una hipótesis de por qué. |
| **Cuestionó el material** | Encontró una limitación de los datos simulados, o un supuesto discutible del anexo, y lo argumentó. |
| **Extendió A-6 o A-7** | Completó A6.4 (dos capas ocultas con gradiente verificado) o A7.2 con un corpus propio no trivial. |
| **Análisis de sensibilidad** | Mostró cómo cambia su conclusión al variar un parámetro (umbral, ventana, profundidad), en lugar de reportar un único número. |
| **Ética aplicada, no recitada** | En A7.4 o A8.4 identificó un riesgo específico y verificable **de su propio proyecto**, con una mitigación concreta. |

---

## Señales de alarma en la corrección

Cosas que conviene mirar específicamente, porque son fáciles de pasar por alto:

| Señal | Qué revisar |
|---|---|
| Métricas sospechosamente altas (F1 > 0.99) | Casi seguro hay fuga de datos: ¿interpoló el objetivo? ¿partió al azar? ¿usó una característica que contiene la respuesta? |
| El umbral "elegido" coincide exactamente con el óptimo de prueba | Ajustó mirando el conjunto de prueba. |
| Conclusiones sin ningún número | Describe el gráfico en vez del fenómeno. |
| El texto de A5.4 contradice la tabla de A5.3 | No leyó sus propios resultados. |
| Todos los ejercicios `[A]` con la misma estructura entre grupos distintos | Probable copia o texto generado; preguntar en la defensa oral. |
| El código usa funciones que el anexo no introdujo | No es un problema en sí. Preguntar en la defensa: si lo puede explicar, suma. |

---

## Sobre la defensa oral

Cinco minutos, un cuaderno elegido por el docente. Tres preguntas que funcionan
bien porque no se pueden contestar sin haber hecho el trabajo:

0. *"Mostrame un gráfico de tu carpeta y aplicale los cinco pasos del Cuaderno 1."*
1. *"Mostrame el número que más te sorprendió y explicame por qué."*
2. *"Si tuvieras que bajar el umbral a la mitad, ¿qué pasaría con tu sistema y
   quién se quejaría?"*
3. *"¿Qué no podés afirmar con estos datos?"*

La tercera es la que mejor discrimina. Quien contesta "nada, funciona bien" no
llegó al nivel de PROMOVIDO, por más que todos los cuadernos estén en `[OK]`.
