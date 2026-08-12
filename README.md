# Anexo Práctico de Inteligencia Artificial — 2.º cuatrimestre 2026

**Desarrollo de Sistemas de Inteligencia Artificial** · Trayecto F · 2.º año
Tecnicatura Superior en Internet de las Cosas y Sistemas Embebidos
ISFT N.º 238 — Prof. Lucas Daniel Gómez

---

## Qué es esto

Once cuadernos de Jupyter que acompañan al **Cuadernillo del Trayecto F**
(módulos F-0 a F-5) y a la parte de aprendizaje automático del **Trayecto G**
(módulos G-2 y G-3).

El cuadernillo dice **qué** hay que decidir sobre el proyecto. Este anexo da el
**cómo**, con código que corre de verdad y que el estudiante puede modificar.
Cada cuaderno mezcla teoría, demostración ejecutable y ejercicios con
autoverificación.

**Todo funciona sin conexión a internet y sin placa de video**, con la única
excepción del cuaderno A-8, que es opcional y está marcado como tal. Los modelos
que se entrenan son deliberadamente diminutos: la escala chica es lo que permite
ver el mecanismo completo en lugar de una caja negra.

---

## Instalación

### Opción A — Guion automático (Linux o macOS)

```bash
cd "Anexo_Practico_IA_2026"
bash instalar.sh
```

Crea un entorno virtual en `entorno/`, instala todo y deja andando Jupyter.

### Opción B — Paso a paso (cualquier sistema)

```bash
python3 -m venv entorno

# Linux / macOS
source entorno/bin/activate
# Windows
entorno\Scripts\activate

pip install -r requirements.txt
jupyter lab notebooks/
```

### Verificación

Abrí `notebooks/A0_Entorno_y_herramientas.ipynb` y ejecutá las dos primeras
celdas. Si ves cuatro líneas con `OK`, está todo listo.

Para dar clase, en cambio, se empieza por `notebooks/00_…` y `notebooks/01_…`,
que son los dos cuadernos de arranque.

### Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `externally-managed-environment` al instalar | Estás usando el Python del sistema. Creá el entorno virtual (paso 1). |
| `ModuleNotFoundError: estilo_grafico` | Abriste el cuaderno desde otra carpeta. Jupyter tiene que arrancar en `notebooks/`. |
| `FileNotFoundError` con los `.csv` | Ídem: los cuadernos buscan los datos en `../datos/`. |
| Los gráficos no aparecen | Reiniciá el kernel (*Kernel → Restart & Run All*). |
| Todo anda raro después de editar | Casi siempre es orden de ejecución de celdas. *Restart & Run All*. |

---

## Los once cuadernos

Los dos primeros son de **arranque**: no suponen nada previo, no tienen fórmulas
y se dan antes que todo lo demás. Nacieron de una constatación del ciclo 2026:
los cuadernos A daban por sabido de qué se estaba hablando, y enseñaban a
*hacer* gráficos sin haber enseñado nunca a *leerlos*.

| # | Cuaderno | Vinculación | Duración | Contenido central |
|---|---|---|---|---|
| **0** | ¿De qué hablamos cuando hablamos de IA? | arranque | 100 min | Qué es y qué no; las muñecas rusas; cinco mitos; historia en cinco momentos; dónde entra la IA en tu nodo |
| **1** | Cómo leer un gráfico | arranque | 100 min | El protocolo de los cinco segundos; tendencia contra ruido; el cuarteto de Anscombe; nueve formas de engañar sin datos falsos |
| **A-0** | Entorno y herramientas | previo a F-0 | 90 min | NumPy, pandas, los tres conjuntos de datos |
| **A-1** | Agentes y reglas | **F-0** | 120 min | PEAS, tipos de agente, motor de inferencia IF-THEN |
| **A-2** | Ciclo de vida del dato | **F-1, F-2** | 150 min | Diccionario de variables, los 4 datos inválidos, limpieza |
| **A-3** | Exploración y visualización | **F-3** | 120 min | La pregunta antes del gráfico; tres formas de mentir sin mentir |
| **A-4** | Primer modelo supervisado | Bloque 4, **G-2/G-3** | 150 min | Clasificación, regresión, métricas, baseline, sobreajuste |
| **A-5** | Reglas contra modelo | **G-2, G-3** | 120 min | Comparación honesta; derivar una regla para el ESP32 |
| **A-6** | Red neuronal desde cero | profundización | 150 min | Perceptrón, XOR, retropropagación en NumPy |
| **A-7** | Modelo de lenguaje diminuto | profundización | 150 min | Bigramas, embeddings, softmax, temperatura, perplejidad |
| **A-8** | LLM local *(opcional)* | ampliación | 90 min | Correr un modelo real de 135M en tu máquina |

Total: unas **23 horas** de trabajo en el aula, distribuibles a lo largo del
cuatrimestre según el calendario de los módulos F.

### Correspondencia con el cronograma del cuadernillo

| Semana | Módulo del cuadernillo | Cuaderno del anexo |
|---|---|---|
| 1 | — | **0**, **1** y A-0 |
| 2 | F-0 | A-1 |
| 3 y 4 | F-1 | A-2 (primera parte) |
| 6 | F-2 | A-2 (segunda parte) |
| 7 | F-3 | A-3 |
| 13 | G-2, F-3 | A-4 |
| 13 y 14 | G-3 | A-5 |
| — | ampliación | A-6, A-7, A-8 |

Los cuadernos **0** y **1** van sí o sí antes que el resto: el 0 instala el
vocabulario y el 1 la lectura de gráficos, y a partir de A-2 todo lo que hay que
decidir se decide mirando un gráfico.

Los cuadernos A-6, A-7 y A-8 no están atados a un módulo: son la profundización
teórica que el primer cuatrimestre no alcanzó a cubrir. Se pueden dar como
unidad aparte, repartir entre las semanas libres, o dejar como trabajo optativo
para quien quiera ir más lejos.

---

## Los datos

Tres conjuntos simulados, uno por cada proyecto de la cohorte 2026:

| Archivo | Proyecto | Filas | Período |
|---|---|---|---|
| `datos/enchufe_consumo.csv` | Enchufe inteligente | 2880 | 30 días, cada 15 min |
| `datos/riego_humedad.csv` | Riego automático | 2880 | 60 días, cada 30 min |
| `datos/silobolsa_gas.csv` | Monitoreo de silobolsa | 4320 | 90 días, cada 30 min |

**Están sucios a propósito.** Llevan inyectados los cuatro tipos de dato
inválido del módulo F-2 (fuera de rango físico, faltante, repetido sospechoso y
fuera de tiempo), además del precalentamiento del sensor de gas, que es un
comportamiento real documentado en el datasheet del MQ-135. Limpiarlos es parte
del trabajo del estudiante.

También traen fenómenos que hay que descubrir: el compresor que queda pegado en
el enchufe, y en la silobolsa **tres focos de fermentación reales más cuatro
arranques falsos**, que son los que impiden resolver el problema con un simple
umbral.

La semilla está fija (238), así que todos los grupos trabajan sobre los mismos
números y los resultados se pueden comparar entre equipos. Para regenerarlos:

```bash
python3 datos/generar_datasets.py
```

La clave de corrección —qué se rompió exactamente y en qué filas— queda en
`docente/fallas_inyectadas.json`.

---

## Cómo está armado el repositorio

```
Anexo_Practico_IA_2026/
├── README.md                    este archivo
├── GUIA_DEL_ESTUDIANTE.md       cómo trabajar y qué se entrega
├── requirements.txt
├── instalar.sh
├── notebooks/                   los .ipynb (acá trabaja el estudiante)
│   ├── 00_… 01_…               los dos cuadernos de arranque
│   ├── A0_… … A8_…             el recorrido principal
│   └── estilo_grafico.py        paleta y estilo común de los gráficos
├── datos/
│   ├── generar_datasets.py
│   └── *.csv
├── docente/
│   ├── rubrica.md               criterios de evaluación
│   ├── notas_de_catedra.md      qué esperar de cada cuaderno
│   └── fallas_inyectadas.json   clave de corrección de A-2
└── _fuentes/                    fuentes de los cuadernos (ver abajo)
```

### Sobre `_fuentes/` — cómo editar los cuadernos

Los `.ipynb` **se generan**; no se editan a mano. Las fuentes son archivos `.py`
en formato *percent* (`# %%` separa celdas, `# %% [markdown]` marca celdas de
texto). Después de editar una fuente:

```bash
python3 _fuentes/build_notebooks.py
```

La razón de este rodeo es práctica: un `.ipynb` es un JSON con la salida
embebida, y en control de versiones cada corrida produce un diff enorme e
ilegible. Los `.py` se leen, se comparan y se corrigen como cualquier código.

Si preferís editar los `.ipynb` directamente y olvidarte de las fuentes,
podés hacerlo — pero acordate de que el próximo `build_notebooks.py` los
sobrescribe.

---

## Notas para quien dé la clase

- **Los ejercicios se autoverifican.** Cada uno imprime `[OK]` o `[REVISAR]` con
  una pista. Eso no reemplaza la corrección: los ejercicios `[A]` (avanzados) son
  todos de redacción y no tienen verificación automática, porque son justamente
  los que evalúan criterio.
- **Los resultados numéricos están verificados.** Todas las afirmaciones del
  texto sobre qué pasa al correr el código fueron comprobadas contra la salida
  real. Si algún número no coincide, es que algo cambió en las versiones de las
  librerías: reportalo.
- **Varios cuadernos terminan en una conclusión incómoda a propósito**: la
  regresión lineal pierde contra un baseline trivial (A-4), la red neuronal no le
  gana al árbol (A-6), el modelo comercial y el del cuaderno hacen lo mismo
  (A-7). Eso no es un defecto del material: es el punto. La costumbre de comparar
  contra un baseline y de reportar lo que salió —y no lo que uno esperaba— es la
  competencia profesional que estos cuadernos intentan formar.
- **A-5 tiene la idea más aprovechable para esta carrera**: usar el modelo para
  *descubrir* la regla, y después escribir la regla a mano en el firmware. Si hay
  que recortar contenido, ese bloque no se recorta.

---

## Para usarlo en otra institución

Está pensado para que se pueda adaptar. Lo que hay que tocar:

| Qué | Dónde |
|---|---|
| Los tres proyectos de la cohorte | `datos/generar_datasets.py` — cambiá las tres funciones por los fenómenos de tus estudiantes |
| El encabezado de cada cuaderno | `_fuentes/A*.py`, primeras líneas |
| La correspondencia con módulos | tablas de este README y de `GUIA_DEL_ESTUDIANTE.md` |
| Los criterios de evaluación | `docente/rubrica.md` |
| La paleta de gráficos | `notebooks/estilo_grafico.py` |

Después de cualquier cambio en `_fuentes/`, regenerá con
`python3 _fuentes/build_notebooks.py`.

Lo que **no** conviene tocar, porque es donde está el valor del material: las
demostraciones que terminan en un resultado contrario a la intuición (el agua del
riego en A-1, el promedio que no se mueve en A-2, la regresión lineal que pierde
contra el baseline en A-4, la regla que le gana al árbol en A-5). Todas están
verificadas contra la salida real y documentadas en `docente/notas_de_catedra.md`.

---

## Licencia

[![CC BY-SA 4.0](https://img.shields.io/badge/licencia-CC%20BY--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-sa/4.0/deed.es)

Este material se publica bajo **Creative Commons
Atribución-CompartirIgual 4.0 Internacional** (CC BY-SA 4.0).

Podés **usarlo, adaptarlo y redistribuirlo**, incluso con fines comerciales,
siempre que:

- **cites la fuente** (atribución), y
- **distribuyas lo derivado bajo la misma licencia** (compartir igual).

El texto completo está en [`LICENSE`](LICENSE).

### Cómo citarlo

> Gómez, Lucas Daniel (2026). *Anexo Práctico de Inteligencia Artificial —
> Trayecto F, Tecnicatura Superior en IoT y Sistemas Embebidos*.
> ISFT N.º 238, Provincia de Buenos Aires. CC BY-SA 4.0.
> https://github.com/resourceldg/ia-iot-cuadernos

### Sobre lo que este repositorio incluye y lo que no

- El **código, los textos didácticos y los datos simulados** son originales y
  quedan cubiertos por la licencia de arriba.
- El cuaderno A-8 **descarga en tiempo de ejecución** el modelo
  `HuggingFaceTB/SmolLM2-135M-Instruct`, que tiene su propia licencia (Apache
  2.0) y **no** se distribuye acá.
- El repositorio **no incluye** los cuadernillos de los Trayectos F, G y H, ni el
  programa de la materia, ni ningún dato real de la cooperativa. El anexo los
  referencia, pero se entiende y se usa sin ellos.

---

## Aportes

Si lo usás en tu curso y encontrás un error, un número que no reproduce o una
explicación que confunde a los estudiantes, abrí un *issue*. Especialmente
valioso:

- Un resultado numérico que no coincide con lo documentado en
  `docente/notas_de_catedra.md` (indicá tus versiones de las librerías).
- Un ejercicio cuya verificación automática acepta una respuesta incorrecta.
- Una demostración que en tu aula no produjo el efecto pedagógico esperado.
