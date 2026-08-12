# %% [markdown]
# # Anexo A-0 · El entorno de trabajo y las tres herramientas
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Tecnicatura Superior en IoT y Sistemas Embebidos
# Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Previo a F-0. Se hace una sola vez, al principio del cuatrimestre. |
# | **Duración** | 90 minutos |
# | **Modalidad** | Individual, cada uno en su máquina |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Verificar que tu máquina tiene todo lo necesario para el resto del anexo.
# 2. Manejar un array de NumPy y un DataFrame de pandas: las dos estructuras
#    sobre las que se apoya absolutamente todo lo que sigue.
# 3. Cargar los datos de los tres proyectos de la cohorte y responder preguntas
#    simples sobre ellos.
#
# ### Antes de arrancar
#
# Este anexo no reemplaza al cuadernillo del Trayecto F: lo acompaña. El
# cuadernillo te dice **qué** tenés que decidir sobre tu proyecto; el anexo te
# da el **cómo**, con código que corre de verdad en tu computadora. Ningún
# ejercicio de acá necesita internet ni un servicio en la nube: todo se ejecuta
# local, incluso los modelos de los cuadernos A-6 y A-7.

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · Por qué un cuaderno y no un `.py` (10 minutos)
#
# Un archivo `.py` se ejecuta de arriba a abajo y termina. Un cuaderno de
# Jupyter mantiene el estado vivo entre celdas: cargás los datos una vez y
# después probás veinte ideas distintas sobre esos mismos datos sin volver a
# cargarlos. Para trabajo exploratorio con datos —que es exactamente lo que
# hace un científico de datos— esa diferencia es enorme.
#
# La contracara: **el orden en que ejecutás las celdas importa, y no siempre es
# el orden en que están escritas.** Si algo se comporta raro, andá al menú
# *Kernel → Restart & Run All*. Es el equivalente a apagar y prender, y
# resuelve el 90 % de los problemas de un cuaderno.
#
# Tres atajos que conviene aprender ahora:
#
# | Atajo | Qué hace |
# |---|---|
# | `Shift + Enter` | Ejecuta la celda y pasa a la siguiente |
# | `Ctrl + Enter` | Ejecuta la celda y se queda ahí |
# | `Esc` y después `B` | Inserta una celda nueva abajo |

# %% [markdown]
# ---
# ## Bloque 2 — Práctica · Verificación del entorno (10 minutos)
#
# Ejecutá la celda que sigue. Si todo está bien, vas a ver una lista de tildes
# verdes. Si aparece alguna cruz, revisá el `README.md` del anexo: la sección
# de instalación explica cómo resolver cada caso.

# %%
import sys
import platform

print(f"Python   {sys.version.split()[0]}")
print(f"Sistema  {platform.system()} {platform.machine()}")
print()

faltan = []
for nombre, minimo in [("numpy", "1.24"), ("pandas", "2.0"),
                       ("matplotlib", "3.7"), ("sklearn", "1.3")]:
    try:
        modulo = __import__(nombre)
        print(f"  OK   {nombre:12s} {modulo.__version__}")
    except ImportError:
        print(f"  --   {nombre:12s} NO INSTALADO (mínimo sugerido {minimo})")
        faltan.append(nombre)

print()
if faltan:
    print("Faltan paquetes:", ", ".join(faltan))
    print("Corré en la terminal:  pip install -r requirements.txt")
else:
    print("Entorno completo. Podés seguir.")

# %% [markdown]
# Esta segunda celda define un ayudante que vas a ver en todos los cuadernos
# del anexo: `check()`. Sirve para que puedas verificar vos mismo si un
# ejercicio te salió, sin esperar a la corrección. **No es una nota**: es una
# forma de que no avances sobre una base equivocada.

# %%
def check(descripcion, condicion, pista=""):
    """Verifica una condición de un ejercicio e informa el resultado."""
    if condicion:
        print(f"  [OK]     {descripcion}")
    else:
        mensaje = f"  [REVISAR] {descripcion}"
        if pista:
            mensaje += f"\n            Pista: {pista}"
        print(mensaje)
    return bool(condicion)


check("El ayudante check() funciona", 2 + 2 == 4)
check("Ejemplo de algo que falla", 2 + 2 == 5, "esto es a propósito, para que veas cómo se ve")

# %% [markdown]
# ---
# ## Bloque 3 — Teoría · NumPy, o cómo se piensa con muchos números a la vez (20 minutos)
#
# Un sensor no produce *un* número: produce miles. Si tu nodo mide cada 30
# segundos, en un día junta 2880 lecturas. Escribir un `for` para recorrer esas
# 2880 lecturas funciona, pero es lento y, sobre todo, se lee mal.
#
# NumPy propone otra cosa: **operar sobre el array entero de una sola vez**. La
# idea se llama *vectorización* y es la base de todo el cálculo numérico
# moderno, incluidas las redes neuronales.

# %%
import numpy as np

# Diez lecturas de un sensor de temperatura, en grados Celsius.
lecturas = np.array([21.4, 22.1, 21.9, 23.5, 24.8, 26.2, 25.1, 24.0, 22.8, 21.6])

print("Lecturas:      ", lecturas)
print("Cantidad:      ", lecturas.size)
print("Promedio:      ", lecturas.mean().round(2))
print("Máximo:        ", lecturas.max())
print("Desvío estándar:", lecturas.std().round(2))

# %% [markdown]
# La operación se aplica a todos los elementos sin escribir un solo `for`:

# %%
# Pasar de Celsius a Fahrenheit: una línea, diez conversiones.
en_fahrenheit = lecturas * 9 / 5 + 32
print("En Fahrenheit:", en_fahrenheit.round(1))

# Diferencia entre cada lectura y la anterior: cuánto cambió el fenómeno.
cambios = np.diff(lecturas)
print("Cambio entre lecturas:", cambios.round(2))

# %% [markdown]
# ### Máscaras booleanas: la herramienta más útil de todas
#
# Comparar un array con un número no devuelve `True` o `False`: devuelve **un
# array de `True` y `False`, uno por elemento**. Eso se llama máscara, y sirve
# para filtrar, contar y marcar. Es exactamente lo que vas a usar en el
# cuaderno A-2 para detectar datos inválidos.

# %%
mascara = lecturas > 24.0
print("La máscara:        ", mascara)
print("Cuántas superan 24:", mascara.sum())        # True vale 1, False vale 0
print("Cuáles son:        ", lecturas[mascara])    # filtrar con la máscara
print("Porcentaje:        ", f"{mascara.mean() * 100:.1f} %")

# Y se pueden combinar con & (y), | (o), ~ (no). Ojo con los paréntesis.
templada = (lecturas > 22.0) & (lecturas < 25.0)
print("Entre 22 y 25:     ", lecturas[templada])

# %% [markdown]
# > **Error clásico:** escribir `lecturas > 22 and lecturas < 25`. En NumPy hay
# > que usar `&`, `|` y `~`, y encerrar cada comparación entre paréntesis. Con
# > `and` vas a recibir el error *"truth value of an array is ambiguous"*.

# %% [markdown]
# ---
# ## Bloque 4 — Teoría · pandas, la planilla con memoria (20 minutos)
#
# NumPy maneja números; pandas maneja **tablas con nombres de columna y con
# tiempo**. Un `DataFrame` es, conceptualmente, una planilla de cálculo que se
# programa. Para telemetría de IoT —donde cada fila es un instante y cada
# columna una variable— es la estructura natural.

# %%
import pandas as pd

registro = pd.DataFrame({
    "timestamp": pd.date_range("2026-08-17 08:00", periods=6, freq="30min"),
    "temperatura_C": [21.4, 22.1, 21.9, 23.5, 24.8, 26.2],
    "humedad_pct": [58.0, 57.2, 57.8, 55.1, 52.4, 49.9],
})

registro

# %%
# Una columna sola se llama Serie y se comporta casi como un array de NumPy.
print(registro["temperatura_C"].mean().round(2))

# Filas que cumplen una condición: misma idea de máscara que en NumPy.
print(registro[registro["temperatura_C"] > 23.0])

# Columna nueva calculada a partir de las existentes.
registro["indice_calor"] = registro["temperatura_C"] + 0.05 * registro["humedad_pct"]
print(registro[["timestamp", "indice_calor"]].round(2))

# %% [markdown]
# ### El tiempo como índice
#
# Cuando el `timestamp` pasa a ser el índice de la tabla, pandas habilita
# operaciones que en NumPy costarían mucho trabajo: agrupar por hora, por día,
# calcular promedios móviles, rellenar huecos. Todo el cuaderno A-3 vive de
# esto.

# %%
por_tiempo = registro.set_index("timestamp")

# Promedio por hora del reloj: 'h' agrupa en ventanas de una hora.
print(por_tiempo["temperatura_C"].resample("h").mean().round(2))

print()
# Promedio móvil de 3 muestras: suaviza el ruido del sensor.
print(por_tiempo["temperatura_C"].rolling(3).mean().round(2))

# %% [markdown]
# ---
# ## Bloque 5 — Práctica · Los datos de la cohorte (20 minutos)
#
# El anexo trae tres conjuntos de datos simulados, uno por cada proyecto de la
# cohorte 2026. **Están sucios a propósito**: tienen los cuatro tipos de dato
# inválido que define el módulo F-2. Por ahora solo los vamos a mirar; en el
# cuaderno A-2 los vamos a limpiar.

# %%
from pathlib import Path

DATOS = Path("..") / "datos"
if not DATOS.exists():
    DATOS = Path("datos")   # por si abriste el cuaderno desde la raíz del anexo

enchufe = pd.read_csv(DATOS / "enchufe_consumo.csv", parse_dates=["timestamp"])
riego = pd.read_csv(DATOS / "riego_humedad.csv", parse_dates=["timestamp"])
silo = pd.read_csv(DATOS / "silobolsa_gas.csv", parse_dates=["timestamp"])

for nombre, tabla in [("Enchufe inteligente", enchufe),
                      ("Riego automático", riego),
                      ("Monitoreo de silobolsa", silo)]:
    print(f"{nombre:24s} {len(tabla):5d} filas × {len(tabla.columns)} columnas")
    print(f"{'':24s} columnas: {', '.join(tabla.columns)}")
    print()

# %%
# Las primeras filas del proyecto de silobolsa.
silo.head()

# %%
# .describe() es el primer vistazo obligatorio a cualquier dato nuevo.
# Mirá con atención los mínimos y los máximos: ahí ya se ve que algo anda mal.
silo.describe().round(2)

# %% [markdown]
# > **Parate acá un momento.** En la tabla de arriba, el `co2_ppm` tiene un
# > mínimo negativo y un máximo de más de 3000 ppm. Ninguno de los dos valores
# > es físicamente posible en una silobolsa. Todavía no los vamos a arreglar,
# > pero registrá el hallazgo: **mirar los extremos antes que el promedio** es
# > un hábito que te va a ahorrar horas.

# %% [markdown]
# ---
# ## Bloque 6 — Ejercicios
#
# Resolvé cada ejercicio en la celda que dice `# TU CÓDIGO ACÁ` y después
# ejecutá la celda de verificación que viene abajo.

# %% [markdown]
# ### Ejercicio A0.1 [B] — Estadísticos básicos
#
# Calculá, sobre el conjunto del **riego automático**:
#
# - `humedad_promedio`: el promedio de `humedad_suelo_pct`
# - `temp_maxima`: la temperatura máxima registrada
# - `cantidad_riegos`: cuántas veces se activó la bomba (columna `bomba_activa`,
#   vale 1 cuando riega)

# %%
# TU CÓDIGO ACÁ
humedad_promedio = None
temp_maxima = None
cantidad_riegos = None

# %%
check("humedad_promedio calculado", humedad_promedio is not None and abs(humedad_promedio - 52.03) < 0.5,
      "usá riego['humedad_suelo_pct'].mean()")
check("temp_maxima calculada", temp_maxima is not None and abs(temp_maxima - 29.53) < 0.1,
      "usá .max() sobre la columna de temperatura")
check("cantidad_riegos calculada", cantidad_riegos == 25,
      "la columna vale 1 cuando riega: sumarla cuenta los riegos")

# %% [markdown]
# ### Ejercicio A0.2 [B] — Filtrar con una máscara
#
# Del conjunto del **enchufe**, obtené un DataFrame llamado `consumo_alto` con
# todas las filas donde `potencia_W` sea mayor a 300. Después guardá en
# `filas_consumo_alto` la cantidad de filas que quedaron.

# %%
# TU CÓDIGO ACÁ
consumo_alto = None
filas_consumo_alto = None

# %%
check("consumo_alto es un DataFrame", isinstance(consumo_alto, pd.DataFrame))
check("el filtro es correcto",
      isinstance(consumo_alto, pd.DataFrame) and len(consumo_alto) > 0
      and bool((consumo_alto["potencia_W"] > 300).all()),
      "enchufe[enchufe['potencia_W'] > 300]")
check("filas_consumo_alto coincide",
      isinstance(consumo_alto, pd.DataFrame) and filas_consumo_alto == len(consumo_alto))

# %% [markdown]
# ### Ejercicio A0.3 [I] — Agrupar por hora del día
#
# ¿A qué hora del día consume más el enchufe? Construí una Serie llamada
# `consumo_por_hora` con el promedio de `potencia_W` para cada hora del reloj
# (0 a 23), y guardá en `hora_pico` la hora con el promedio más alto.
#
# *Pista:* `enchufe["timestamp"].dt.hour` te da la hora de cada fila. Después
# `.groupby(...)` agrupa y `.mean()` promedia. Para la hora del máximo,
# `.idxmax()`.

# %%
# TU CÓDIGO ACÁ
consumo_por_hora = None
hora_pico = None

# %%
check("consumo_por_hora tiene 24 valores",
      consumo_por_hora is not None and len(consumo_por_hora) == 24,
      "agrupá por .dt.hour, que va de 0 a 23")
check("hora_pico está en el pico de la noche",
      hora_pico in (20, 21, 22),
      "el generador puso el pico de uso alrededor de las 21")

# %% [markdown]
# ### Ejercicio A0.4 [I] — Contar lo que no se puede medir
#
# En el conjunto de la **silobolsa**, contá cuántas lecturas de `co2_ppm` son
# físicamente imposibles. Definimos imposible como: **menor a 300 ppm** (el aire
# libre ya tiene ~420 ppm, así que menos de 300 dentro de una bolsa cerrada no
# tiene sentido) **o mayor a 1500 ppm** (por encima de eso, en este sensor, la
# lectura corresponde al precalentamiento y no al gas real).
#
# Guardá el resultado en `lecturas_imposibles`.

# %%
# TU CÓDIGO ACÁ
lecturas_imposibles = None

# %%
_esperado = int(((silo["co2_ppm"] < 300) | (silo["co2_ppm"] > 1500)).sum())
if check("lecturas_imposibles calculado", lecturas_imposibles == _esperado,
         "combiná las dos condiciones con | y sumá la máscara; ojo con los paréntesis"):
    print(f"\n  De {len(silo)} lecturas, {_esperado} son imposibles "
          f"({_esperado / len(silo) * 100:.1f} %).")
    print("  Eso lo vamos a trabajar en serio en el cuaderno A-2.")

# %% [markdown]
# ### Ejercicio A0.5 [A] — Tu propio proyecto
#
# Este ejercicio no tiene verificación automática porque la respuesta es
# distinta para cada equipo.
#
# 1. Si tu nodo ya generó datos, exportalos a un `.csv` con una columna
#    `timestamp` y una columna por variable, y cargalos acá con `pd.read_csv`.
# 2. Si todavía no generó datos, elegí el conjunto de la cohorte que más se
#    parezca a tu proyecto y trabajá sobre ese durante todo el anexo.
# 3. Corré `.describe()` sobre tus datos y anotá en una celda de texto: ¿algún
#    mínimo o máximo te llama la atención? ¿por qué?

# %%
# TU CÓDIGO ACÁ
# mis_datos = pd.read_csv("ruta/a/mis_datos.csv", parse_dates=["timestamp"])
# mis_datos.describe()

# %% [markdown]
# ---
# ## Cierre del cuaderno A-0
#
# **Lo que quedó instalado en tu cabeza:**
#
# - Un array de NumPy se opera entero, sin `for`. Las máscaras booleanas
#   filtran, cuentan y marcan.
# - Un DataFrame de pandas es una tabla con nombres y con tiempo. `.describe()`
#   es el primer vistazo obligatorio; los extremos dicen más que el promedio.
# - Los tres conjuntos de la cohorte están cargados y ya sabés que tienen
#   problemas.
#
# **Checklist de entrega**
#
# - [ ] Los cinco ejercicios ejecutados, con las verificaciones en `[OK]`
#       (el A0.5 no tiene verificación: se entrega la celda de texto).
# - [ ] El cuaderno guardado con las salidas visibles.
# - [ ] En el A0.5, escribiste cuál de los tres conjuntos vas a usar durante el
#       anexo, o adjuntaste los datos de tu propio nodo.
#
# **Sigue en:** `A1_Agentes_y_reglas.ipynb` — donde tu nodo se convierte
# formalmente en un agente y escribimos el primer sistema de decisión.
