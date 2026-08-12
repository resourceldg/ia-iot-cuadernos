# %% [markdown]
# # Anexo A-2 · Ciclo de vida del dato: definir, validar y limpiar
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Módulos **F-1** y **F-2** del cuadernillo |
# | **Duración** | 150 minutos (se puede partir en dos clases) |
# | **Modalidad** | En equipo, sobre el proyecto propio |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Escribir el **diccionario de variables** de tu proyecto en código, y usarlo
#    para validar automáticamente cada lectura.
# 2. Detectar los **cuatro tipos de dato inválido** del módulo F-2 con funciones
#    reutilizables, sin mirar los datos a ojo.
# 3. Decidir **con criterio** qué hacer con cada dato inválido: descartar,
#    interpolar o marcar, y explicar por qué en cada caso.
# 4. Producir el **reporte de calidad** que el módulo F-5 te va a pedir para el
#    informe final ("cuántos datos descartaste y por qué motivo").
#
# ### Antes de arrancar
#
# Este es el cuaderno más importante del anexo. No porque el contenido sea el más
# difícil, sino porque **todo lo que viene después depende de que esto esté bien
# hecho**. Un modelo entrenado sobre datos sucios no falla con un error: falla
# dando un número que parece razonable y no lo es.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

DATOS = Path("..") / "datos"
if not DATOS.exists():
    DATOS = Path("datos")


def check(descripcion, condicion, pista=""):
    if condicion:
        print(f"  [OK]     {descripcion}")
    else:
        print(f"  [REVISAR] {descripcion}" + (f"\n            Pista: {pista}" if pista else ""))
    return bool(condicion)


silo = pd.read_csv(DATOS / "silobolsa_gas.csv", parse_dates=["timestamp"])
enchufe = pd.read_csv(DATOS / "enchufe_consumo.csv", parse_dates=["timestamp"])
riego = pd.read_csv(DATOS / "riego_humedad.csv", parse_dates=["timestamp"])

print(f"silobolsa {len(silo)} filas · enchufe {len(enchufe)} filas · riego {len(riego)} filas")

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · Un dato sin definición es solo un número (20 minutos)
#
# El módulo F-1 lo plantea así: *"voy a medir la temperatura"* **no es** una
# definición de dato. Una definición completa tiene cuatro campos:
#
# | Campo | Por qué es imprescindible |
# |---|---|
# | **Variable** | Qué fenómeno físico se mide. "Temperatura" no alcanza: ¿del aire, del grano, de la placa? |
# | **Unidad** | Sin unidad, un 25 puede ser °C, °F o un porcentaje. |
# | **Rango esperado** | Es lo que te permite detectar automáticamente un dato imposible. |
# | **Frecuencia** | Determina qué fenómenos podés ver y cuáles se te escapan. |
#
# ### Por qué la frecuencia decide qué podés ver
#
# Este punto casi siempre se subestima. Si muestreás cada 30 minutos, un pico de
# consumo que dura 4 minutos **no existe** en tus datos. No es que se vea mal: no
# está. Ningún modelo por sofisticado que sea va a recuperar lo que el muestreo
# tiró.
#
# Hay un resultado clásico de procesamiento de señales, el **teorema de muestreo
# de Nyquist-Shannon**, que dice que para reconstruir un fenómeno periódico hay
# que muestrear a más del doble de su frecuencia. En criollo, para tu proyecto:
# **si querés ver algo que dura X, muestreá al menos cada X/2.**
#
# Veámoslo, en lugar de creerlo.

# %%
# Un fenómeno real: un pico de consumo de unos 20 minutos de ancho, que
# arranca a las 13:17. El sensor podría medirlo perfecto... si lo mirara.
minutos = np.arange(0, 24 * 60)          # un día completo, minuto a minuto
consumo_real = 40 + 260 * np.exp(-((minutos - 797) ** 2) / (2 * 4.0 ** 2))
print(f"Pico real del fenómeno: {consumo_real.max():.0f} W\n")

# Para cada frecuencia probamos TODAS las fases posibles: el resultado depende
# de en qué minuto exacto arrancó el muestreo, y eso no lo controlás.
for cada in [1, 5, 15, 30, 60]:
    picos = [consumo_real[fase::cada].max() for fase in range(cada)]
    print(f"muestreando cada {cada:2d} min -> pico observado entre "
          f"{min(picos):5.0f} y {max(picos):5.0f} W  segun donde caigan las muestras")

# %% [markdown]
# Muestreando cada minuto, el pico se ve siempre. Muestreando cada 30 o 60
# minutos, **en el peor caso se reporta como 40 W**: el consumo de fondo, como si
# el evento nunca hubiera ocurrido. Y no hay ningún error de medición; el sensor
# midió bien cada vez que midió. Simplemente no estaba mirando cuando pasó lo
# interesante.
#
# Ahora fijate en lo más incómodo de la tabla: **la franja**. Con muestreo cada
# 15, 30 o 60 minutos, el mismo fenómeno puede verse como 300 W o como 40 W según
# en qué minuto exacto haya arrancado tu nodo. Eso no lo controlás: depende de
# cuándo se enchufó la placa o de cuándo se reinició por última vez.
#
# > **Un sistema de alerta cuyo resultado depende de en qué momento se encendió
# > la placa no es un sistema de alerta.** Por eso la regla práctica es
# > *muestreá al menos al doble de la frecuencia del fenómeno*, y no *justo a la
# > frecuencia*: el margen es lo que te saca la suerte de encima.
#
# > **Consecuencia para tu proyecto:** la frecuencia de muestreo no se elige por
# > comodidad ni por lo que aguanta la memoria de la placa. Se elige a partir de
# > la **duración del fenómeno más corto que te importa detectar**. Anotá esa
# > duración en tu diccionario de variables: es la justificación de la
# > frecuencia.

# %% [markdown]
# ### El diccionario de variables, en código
#
# En el cuadernillo el diccionario es una tabla en papel. Acá lo escribimos como
# un diccionario de Python, y eso lo vuelve **ejecutable**: la misma definición
# que documenta el proyecto sirve para validar los datos.

# %%
DICCIONARIO_SILOBOLSA = {
    "co2_ppm": {
        "descripcion": "Concentración de CO2 dentro de la bolsa",
        "unidad": "ppm",
        "minimo": 350.0,      # el aire libre ya tiene ~420 ppm
        "maximo": 1500.0,     # por encima, en este sensor, es precalentamiento
        "frecuencia_min": 30,
        "fenomeno_mas_corto": "un foco de fermentación tarda días en desarrollarse",
        "uso": "umbral de alerta e histórico de tendencia",
    },
    "temperatura_C": {
        "descripcion": "Temperatura del aire dentro de la bolsa",
        "unidad": "°C",
        "minimo": -5.0,
        "maximo": 55.0,
        "frecuencia_min": 30,
        "fenomeno_mas_corto": "ciclo día/noche",
        "uso": "contexto para interpretar el CO2",
    },
    "humedad_rel_pct": {
        "descripcion": "Humedad relativa del aire dentro de la bolsa",
        "unidad": "%",
        "minimo": 0.0,
        "maximo": 100.0,
        "frecuencia_min": 30,
        "fenomeno_mas_corto": "ciclo día/noche",
        "uso": "contexto para interpretar el CO2",
    },
}

pd.DataFrame(DICCIONARIO_SILOBOLSA).T

# %% [markdown]
# ---
# ## Bloque 2 — Teoría y práctica · Los cuatro tipos de dato inválido (40 minutos)
#
# El módulo F-2 define cuatro categorías. Para cada una escribimos una función
# que devuelve una **máscara booleana**: `True` en las filas problemáticas. Las
# máscaras se pueden combinar, contar y comparar, que es exactamente lo que
# necesitamos.

# %% [markdown]
# ### Tipo 1 — Fuera de rango físico
#
# El valor no es posible dado lo que se está midiendo. Un CO₂ negativo, una
# humedad de 180 %, una potencia menor que cero. Esta es la validación más
# barata y la que más problemas evita: **sale directamente del diccionario de
# variables**, sin ninguna estadística.

# %%
def fuera_de_rango(serie, minimo, maximo):
    """True donde el valor existe pero es físicamente imposible."""
    return serie.notna() & ((serie < minimo) | (serie > maximo))


rango_co2 = fuera_de_rango(silo["co2_ppm"],
                           DICCIONARIO_SILOBOLSA["co2_ppm"]["minimo"],
                           DICCIONARIO_SILOBOLSA["co2_ppm"]["maximo"])

print(f"Lecturas de CO2 fuera de rango físico: {rango_co2.sum()} de {len(silo)}")
print("\nAlgunos ejemplos:")
print(silo.loc[rango_co2, ["timestamp", "co2_ppm"]].head(8).to_string(index=False))

# %% [markdown]
# > **Ojo con esto:** hay dos poblaciones distintas mezcladas ahí. Los valores
# > negativos son un error del sensor o de la transmisión. Los valores de 2000 o
# > 3000 ppm son el **precalentamiento** del sensor MQ: después de cada reinicio
# > del nodo, el elemento calefactor tarda unas horas en estabilizarse y hasta
# > entonces la lectura no significa nada. Están documentados en el datasheet.
# >
# > Los dos casos se descartan, pero **no son el mismo problema** y no se
# > reportan igual: uno es una falla, el otro es una característica conocida del
# > hardware que debería estar contemplada en el firmware.

# %% [markdown]
# ### Tipo 2 — Faltante
#
# El sensor no respondió y quedó el hueco. En pandas eso es `NaN`.

# %%
def faltantes(serie):
    """True donde no hay valor."""
    return serie.isna()


for nombre, tabla, columna in [("silobolsa", silo, "co2_ppm"),
                               ("enchufe", enchufe, "potencia_W"),
                               ("riego", riego, "humedad_suelo_pct")]:
    m = faltantes(tabla[columna])
    print(f"{nombre:12s} {columna:20s} {m.sum():3d} faltantes ({m.mean() * 100:.2f} %)")

# %% [markdown]
# ### Tipo 3 — Repetido sospechoso
#
# El valor se congela: la misma lectura exacta muchas veces seguidas, cuando el
# fenómeno debería variar aunque sea un poco. Casi siempre significa que el
# sensor se desconectó y la librería devuelve el último valor en caché.
#
# La detección necesita **una decisión tuya**: ¿cuántas repeticiones seguidas son
# sospechosas? Eso depende del fenómeno. La temperatura de una silobolsa puede
# quedarse quieta un rato; un sensor de corriente, no.

# %%
def repetido_sospechoso(serie, minimo_repeticiones=8):
    """True donde el valor exacto se repite muchas veces seguidas.

    minimo_repeticiones se elige según el fenómeno, no según los datos.
    """
    # Cada vez que el valor cambia, empieza un "grupo" nuevo.
    grupo = (serie != serie.shift()).cumsum()
    largo_del_grupo = serie.groupby(grupo).transform("size")
    return serie.notna() & (largo_del_grupo >= minimo_repeticiones)


congelado_co2 = repetido_sospechoso(silo["co2_ppm"], minimo_repeticiones=8)
print(f"Lecturas congeladas: {congelado_co2.sum()}")

if congelado_co2.any():
    tramo = silo.loc[congelado_co2, ["timestamp", "co2_ppm"]]
    print(f"\nSe agrupan en {(congelado_co2 != congelado_co2.shift()).cumsum()[congelado_co2].nunique()} "
          f"tramos distintos. El primero:")
    print(tramo.head(6).to_string(index=False))

# %% [markdown]
# ### Tipo 4 — Fuera de tiempo
#
# La marca de tiempo no corresponde: el reloj de la placa se desconfiguró, o el
# nodo arrancó sin sincronizar contra NTP y quedó con la fecha de fábrica.
#
# La forma de detectarlo es preguntar si el tiempo **avanza siempre hacia
# adelante**. Si una fila tiene un timestamp anterior al de la fila previa, algo
# pasó.

# %%
def fuera_de_tiempo(timestamps):
    """True donde la marca de tiempo no avanza como corresponde."""
    salto = timestamps.diff()
    # El primer valor no tiene anterior: no se puede juzgar.
    return (salto <= pd.Timedelta(0)) & salto.notna()


tiempo_malo = fuera_de_tiempo(silo["timestamp"])
print(f"Marcas de tiempo fuera de orden: {tiempo_malo.sum()}")
print("\nMirá las fechas: son anteriores al inicio del proyecto.")
print(silo.loc[tiempo_malo, ["timestamp", "co2_ppm"]].to_string(index=False))

_ordenadas = silo["timestamp"].sort_values()
print(f"\nEl proyecto va del {_ordenadas.iloc[len(tiempo_malo[tiempo_malo]):].min().date()} "
      f"al {_ordenadas.max().date()}, pero la fecha más vieja del archivo es "
      f"{_ordenadas.min().date()}: más de un año antes de que existiera el nodo.")

# %% [markdown]
# ---
# ## Bloque 3 — Teoría · Qué hacer con lo que encontraste (25 minutos)
#
# Detectar es la mitad del trabajo. La otra mitad es decidir qué hacer, y ahí hay
# tres opciones, no una:
#
# | Estrategia | Cuándo corresponde | Riesgo |
# |---|---|---|
# | **Descartar** la fila | El dato es imposible y no se puede recuperar | Si descartás mucho, el resto puede quedar sesgado |
# | **Interpolar** el hueco | El hueco es corto y el fenómeno es continuo | Estás **inventando** datos: no puede usarse como verdad para evaluar |
# | **Marcar** y conservar | Querés dejar registro sin perder la fila | Hay que acordarse de filtrar después |
#
# ### La regla que no se puede violar
#
# > **Nunca interpoles la variable que vas a predecir.**
#
# Si vas a entrenar un modelo para anticipar el CO₂ y rellenás los huecos de CO₂
# interpolando, el modelo va a aprender a predecir tu interpolación —que es una
# línea recta— y va a parecer buenísimo. La métrica te va a dar hermosa y no vas
# a haber medido nada. Este error tiene nombre en la jerga: **fuga de datos**
# (*data leakage*), y lo vamos a ver otra vez, con otra cara, en el cuaderno A-4.
#
# Interpolar una variable de contexto (la temperatura, para tener con qué
# acompañar el análisis) es aceptable. Interpolar el objetivo, no.

# %%
def limpiar(tabla, columna, minimo, maximo, minimo_repeticiones=8,
            interpolar_huecos_hasta=2):
    """Aplica las cuatro validaciones y devuelve la tabla limpia + el reporte.

    interpolar_huecos_hasta: cantidad máxima de muestras consecutivas que se
    rellenan por interpolación. Los huecos más largos se descartan, porque
    inventar una recta de tres horas es inventar el fenómeno.
    """
    tabla = tabla.copy()
    serie = tabla[columna]

    marcas = {
        "fuera_de_rango": fuera_de_rango(serie, minimo, maximo),
        "faltante": faltantes(serie),
        "repetido_sospechoso": repetido_sospechoso(serie, minimo_repeticiones),
        "fuera_de_tiempo": fuera_de_tiempo(tabla["timestamp"]),
    }
    for nombre, mascara in marcas.items():
        tabla[f"invalido_{nombre}"] = mascara

    invalido = np.logical_or.reduce(list(marcas.values()))
    tabla["invalido"] = invalido

    # Los valores inválidos se anulan; después se decide qué hacer con el hueco.
    tabla.loc[invalido, columna] = np.nan

    # Interpolación solo de huecos cortos.
    tabla[columna] = tabla[columna].interpolate(limit=interpolar_huecos_hasta,
                                                limit_area="inside")
    tabla["interpolado"] = invalido & tabla[columna].notna()

    reporte = pd.DataFrame({
        "detectados": {k: int(v.sum()) for k, v in marcas.items()},
    })
    reporte["porcentaje"] = (reporte["detectados"] / len(tabla) * 100).round(2)
    reporte.loc["TOTAL inválidos"] = [int(invalido.sum()),
                                      round(invalido.mean() * 100, 2)]
    reporte.loc["recuperados por interpolación"] = [int(tabla["interpolado"].sum()),
                                                    round(tabla["interpolado"].mean() * 100, 2)]
    reporte.loc["descartados definitivamente"] = [int(tabla[columna].isna().sum()),
                                                  round(tabla[columna].isna().mean() * 100, 2)]
    return tabla, reporte


silo_limpio, reporte_silo = limpiar(
    silo, "co2_ppm",
    minimo=DICCIONARIO_SILOBOLSA["co2_ppm"]["minimo"],
    maximo=DICCIONARIO_SILOBOLSA["co2_ppm"]["maximo"],
)
reporte_silo

# %% [markdown]
# Ese reporte, tal cual está, es lo que el módulo **F-5** pide en la sección
# *"cómo se limpió"* del informe de datos: qué reglas aplicaste y cuántos datos
# descartaste.

# %%
# Comparación antes / después, en números.
antes = silo["co2_ppm"]
despues = silo_limpio["co2_ppm"]
comparacion = pd.DataFrame({
    "antes": antes.describe()[["count", "mean", "std", "min", "max"]],
    "después": despues.describe()[["count", "mean", "std", "min", "max"]],
}).round(1)
comparacion

# %% [markdown]
# ### Leé bien esta tabla, porque no dice lo que uno esperaría
#
# El **promedio casi no se movió** (unos 15 ppm sobre 545). Si hubieras mirado
# solo el promedio, habrías concluido que los datos sucios "no afectaban tanto".
#
# Lo que se derrumbó es otra cosa: el **máximo** pasó de más de 3000 ppm a menos
# de 1400, el **mínimo** dejó de ser negativo, y el **desvío estándar** se redujo
# a una fracción del original.
#
# Y eso sí cambia decisiones concretas. Supongamos que fijás el umbral de alerta
# como *"el 80 % del máximo histórico observado"*, que es una heurística común y
# razonable:

# %%
for etiqueta, serie in [("con datos sucios", antes), ("con datos limpios", despues)]:
    umbral = 0.8 * serie.max()
    disparos = int((despues > umbral).sum())
    print(f"{etiqueta:20s} umbral = {umbral:7.1f} ppm  ->  "
          f"alertaría {disparos:4d} veces en el histórico real")

# %% [markdown]
# Con los datos sucios, el umbral queda tan alto que **el sistema no alerta
# nunca**: los tres focos de fermentación pasan desapercibidos. Un puñado de
# lecturas de precalentamiento —el 1 % de los datos— alcanzó para inutilizar el
# sistema de alerta entero.
#
# > **La moraleja no es "los datos sucios corren el promedio".** Es que los datos
# > sucios corren los **extremos**, y muchas decisiones de ingeniería (umbrales,
# > escalas de gráficos, normalización de variables para un modelo) se toman
# > justamente a partir de los extremos.

# %% [markdown]
# ---
# ## Bloque 4 — Ejercicios

# %% [markdown]
# ### Ejercicio A2.1 [B] — El diccionario de tu proyecto
#
# Escribí `MI_DICCIONARIO` con **al menos dos variables** de tu proyecto, con
# todos los campos que tiene `DICCIONARIO_SILOBOLSA`. Si tu nodo todavía no mide,
# usá el proyecto de la cohorte que elegiste en A-0.
#
# El campo `fenomeno_mas_corto` no es decorativo: es la justificación de la
# frecuencia que elegiste. Escribilo en serio.

# %%
# TU CÓDIGO ACÁ
MI_DICCIONARIO = {
    # "nombre_variable": {
    #     "descripcion": "", "unidad": "", "minimo": 0.0, "maximo": 0.0,
    #     "frecuencia_min": 0, "fenomeno_mas_corto": "", "uso": "",
    # },
}

# %%
_campos = {"descripcion", "unidad", "minimo", "maximo", "frecuencia_min",
           "fenomeno_mas_corto", "uso"}
check("Definiste al menos dos variables", len(MI_DICCIONARIO) >= 2)
check("Todas tienen los siete campos",
      bool(MI_DICCIONARIO) and all(set(v) == _campos for v in MI_DICCIONARIO.values()),
      f"los campos son: {sorted(_campos)}")
check("Los rangos son coherentes (mínimo < máximo)",
      bool(MI_DICCIONARIO) and all(v["minimo"] < v["maximo"] for v in MI_DICCIONARIO.values()))
check("Justificaste la frecuencia con un fenómeno concreto",
      bool(MI_DICCIONARIO) and all(len(str(v["fenomeno_mas_corto"]).split()) >= 4
                                   for v in MI_DICCIONARIO.values()),
      "no alcanza con 'rápido' o 'lento': decí qué fenómeno y cuánto dura")

# %% [markdown]
# ### Ejercicio A2.2 [B] — Limpiar el enchufe
#
# Aplicá `limpiar()` sobre `enchufe`, columna `potencia_W`. El rango físico:
# la potencia no puede ser negativa y este enchufe está especificado hasta
# 2200 W. Guardá los resultados en `enchufe_limpio` y `reporte_enchufe`.

# %%
# TU CÓDIGO ACÁ
enchufe_limpio = None
reporte_enchufe = None

# %%
if isinstance(enchufe_limpio, pd.DataFrame) and isinstance(reporte_enchufe, pd.DataFrame):
    check("No quedaron potencias negativas",
          not (enchufe_limpio["potencia_W"] < 0).any())
    check("Se detectaron los faltantes originales",
          reporte_enchufe.loc["faltante", "detectados"] == 22)
    check("Se detectaron valores fuera de rango",
          reporte_enchufe.loc["fuera_de_rango", "detectados"] > 0)
    print()
    print(reporte_enchufe.to_string())
else:
    print("  [REVISAR] Todavía no llamaste a limpiar() sobre el enchufe.")

# %% [markdown]
# ### Ejercicio A2.3 [I] — Elegir el umbral de "congelado"
#
# El parámetro `minimo_repeticiones` no tiene un valor correcto universal:
# depende del fenómeno. Corré la detección sobre `riego`, columna
# `humedad_suelo_pct`, con distintos valores, y armá `sensibilidad`: un
# diccionario `{repeticiones: cantidad_detectada}` para 3, 5, 8, 12 y 20.
#
# Después contestá en la celda de texto: ¿cuál elegirías para tu proyecto y por
# qué? La respuesta tiene que mencionar **cuánto tiempo real** representan esas
# repeticiones dada tu frecuencia de muestreo.

# %%
# TU CÓDIGO ACÁ
sensibilidad = {}

# %%
check("Probaste los cinco valores", set(sensibilidad) == {3, 5, 8, 12, 20})
check("La cantidad detectada baja al exigir más repeticiones",
      len(sensibilidad) == 5 and
      list(sensibilidad.values()) == sorted(sensibilidad.values(), reverse=True),
      "cuanto más repeticiones exigís, menos casos deberían dar positivo")
if len(sensibilidad) == 5:
    print()
    for k in sorted(sensibilidad):
        print(f"   {k:2d} repeticiones seguidas -> {sensibilidad[k]:4d} lecturas marcadas "
              f"({k * 30 / 60:.1f} horas de sensor quieto)")

# %% [markdown]
# **Tu respuesta:** *(doble clic para editar)*
#
# Elegiría `minimo_repeticiones = ___` porque…

# %% [markdown]
# ### Ejercicio A2.4 [I] — Demostrar la fuga de datos
#
# Vamos a comprobar que interpolar el objetivo infla la métrica. El
# procedimiento:
#
# 1. Tomá `silo_limpio` y quedate solo con las filas donde `co2_ppm` no es nulo.
# 2. Escondé artificialmente el 20 % de los valores de `co2_ppm` (elegidos con
#    `rng = np.random.default_rng(0)`), guardando aparte los valores verdaderos.
# 3. Rellená esos huecos de dos maneras:
#    - `estimacion_interpolada`: con `.interpolate()`
#    - `estimacion_promedio`: con el promedio de la columna
# 4. Calculá el error absoluto medio de cada una contra los valores verdaderos y
#    guardalos en `error_interpolado` y `error_promedio`.
#
# Vas a ver que la interpolación da un error bajísimo. **Ese número es mentira**:
# es bajo porque los vecinos de cada hueco son casi iguales al valor escondido,
# no porque hayas aprendido nada del fenómeno.

# %%
# TU CÓDIGO ACÁ
error_interpolado = None
error_promedio = None

# %%
check("Calculaste los dos errores",
      error_interpolado is not None and error_promedio is not None)
if error_interpolado is not None and error_promedio is not None:
    check("La interpolación da un error mucho menor",
          error_interpolado < error_promedio / 3,
          "si no da eso, revisá que estés comparando contra los valores verdaderos")
    print(f"\n   error interpolando: {error_interpolado:7.2f} ppm")
    print(f"   error con promedio: {error_promedio:7.2f} ppm")
    print("\n   La interpolación 'gana' porque conoce a los vecinos del hueco.")
    print("   Un modelo real no tiene ese privilegio: predice sin ver el futuro.")

# %% [markdown]
# ### Ejercicio A2.5 [A] — El reporte de calidad de tu proyecto
#
# Sin verificación automática: es la entrega del módulo F-2.
#
# 1. Corré `limpiar()` sobre los datos de **tu** proyecto (o el de la cohorte que
#    elegiste) con los rangos de **tu** diccionario del A2.1.
# 2. Escribí abajo, en texto, el párrafo de "cómo se limpió" que va a ir en tu
#    informe F-5. Tiene que decir: qué reglas aplicaste, cuántas lecturas
#    descartó cada una, y **qué decidiste hacer con los huecos y por qué**.
# 3. Agregá una oración sobre el dato inválido que *no* pudiste detectar
#    automáticamente, si lo hay. (Ejemplo: un sensor mal calibrado da valores
#    perfectamente dentro del rango y perfectamente equivocados. Ninguna de las
#    cuatro reglas lo agarra.)

# %%
# TU CÓDIGO ACÁ


# %% [markdown]
# **Tu párrafo para el informe F-5:** *(doble clic para editar)*
#

# %% [markdown]
# ---
# ## Cierre del cuaderno A-2
#
# **Lo que quedó instalado en tu cabeza:**
#
# - La frecuencia de muestreo define qué fenómenos existen en tus datos. Lo que
#   el muestreo tira no se recupera después con ningún modelo.
# - El diccionario de variables no es papeleo: es la fuente de la validación
#   automática de rango.
# - Los cuatro tipos de dato inválido se detectan con máscaras booleanas, y cada
#   uno admite tres tratamientos posibles con consecuencias distintas.
# - Interpolar la variable objetivo produce métricas espectaculares y falsas.
#
# **Checklist de entrega**
#
# - [ ] `MI_DICCIONARIO` con al menos dos variables y la frecuencia justificada.
# - [ ] El reporte de calidad de tu conjunto, con los números por tipo de falla.
# - [ ] La decisión sobre `minimo_repeticiones` argumentada en tiempo real, no en
#       cantidad de muestras.
# - [ ] El párrafo "cómo se limpió" redactado, listo para pegar en el informe F-5.
#
# **Sigue en:** `A3_Exploracion_y_visualizacion.ipynb`
