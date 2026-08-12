# %% [markdown]
# # Anexo A-5 · Reglas contra modelo: la comparación honesta
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Módulos **G-2** y **G-3** · cierra lo abierto en A-1 |
# | **Duración** | 120 minutos |
# | **Modalidad** | En equipo, sobre el proyecto propio |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Medir tu sistema de reglas y tu modelo **con la misma métrica y el mismo
#    conjunto de prueba**, que es la única forma de que la comparación signifique
#    algo.
# 2. Entender que un umbral no es un número sino una **familia de sistemas**, y
#    elegir dentro de esa familia según el costo de cada error.
# 3. Usar el modelo como **herramienta de descubrimiento** para escribir una regla
#    mejor, y desplegar esa regla en un microcontrolador que no podría correr el
#    modelo.
# 4. Escribir la decisión fundamentada que pide el módulo G-3, incluido el caso
#    —perfectamente válido— de decidir que todavía no corresponde un modelo.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, precision_recall_curve)

from estilo_grafico import aplicar_estilo, titular, SERIE, ESTADO, TINTA_SUAVE, TINTA_APAGADA

aplicar_estilo()
DATOS = Path("..") / "datos"
if not DATOS.exists():
    DATOS = Path("datos")


def check(descripcion, condicion, pista=""):
    if condicion:
        print(f"  [OK]     {descripcion}")
    else:
        print(f"  [REVISAR] {descripcion}" + (f"\n            Pista: {pista}" if pista else ""))
    return bool(condicion)


# Preparación idéntica a la de A-4, para que la comparación sea sobre lo mismo.
silo = pd.read_csv(DATOS / "silobolsa_gas.csv", parse_dates=["timestamp"])
s = silo["co2_ppm"]
g = (s != s.shift()).cumsum()
silo.loc[(s < 350) | (s > 1500) | s.isna()
         | (s.groupby(g).transform("size") >= 8)
         | (silo["timestamp"].diff() <= pd.Timedelta(0)), "co2_ppm"] = np.nan
silo = silo.sort_values("timestamp").reset_index(drop=True)
silo["co2_media_6h"] = silo["co2_ppm"].rolling(12, min_periods=6).mean()
silo["co2_pendiente_6h"] = silo["co2_ppm"] - silo["co2_ppm"].shift(12)
silo["co2_pendiente_24h"] = silo["co2_ppm"] - silo["co2_ppm"].shift(48)

CARACTERISTICAS = ["co2_ppm", "co2_media_6h", "co2_pendiente_6h",
                   "co2_pendiente_24h", "temperatura_C"]
datos = silo.dropna(subset=CARACTERISTICAS + ["riesgo_24h"]).reset_index(drop=True)

corte = int(len(datos) * 0.7)
entrenamiento = datos.iloc[:corte]
prueba = datos.iloc[corte:].reset_index(drop=True)
y_prueba = prueba["riesgo_24h"]

print(f"Entrenamiento: {len(entrenamiento)} ejemplos "
      f"({entrenamiento['timestamp'].iloc[0].date()} a "
      f"{entrenamiento['timestamp'].iloc[-1].date()})")
print(f"Prueba:        {len(prueba)} ejemplos "
      f"({prueba['timestamp'].iloc[0].date()} a "
      f"{prueba['timestamp'].iloc[-1].date()})")
print(f"Positivos en prueba: {y_prueba.sum()} ({y_prueba.mean() * 100:.1f} %)")

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · La pregunta de G-2, formulada bien (20 minutos)
#
# El cuadernillo del Trayecto G lo plantea con precisión: la pregunta **no** es
# *"¿le pongo IA?"*. Es:
#
# > *¿Mi problema tiene un patrón que valga la pena que un modelo aprenda, en
# > lugar de que yo lo defina a mano?*
#
# Y esa pregunta se contesta con cinco criterios, no con entusiasmo:
#
# | Criterio | Favorece **reglas** | Favorece **modelo** |
# |---|---|---|
# | **Cantidad de datos** | pocos días, pocos eventos | meses de historia, muchos eventos |
# | **Conocimiento del dominio** | un experto sabe el umbral | nadie sabe bien dónde está el corte |
# | **Cantidad de variables** | una o dos | varias que interactúan entre sí |
# | **Explicabilidad exigida** | hay que justificar cada alerta | alcanza con que funcione |
# | **Dónde corre** | microcontrolador, sin internet | servidor o PC |
#
# La última fila pesa muchísimo en esta carrera y casi nunca se menciona en los
# cursos de IA: **un ESP32 no va a correr scikit-learn**. Volvemos sobre esto al
# final del cuaderno, y no como un detalle.
#
# ### La condición que hace válida cualquier comparación
#
# Los dos sistemas se miden **con la misma métrica, sobre el mismo conjunto de
# prueba, que ninguno de los dos vio**. Suena obvio y se viola todo el tiempo:
# es muy común ajustar el umbral de la regla mirando los datos de prueba y
# después compararla contra un modelo que no tuvo esa ventaja.

# %% [markdown]
# ---
# ## Bloque 2 — Práctica · Un umbral no es un número, es una familia (30 minutos)
#
# La regla más simple para la silobolsa es *"alertar si el CO₂ supera U"*. Pero
# `U` no está dado: elegirlo **es** el diseño del sistema. Y cada `U` produce un
# sistema con un compromiso distinto entre alertas falsas y episodios perdidos.
#
# Lo que sigue barre todos los umbrales posibles y mide los tres números en cada
# uno. **Ojo con el detalle metodológico:** el umbral se elige mirando el
# conjunto de entrenamiento, y recién después se evalúa en el de prueba. Si lo
# eligiéramos mirando la prueba, estaríamos haciendo lo mismo que criticamos.

# %%
def evaluar(prediccion, verdad):
    return {
        "precisión": precision_score(verdad, prediccion, zero_division=0),
        "sensibilidad": recall_score(verdad, prediccion, zero_division=0),
        "F1": f1_score(verdad, prediccion, zero_division=0),
        "alertas emitidas": int(prediccion.sum()),
    }


umbrales = np.arange(500, 1200, 10)
barrido = pd.DataFrame([
    {"umbral": u,
     **evaluar((entrenamiento["co2_ppm"] > u).astype(int).to_numpy(),
               entrenamiento["riesgo_24h"])}
    for u in umbrales
]).set_index("umbral")

umbral_elegido = int(barrido["F1"].idxmax())
print(f"Umbral que maximiza F1 EN ENTRENAMIENTO: {umbral_elegido} ppm")
print(barrido.loc[umbral_elegido - 20:umbral_elegido + 20].round(3).to_string())

# %%
fig, ax = plt.subplots()
ax.plot(barrido.index, barrido["precisión"], color=SERIE[0], label="precisión")
ax.plot(barrido.index, barrido["sensibilidad"], color=SERIE[1], label="sensibilidad")
ax.plot(barrido.index, barrido["F1"], color=SERIE[2], linewidth=2.6, label="F1")
ax.axvline(umbral_elegido, color=TINTA_APAGADA, linestyle="--", linewidth=1.2)
ax.text(umbral_elegido + 10, 0.06, f"umbral elegido: {umbral_elegido} ppm",
        color=TINTA_SUAVE, fontsize=9)
ax.set_xlabel("umbral de alerta (ppm de CO₂)")
ax.set_ylabel("valor de la métrica")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left")
titular(ax, "¿Qué umbral conviene para la regla?",
        "Bajarlo detecta más episodios y dispara más alertas falsas. No hay un valor «correcto» sin declarar cuál error duele más.")
plt.show()

# %% [markdown]
# ### Ese gráfico es el corazón del módulo G-3
#
# Mirá lo que dice: **no hay un umbral óptimo en abstracto**. Con el umbral bajo,
# la sensibilidad es 1 (no se te escapa ningún foco) y la precisión es mala
# (alertás muchas veces al pedo). Con el umbral alto pasa exactamente al revés.
#
# El F1 marca un compromiso entre los dos, pero el F1 **supone que los dos errores
# cuestan lo mismo**, y en la silobolsa no cuestan lo mismo ni cerca: una alerta
# falsa cuesta un viaje al campo, y un foco no detectado cuesta la carga.
#
# > Por eso la pregunta *"¿cuál es el mejor umbral?"* no tiene respuesta técnica.
# > La respuesta sale de la conversación con quien se come el costo del error.

# %% [markdown]
# ---
# ## Bloque 3 — Práctica · Los tres sistemas, misma métrica, mismo conjunto (25 minutos)
#
# Ahora sí: el sistema de reglas simple, una regla de dos condiciones, y el árbol
# de A-4. Todos evaluados sobre el conjunto de prueba, que ninguno vio.

# %%
arbol = DecisionTreeClassifier(max_depth=3, random_state=238)
arbol.fit(entrenamiento[CARACTERISTICAS], entrenamiento["riesgo_24h"])

sistemas = {
    "Regla 1 · umbral de nivel":
        (prueba["co2_ppm"] > umbral_elegido).astype(int).to_numpy(),
    "Regla 2 · nivel + tendencia":
        ((prueba["co2_ppm"] > 800) & (prueba["co2_pendiente_24h"] > 100)).astype(int).to_numpy(),
    "Árbol de decisión":
        arbol.predict(prueba[CARACTERISTICAS]),
}

tabla = pd.DataFrame({nombre: evaluar(pred, y_prueba)
                      for nombre, pred in sistemas.items()}).round(3)
tabla

# %% [markdown]
# ### Cómo se lee esta tabla
#
# Hay tres cosas para sacar de ahí, y ninguna es "ganó el modelo".
#
# **Primero: la Regla 2 es peor que la Regla 1.** Le puse dos condiciones y dos
# números elegidos a ojo (800 ppm y 100 ppm de pendiente), y perdió contra el
# umbral simple que salió de barrer todos los valores. Agregarle condiciones a
# una regla no la mejora sola: **los parámetros hay que buscarlos, no
# adivinarlos.** Una regla con parámetros barridos es un modelo entrenado, aunque
# no lo llamemos así.
#
# **Segundo: el árbol gana, pero fijate en cuánto.** Le saca unas pocas
# centésimas de F1 a la mejor regla. Esa diferencia tiene que pagar **todo** lo
# que el modelo trae de arrastre: entrenarlo, guardarlo, versionarlo, reentrenarlo
# cuando cambie la bolsa, y necesitar una computadora que lo pueda correr.
#
# **Tercero: mirá dónde gana.** El árbol tiene bastante más sensibilidad y algo
# menos de precisión: emite más alertas y por eso no se le escapa casi nada. Si
# tu costo del falso negativo es alto —y en una silobolsa lo es— eso es
# exactamente el intercambio que querés.
#
# > **Un modelo que le gana a la regla por poco no le gana a la regla.** El
# > criterio no es el número solo: es el número contra el costo de sostener el
# > sistema durante toda su vida útil.

# %% [markdown]
# ---
# ## Bloque 4 — La jugada que sirve de verdad en sistemas embebidos (25 minutos)
#
# Acá está la idea más útil de todo el anexo para un técnico en IoT.
#
# El árbol de A-4 encontró algo que vos no sabías: que la pendiente de 24 horas
# separa los focos reales de los arranques falsos. Ese conocimiento **no está
# atrapado dentro del modelo**: está escrito, en castellano, en las reglas que
# imprimimos.
#
# Entonces: usá el modelo para **descubrir** la regla, y después desplegá la
# regla, no el modelo.

# %%
print("Lo que el árbol descubrió, en su idioma:")
print("=" * 62)
print(export_text(arbol, feature_names=CARACTERISTICAS, decimals=0))

# %% [markdown]
# Traducido a algo que entra en un ESP32. **No es una transcripción literal:**
# tomé la estructura del árbol y le hice dos podas de sentido común. Ignoré las
# ramas que terminan en dos hojas de la misma clase (no deciden nada) y le agregué
# un piso de nivel a la rama de la pendiente, porque una subida fuerte partiendo
# de 400 ppm no es un foco: es la bolsa despertándose a la mañana.

# %% [markdown]
# ```c
# // Regla derivada del árbol entrenado sobre 90 días de datos.
# // No necesita librerías, ni punto flotante de 64 bits, ni memoria dinámica.
# bool hay_riesgo(float co2_actual, float co2_hace_24h) {
#     float pendiente = co2_actual - co2_hace_24h;
#     if (co2_actual > 872.0) {
#         return pendiente > -281.0;      // alto y no viniendo en bajada
#     }
#     return pendiente > 171.0 && co2_actual > 750.0;  // todavía bajo, pero trepando
# }
# ```
#
# Comparemos esa función —tal cual, con esos números— contra el árbol que la
# originó:

# %%
def hay_riesgo(co2_actual, co2_hace_24h):
    """La versión C de arriba, en Python, para poder medirla."""
    pendiente = co2_actual - co2_hace_24h
    if co2_actual > 872.0:
        return pendiente > -281.0
    return pendiente > 171.0 and co2_actual > 750.0


regla_embebida = np.array([
    int(hay_riesgo(fila["co2_ppm"], fila["co2_ppm"] - fila["co2_pendiente_24h"]))
    for _, fila in prueba.iterrows()
])

comparacion_final = pd.DataFrame({
    "Árbol (en la PC)": evaluar(sistemas["Árbol de decisión"], y_prueba),
    "Regla derivada (en el ESP32)": evaluar(regla_embebida, y_prueba),
}).round(3)
comparacion_final

# %% [markdown]
# ### Parate acá, porque el resultado no es el que uno esperaría
#
# La regla escrita a mano **no empata con el árbol: le gana**. Misma
# sensibilidad, mejor precisión, menos alertas emitidas.
#
# Eso no es un error ni un golpe de suerte, y tiene una explicación que vale más
# que el resultado: el árbol usó una de sus ramas para ajustarse a detalles del
# conjunto de entrenamiento que no se repiten después. Al podar esa rama y
# reemplazarla por una condición con sentido físico —*"para que sea un foco, además
# de subir tiene que estar arriba de cierto nivel"*— el sistema generaliza mejor.
#
# > **Meterle conocimiento del dominio a un modelo aprendido no es hacer trampa:
# > es la forma más barata de regularizarlo.** El algoritmo solo vio números; vos
# > sabés además cómo se comporta una silobolsa. Esa información no está en el
# > dataset y es gratis.
#
# Cuidado con el entusiasmo, igual: esta comparación se hizo sobre **un solo
# conjunto de prueba con tres episodios**. La diferencia es chica y bien podría
# darse vuelta con otros datos. Lo correcto es reportarla como lo que es —una
# observación sobre este conjunto— y no como una ley.

# %% [markdown]
# ### Por qué esto importa tanto en esta carrera
#
# La versión en C, además de andar igual o mejor:
#
# - corre en un micro de 240 MHz sin sistema operativo;
# - no necesita conexión: decide en el nodo, aunque se caiga internet;
# - se puede leer, auditar y discutir con el productor;
# - no se "desactualiza" en silencio.
#
# > **El aprendizaje automático no siempre termina en un modelo desplegado.**
# > Muchas veces su mejor uso es encontrar la regla que después escribís a mano.
# > Eso no es hacer trampa ni "no usar IA": es usarla donde aporta —descubrir el
# > patrón— y no donde estorba —el nodo.
#
# Cuando la regla derivada **no** alcanza a igualar al modelo, ahí sí tenés un
# argumento sólido para desplegar el modelo: pudiste mostrar que la simplicidad
# cuesta rendimiento, y cuánto.

# %% [markdown]
# ---
# ## Bloque 5 — Ejercicios

# %% [markdown]
# ### Ejercicio A5.1 [B] — Los cinco criterios, para tu proyecto
#
# Completá `criterios` con `"reglas"` o `"modelo"` según hacia dónde empuja cada
# criterio **en tu proyecto**. Está perfecto que queden mezclados: casi siempre
# quedan.

# %%
# TU CÓDIGO ACÁ
criterios = {
    "cantidad_de_datos": "",
    "conocimiento_del_dominio": "",
    "cantidad_de_variables": "",
    "explicabilidad_exigida": "",
    "donde_corre": "",
}
justificacion_donde_corre = ""

# %%
check("Completaste los cinco criterios",
      all(v.strip().lower() in {"reglas", "modelo"} for v in criterios.values()),
      "cada valor tiene que ser exactamente 'reglas' o 'modelo'")
check("Justificaste dónde va a correr el sistema",
      len(justificacion_donde_corre.split()) >= 12,
      "decí en qué hardware corre y si tiene o no conexión permanente")
if all(v.strip().lower() in {"reglas", "modelo"} for v in criterios.values()):
    _votos = pd.Series([v.lower() for v in criterios.values()]).value_counts()
    print(f"\n   Tus criterios votan: {_votos.to_dict()}")
    print("   Si quedó parejo, la decisión la define el criterio que más pese en")
    print("   tu contexto, no la mayoría simple. Escribilo en el A5.4.")

# %% [markdown]
# ### Ejercicio A5.2 [I] — Elegí el umbral por costo, no por F1
#
# El F1 supone que los dos errores cuestan lo mismo. Rompé ese supuesto.
#
# Definí una función `costo_total(prediccion, verdad, costo_fp, costo_fn)` que
# devuelva el costo total del sistema, y usala para elegir el umbral que
# **minimiza el costo** con `costo_fp = 1` (un viaje al campo al pedo) y
# `costo_fn = 30` (perder parte de la carga).
#
# Guardá el resultado en `umbral_por_costo`.

# %%
# TU CÓDIGO ACÁ
def costo_total(prediccion, verdad, costo_fp, costo_fn):
    pass


umbral_por_costo = None

# %%
try:
    _fp = costo_total(np.array([1, 0, 0]), np.array([0, 0, 0]), 1, 30)
    _fn = costo_total(np.array([0, 0, 0]), np.array([1, 0, 0]), 1, 30)
    check("costo_total cuenta bien un falso positivo", _fp == 1)
    check("costo_total cuenta bien un falso negativo", _fn == 30)
except Exception as e:
    print(f"  [REVISAR] costo_total todavía no funciona: {type(e).__name__}: {e}")

check("Elegiste un umbral por costo", umbral_por_costo is not None)
if umbral_por_costo is not None:
    print(f"\n   Umbral que maximiza F1:        {umbral_elegido} ppm")
    print(f"   Umbral que minimiza el costo:  {umbral_por_costo} ppm")
    check("El umbral por costo es MENOR que el de F1",
          umbral_por_costo < umbral_elegido,
          "si perder la carga cuesta 30 veces más, conviene alertar antes: el umbral baja")

# %% [markdown]
# ### Ejercicio A5.3 [I] — Tu regla derivada
#
# Entrená un árbol sobre **tu** proyecto (o el de la cohorte que elegiste),
# imprimí sus reglas y escribí a mano la función `mi_regla_embebida(...)` que las
# reproduzca sin usar scikit-learn.
#
# Después medí las dos con la misma métrica y guardá la diferencia de F1 en
# `perdida_por_simplificar`.

# %%
# TU CÓDIGO ACÁ
def mi_regla_embebida():
    pass


perdida_por_simplificar = None

# %%
check("Calculaste cuánto F1 perdés al simplificar",
      perdida_por_simplificar is not None)
if perdida_por_simplificar is not None:
    if perdida_por_simplificar < 0.05:
        print(f"\n   Perdés {perdida_por_simplificar:.3f} de F1 al pasar a la regla.")
        print("   Es poco: tenés argumento sólido para desplegar la regla en el nodo.")
    else:
        print(f"\n   Perdés {perdida_por_simplificar:.3f} de F1 al pasar a la regla.")
        print("   Es bastante: tenés argumento para justificar el modelo, o para")
        print("   escribir una regla con una condición más.")

# %% [markdown]
# ### Ejercicio A5.4 [A] — La decisión fundamentada (entrega de G-3)
#
# Sin verificación automática. Este texto es una de las piezas del portafolio
# final. Escribilo como si lo fuera a leer alguien de la cooperativa, no el
# profesor.
#
# Tiene que responder, en no más de una carilla:
#
# 1. **Qué decidí:** reglas, modelo, o prueba de concepto sin decidir todavía.
# 2. **Con qué métrica lo medí** y por qué esa (remitite al costo de cada error,
#    como en A4.4).
# 3. **Los números de los dos enfoques**, sobre el mismo conjunto de prueba.
# 4. **Qué NO puedo afirmar** con los datos que tengo. Por ejemplo: si tu conjunto
#    de prueba contiene un solo episodio, no podés afirmar nada sobre la
#    generalización; podés afirmar que funcionó en ese episodio.
# 5. **Qué haría falta** para poder decidir mejor: ¿más meses de datos? ¿más
#    episodios? ¿un sensor adicional?
#
# > Recordá lo que dice el cuadernillo de G-2: *"decir «todavía no tengo
# > suficientes datos para decidir» es una respuesta técnicamente válida si está
# > fundamentada. Lo que no es válido es no responder nada."*

# %% [markdown]
# **Tu decisión fundamentada:** *(doble clic para editar)*
#
# **1. Qué decidí:**
#
# **2. Métrica y por qué:**
#
# **3. Números de los dos enfoques:**
#
# **4. Qué no puedo afirmar:**
#
# **5. Qué haría falta:**

# %% [markdown]
# ---
# ## Cierre del cuaderno A-5
#
# **Lo que quedó instalado en tu cabeza:**
#
# - Una comparación solo vale si los dos sistemas se miden con la misma métrica
#   sobre el mismo conjunto que ninguno vio.
# - Un umbral es una familia de sistemas. Elegir dentro de esa familia es una
#   decisión de costos, no de estadística.
# - Un modelo que le gana a la regla por poco, no le gana: hay que descontarle el
#   costo de sostenerlo.
# - El mejor uso del aprendizaje automático en un proyecto embebido suele ser
#   **descubrir la regla** que después se escribe a mano en el firmware.
#
# **Checklist de entrega**
#
# - [ ] Los cinco criterios evaluados para tu proyecto (A5.1).
# - [ ] El umbral elegido por costo, con los costos declarados (A5.2).
# - [ ] Tu regla derivada y cuánto rendimiento perdés al simplificar (A5.3).
# - [ ] La decisión fundamentada de G-3, con los cinco puntos (A5.4).
#
# **Sigue en:** `A6_Red_neuronal_desde_cero.ipynb` — donde abrimos la caja y
# programamos una red neuronal con NumPy, sin ningún framework.
