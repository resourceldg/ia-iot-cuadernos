# %% [markdown]
# # Anexo A-4 · Tu primer modelo: clasificación, regresión y métricas
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Bloque 4 del programa anual · módulos **G-2** y **G-3** |
# | **Duración** | 150 minutos (conviene partirlo en dos clases) |
# | **Modalidad** | En equipo, sobre el proyecto propio |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Distinguir **clasificación** de **regresión** con un ejemplo propio de cada
#    una, y armar la matriz `X`, `y` que cualquier modelo necesita.
# 2. Partir los datos en entrenamiento y prueba **respetando el tiempo**, y
#    explicar por qué partirlos al azar arruina el experimento.
# 3. Entrenar un árbol de decisión y **leer las reglas que aprendió**.
# 4. Interpretar `accuracy`, `precision` y `recall`, y demostrar por qué la
#    primera engaña cuando las clases están desbalanceadas.
# 5. Comparar siempre contra un **baseline trivial**, que es lo que separa un
#    resultado de un número.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error)

from estilo_grafico import aplicar_estilo, titular, SERIE, ESTADO, SECUENCIAL, TINTA_SUAVE

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


print("Listo.")

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · Qué es aprender, en concreto (25 minutos)
#
# En **aprendizaje supervisado** le mostramos al programa muchos ejemplos
# resueltos y le pedimos que encuentre solo la regla que los explica. Los
# ejemplos resueltos son dos cosas:
#
# | Símbolo | Nombre | Qué es |
# |---|---|---|
# | `X` | características, *features* | Una tabla: una fila por ejemplo, una columna por variable de entrada |
# | `y` | objetivo, *target* | Una columna: la respuesta correcta para cada fila de `X` |
#
# **El tipo de `y` decide el tipo de problema:**
#
# | Si `y` es… | El problema es… | Ejemplo de esta cohorte |
# |---|---|---|
# | una categoría (sí/no, A/B/C) | **clasificación** | ¿va a haber riesgo de fermentación en 24 h? |
# | un número continuo | **regresión** | ¿cuánta humedad va a haber en 3 horas? |
#
# Esa es toda la distinción que pide el Bloque 4 del programa. No es sutil: mirá
# la columna `y` y listo.
#
# ### Lo que hay que construir antes: las características
#
# Los datos crudos casi nunca sirven tal cual. Una lectura suelta de CO₂ dice
# poco; lo que dice algo es **el CO₂ comparado con cómo venía**. Construir esas
# columnas se llama *ingeniería de características*, y en la práctica es donde se
# gana o se pierde el problema — mucho más que en la elección del modelo.
#
# Para la silobolsa vamos a construir cinco:
#
# | Característica | Por qué |
# |---|---|
# | `co2_ppm` | el nivel actual |
# | `co2_media_6h` | el nivel reciente, sin el ruido de una lectura suelta |
# | `co2_pendiente_6h` | cuánto cambió en las últimas 6 horas |
# | `co2_pendiente_24h` | **cuánto cambió en el último día**: la tendencia de fondo |
# | `temperatura_C` | la fermentación calienta, así que acompaña |
#
# ### Por qué este problema no se resuelve con un umbral y listo
#
# En estos datos hay **tres focos** que se descontrolan y cruzan el umbral, pero
# también **cuatro arranques falsos**: el CO₂ sube hasta 800 o 850 ppm y después
# retrocede solo, sin llegar a nada. Eso es lo que pasa en una silobolsa real
# cuando hay unos días de calor.
#
# La consecuencia es concreta: cuando el sensor marca 820 ppm, **el nivel actual
# no alcanza para saber si esto es un foco o un susto**. Hay que mirar de dónde
# viene. Por eso la pendiente de 24 horas está entre las características, y por
# eso el ejercicio tiene sentido.

# %%
silo = pd.read_csv(DATOS / "silobolsa_gas.csv", parse_dates=["timestamp"])

# Limpieza mínima (la de A-2, condensada).
serie = silo["co2_ppm"]
grupo = (serie != serie.shift()).cumsum()
invalido = ((serie < 350) | (serie > 1500) | serie.isna()
            | (serie.groupby(grupo).transform("size") >= 8)
            | (silo["timestamp"].diff() <= pd.Timedelta(0)))
silo.loc[invalido, "co2_ppm"] = np.nan
silo = silo.sort_values("timestamp").reset_index(drop=True)

# Ingeniería de características. 12 muestras = 6 horas (una cada 30 min).
silo["co2_media_6h"] = silo["co2_ppm"].rolling(12, min_periods=6).mean()
silo["co2_pendiente_6h"] = silo["co2_ppm"] - silo["co2_ppm"].shift(12)
silo["co2_pendiente_24h"] = silo["co2_ppm"] - silo["co2_ppm"].shift(48)

CARACTERISTICAS = ["co2_ppm", "co2_media_6h", "co2_pendiente_6h",
                   "co2_pendiente_24h", "temperatura_C"]
OBJETIVO = "riesgo_24h"

datos = silo.dropna(subset=CARACTERISTICAS + [OBJETIVO]).reset_index(drop=True)
print(f"{len(datos)} ejemplos utilizables de {len(silo)} filas originales")
print(f"\nDistribución del objetivo:")
print(datos[OBJETIVO].value_counts().rename({0: "sin riesgo", 1: "riesgo"}).to_string())
print(f"\nSolo el {datos[OBJETIVO].mean() * 100:.1f} % de los casos son positivos. "
      f"Acordate de este número.")

# %% [markdown]
# ---
# ## Bloque 2 — Teoría · La partición que casi todos hacen mal (20 minutos)
#
# Para saber si un modelo aprendió algo, hay que evaluarlo sobre datos que
# **no vio durante el entrenamiento**. Si lo evaluás sobre los mismos datos con
# los que lo entrenaste, estás tomándole examen con las respuestas a la vista.
#
# La forma habitual de partir es al azar: 80 % para entrenar, 20 % para probar.
# **Con series temporales eso está mal**, y es un error tan común que tiene
# nombre: *fuga temporal*.
#
# Pensalo así: si mezclás al azar, el conjunto de prueba va a contener la lectura
# de las 14:00 mientras que el de entrenamiento tiene la de las 13:30 y la de
# las 14:30 del mismo día. El modelo prácticamente vio la respuesta. Va a dar una
# métrica excelente y va a fracasar en producción, donde el futuro **no está
# disponible**.
#
# La partición correcta para datos con tiempo es **por corte temporal**: entrenás
# con el pasado, evaluás con el futuro. Exactamente como va a funcionar el
# sistema cuando lo instales.
#
# Vamos a hacer las dos y comparar.

# %%
X = datos[CARACTERISTICAS]
y = datos[OBJETIVO]

# --- Partición CORRECTA: por tiempo ---
corte = int(len(datos) * 0.7)
X_ent, X_prueba = X.iloc[:corte], X.iloc[corte:]
y_ent, y_prueba = y.iloc[:corte], y.iloc[corte:]

print(f"Entrenamiento: {datos['timestamp'].iloc[0].date()} a "
      f"{datos['timestamp'].iloc[corte - 1].date()}  ({len(X_ent)} ejemplos)")
print(f"Prueba:        {datos['timestamp'].iloc[corte].date()} a "
      f"{datos['timestamp'].iloc[-1].date()}  ({len(X_prueba)} ejemplos)")

# --- Partición INCORRECTA: al azar ---
from sklearn.model_selection import train_test_split
Xa_ent, Xa_prueba, ya_ent, ya_prueba = train_test_split(
    X, y, test_size=0.3, random_state=238, shuffle=True)

modelo_temporal = DecisionTreeClassifier(max_depth=4, random_state=238).fit(X_ent, y_ent)
modelo_azaroso = DecisionTreeClassifier(max_depth=4, random_state=238).fit(Xa_ent, ya_ent)

print(f"\nExactitud con partición AL AZAR (mal):  "
      f"{accuracy_score(ya_prueba, modelo_azaroso.predict(Xa_prueba)):.3f}")
print(f"Exactitud con partición TEMPORAL (bien): "
      f"{accuracy_score(y_prueba, modelo_temporal.predict(X_prueba)):.3f}")

# %% [markdown]
# ### Y acá viene la parte que no conviene barrer abajo de la alfombra
#
# En **este** problema los dos números dan casi lo mismo. La partición al azar no
# infló nada apreciable.
#
# Es tentador concluir "entonces daba igual". No da igual, y vale la pena
# entender por qué salió parecido acá: el CO₂ de la silobolsa es un fenómeno
# **lento**, y las características que construimos (nivel, media, pendiente)
# describen el estado del sistema más que el instante puntual. Un modelo que
# aprende "CO₂ alto y subiendo hace un día ⇒ riesgo" no gana nada por haber visto
# la muestra vecina.
#
# **En el problema de regresión del Bloque 5, con los mismos criterios, la
# partición al azar sí va a dar un error notoriamente más optimista que el
# real.** Mismo código, mismo cuidado, resultado distinto.
#
# > Esa es justamente la razón de la regla: **no podés saber de antemano cuánto te
# > va a mentir la partición al azar en tu problema.** Si el protocolo correcto
# > cuesta una línea de código y el incorrecto puede inflarte el resultado sin
# > avisarte, la decisión no tiene mucho misterio.
#
# De acá en adelante usamos siempre la partición temporal.

# %% [markdown]
# ---
# ## Bloque 3 — Práctica · El árbol de decisión y sus reglas (30 minutos)
#
# Un **árbol de decisión** aprende una serie de preguntas del tipo *"¿la variable
# tal es mayor que tanto?"*, encadenadas. Es decir: aprende exactamente la misma
# clase de objeto que en el cuaderno A-1 escribimos a mano.
#
# Esa es la razón por la que empezamos por acá y no por una red neuronal: **al
# árbol se le pueden leer las reglas**, y compararlas con las tuyas.

# %%
arbol = DecisionTreeClassifier(max_depth=3, random_state=238)
arbol.fit(X_ent, y_ent)

print("REGLAS QUE APRENDIÓ EL ÁRBOL")
print("=" * 60)
print(export_text(arbol, feature_names=CARACTERISTICAS, decimals=0))
print("Nota: algunas ramas se parten en dos hojas de la MISMA clase. No es un")
print("error: el árbol sigue separando para dejar cada hoja más pura, aunque la")
print("decisión final no cambie. Para leer las reglas, esas ramas se ignoran.")

# %%
importancias = pd.Series(arbol.feature_importances_, index=CARACTERISTICAS).sort_values()

fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(importancias.index, importancias.values, color=SERIE[0], height=0.62)
ax.set_xlabel("importancia relativa")
ax.grid(axis="x")
titular(ax, "¿En qué se apoya el árbol para decidir?",
        f"La variable dominante es «{importancias.idxmax()}».")
plt.show()

# %% [markdown]
# **Leelas de verdad, no las pases por arriba.** Esas reglas son una hipótesis
# sobre el fenómeno, escrita por el algoritmo, y se puede discutir con lo que uno
# sabe de silobolsas.
#
# Mirá qué encontró: primero corta por el **nivel** de CO₂, y después —para los
# casos que quedan en el medio— corta por la **pendiente de 24 horas**. Es decir,
# aprendió solo la distinción que le costó plata a más de un productor: *un CO₂
# de 820 ppm que viene subiendo no es lo mismo que un CO₂ de 820 ppm que viene
# bajando*. Esa segunda rama es exactamente la que separa los focos reales de los
# cuatro arranques falsos.
#
# Nadie le dijo eso. Salió de los datos.
#
# > **El contraejemplo también importa.** Si el árbol se apoyara en algo sin
# > sentido físico —la hora del día para predecir fermentación, digamos—, eso no
# > sería un descubrimiento: sería la señal de que hay una fuga de datos en tus
# > características. Leer las reglas es tu principal herramienta de control de
# > calidad.

# %% [markdown]
# ---
# ## Bloque 4 — Teoría · Por qué la exactitud engaña (30 minutos)
#
# Acordate del número que te pedí guardar: **solo el 10 % de los casos son
# positivos**. Con clases así de desbalanceadas, la exactitud (*accuracy*) deja
# de significar lo que parece.
#
# Comparemos el árbol contra el modelo más tonto que existe: **decir siempre que
# no**. Un modelo que no mira los datos.

# %%
prediccion_arbol = arbol.predict(X_prueba)
prediccion_tonta = np.zeros(len(y_prueba), dtype=int)   # siempre "sin riesgo"

resultados = pd.DataFrame({
    nombre: {
        "exactitud (accuracy)": accuracy_score(y_prueba, pred),
        "precisión (precision)": precision_score(y_prueba, pred, zero_division=0),
        "sensibilidad (recall)": recall_score(y_prueba, pred, zero_division=0),
        "F1": f1_score(y_prueba, pred, zero_division=0),
    }
    for nombre, pred in [("Árbol de decisión", prediccion_arbol),
                         ("«Siempre digo que no»", prediccion_tonta)]
}).round(3)
resultados

# %% [markdown]
# ### Qué acaba de pasar
#
# El modelo que **no mira los datos** consigue una exactitud altísima. Si tu
# informe dijera solamente *"el modelo alcanzó 90 % de exactitud"*, estarías
# reportando algo que un `return 0` también consigue.
#
# Pero su **sensibilidad es 0**: de todos los episodios de fermentación reales,
# no detecta ninguno. Que es, exactamente, lo único que el sistema tenía que
# hacer.
#
# ### Las tres métricas, en criollo
#
# Todo sale de contar los cuatro casos posibles:
#
# | | El modelo dice **riesgo** | El modelo dice **sin riesgo** |
# |---|---|---|
# | **Hubo riesgo** | Verdadero positivo (VP) | Falso negativo (FN) — *se te pasó* |
# | **No hubo riesgo** | Falso positivo (FP) — *alerta al pedo* | Verdadero negativo (VN) |
#
# | Métrica | Fórmula | La pregunta que contesta |
# |---|---|---|
# | **Exactitud** | (VP+VN) / total | ¿Qué proporción de casos acerté? *(inútil si hay desbalance)* |
# | **Precisión** | VP / (VP+FP) | De las veces que alerté, ¿cuántas eran de verdad? |
# | **Sensibilidad** | VP / (VP+FN) | De los problemas que hubo, ¿cuántos detecté? |
# | **F1** | media armónica de las dos | Un solo número cuando las dos importan parejo |

# %%
matriz = confusion_matrix(y_prueba, prediccion_arbol)

fig, ax = plt.subplots(figsize=(5.4, 4.4))
ax.imshow(matriz, cmap="Blues", vmin=0)
etiquetas = ["sin riesgo", "riesgo"]
ax.set_xticks([0, 1], [f"dice\n«{e}»" for e in etiquetas])
ax.set_yticks([0, 1], [f"hubo\n{e}" for e in etiquetas])
ax.grid(False)
nombres = [["VN", "FP (alerta al pedo)"], ["FN (se pasó)", "VP"]]
for i in range(2):
    for j in range(2):
        color = "white" if matriz[i, j] > matriz.max() / 2 else TINTA_SUAVE
        ax.text(j, i, f"{matriz[i, j]}\n{nombres[i][j]}", ha="center", va="center",
                color=color, fontsize=10)
titular(ax, "Matriz de confusión del árbol")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Cuál de las dos priorizar es una decisión de ingeniería, no de estadística
#
# No existe "la métrica correcta" en abstracto. Depende del costo de cada error
# **en tu proyecto**:
#
# | Proyecto | Error más caro | Métrica a priorizar |
# |---|---|---|
# | **Silobolsa** | No detectar una fermentación: se pierde el grano | **Sensibilidad** (recall) |
# | **Enchufe** | Cortar la heladera por una falsa alarma | **Precisión** |
# | **Riego** | Regar de más ahoga la planta; regar de menos la seca | Las dos, o sea **F1** |
#
# En la silobolsa, una alerta falsa cuesta que alguien vaya a mirar la bolsa al
# pedo. Una alerta que no salta cuesta la carga entera. **No son comparables**, y
# por eso ahí se prioriza sensibilidad aunque baje la precisión.
#
# Ese razonamiento, escrito para tu proyecto, es lo que pide el módulo G-3 cuando
# dice *"elegí una métrica simple y aplicala"*. La elección hay que justificarla
# con el costo del error, no con cuál da más lindo.

# %% [markdown]
# ---
# ## Bloque 5 — Práctica · Regresión y el baseline que hay que vencer (25 minutos)
#
# Ahora un problema de regresión: **predecir la humedad del suelo dentro de 3
# horas**. La `y` pasa a ser un número continuo.
#
# Y acá aparece la lección más importante del cuaderno. Antes de festejar
# cualquier resultado, hay que compararlo contra el **baseline de persistencia**:
# predecir que dentro de 3 horas la humedad va a ser **exactamente la misma que
# ahora**. Es el modelo más tonto posible para una serie temporal, y en
# fenómenos lentos es sorprendentemente difícil de superar.

# %%
riego = pd.read_csv(DATOS / "riego_humedad.csv", parse_dates=["timestamp"])
riego.loc[(riego["humedad_suelo_pct"] < 0) | (riego["humedad_suelo_pct"] > 100),
          "humedad_suelo_pct"] = np.nan

riego["hora"] = riego["timestamp"].dt.hour
riego["humedad_pendiente_3h"] = riego["humedad_suelo_pct"] - riego["humedad_suelo_pct"].shift(6)
# El objetivo: la humedad 6 muestras (3 horas) en el futuro.
riego["humedad_futura_3h"] = riego["humedad_suelo_pct"].shift(-6)

CARACT_R = ["humedad_suelo_pct", "temperatura_C", "hora", "humedad_pendiente_3h"]
datos_r = riego.dropna(subset=CARACT_R + ["humedad_futura_3h"]).reset_index(drop=True)

corte_r = int(len(datos_r) * 0.7)
Xr_ent, Xr_prueba = datos_r[CARACT_R].iloc[:corte_r], datos_r[CARACT_R].iloc[corte_r:]
yr_ent, yr_prueba = datos_r["humedad_futura_3h"].iloc[:corte_r], datos_r["humedad_futura_3h"].iloc[corte_r:]

modelos = {
    "Baseline: persistencia": Xr_prueba["humedad_suelo_pct"].to_numpy(),
    "Regresión lineal": LinearRegression().fit(Xr_ent, yr_ent).predict(Xr_prueba),
    "Árbol de regresión": DecisionTreeRegressor(max_depth=5, random_state=238)
                          .fit(Xr_ent, yr_ent).predict(Xr_prueba),
}

comparacion = pd.DataFrame({
    nombre: {
        "MAE (error absoluto medio, %)": mean_absolute_error(yr_prueba, pred),
        "RMSE (%)": np.sqrt(mean_squared_error(yr_prueba, pred)),
    }
    for nombre, pred in modelos.items()
}).round(3)
comparacion

# %% [markdown]
# ### Leé la tabla antes de seguir, porque hay dos sorpresas
#
# **Primera: la regresión lineal pierde contra el baseline.** Un modelo entrenado,
# con cuatro características, anda *peor* que repetir el último valor conocido.
# Eso no es un error de código: es el resultado. La humedad de suelo se comporta
# como una sierra —baja despacio, salta de golpe cuando riega— y una recta no
# puede representar eso. Si hubieras entrenado solo la regresión lineal y no
# hubieras calculado el baseline, habrías reportado un MAE de 5 % como un logro.
#
# **Segunda: el árbol sí le gana**, y por bastante. La diferencia entre los dos
# modelos no está en cuál es "más avanzado", sino en cuál puede representar la
# forma del fenómeno.
#
# > **MAE o RMSE, ¿cuál?** El MAE es el error promedio en las unidades de la
# > variable: se explica en una oración a cualquiera. El RMSE castiga más los
# > errores grandes, porque los eleva al cuadrado. Si en tu proyecto un error
# > grande es mucho peor que varios chicos (por ejemplo, ahogar la planta),
# > mirá el RMSE. Si todos los errores duelen parejo, MAE.

# %% [markdown]
# ### Acá sí: la partición al azar mintiendo
#
# Este es el problema donde se ve lo que en el Bloque 2 quedó pendiente. Mismo
# modelo, mismos datos, misma métrica: lo único que cambia es cómo se parte.

# %%
Xz_ent, Xz_prueba, yz_ent, yz_prueba = train_test_split(
    datos_r[CARACT_R], datos_r["humedad_futura_3h"], test_size=0.3,
    random_state=238, shuffle=True)

mae_temporal = mean_absolute_error(
    yr_prueba, DecisionTreeRegressor(max_depth=5, random_state=238)
    .fit(Xr_ent, yr_ent).predict(Xr_prueba))
mae_azaroso = mean_absolute_error(
    yz_prueba, DecisionTreeRegressor(max_depth=5, random_state=238)
    .fit(Xz_ent, yz_ent).predict(Xz_prueba))

print(f"MAE con partición TEMPORAL (bien): {mae_temporal:.3f} %")
print(f"MAE con partición AL AZAR (mal):   {mae_azaroso:.3f} %")
print(f"\nLa partición al azar reporta un error "
      f"{(1 - mae_azaroso / mae_temporal) * 100:.0f} % menor del real.")

# %% [markdown]
# Ahí está. **La partición al azar te regala una porción de error que no
# existe**,
# porque pone en el conjunto de prueba lecturas cuyos vecinos inmediatos —de
# media hora antes y media hora después— están en el de entrenamiento. El modelo
# no está prediciendo: está interpolando entre datos que ya vio.
#
# Cuando el nodo esté colgado del cantero, no va a tener el dato de media hora
# **después**. Por eso la única evaluación que significa algo es la temporal.

# %%
fig, ax = plt.subplots()
eje_tiempo = datos_r["timestamp"].iloc[corte_r:corte_r + 300]
ax.plot(eje_tiempo, yr_prueba.iloc[:300], color=SERIE[0], linewidth=2.2,
        label="humedad real dentro de 3 h")
ax.plot(eje_tiempo, modelos["Regresión lineal"][:300], color=SERIE[1],
        linewidth=1.6, label="predicción de la regresión lineal")
ax.set_ylabel("humedad de suelo (%)")
ax.legend(loc="upper right")
titular(ax, "¿La predicción sigue al fenómeno o va siempre atrás?")
plt.show()

# %% [markdown]
# ### Sobreajuste: cuando el modelo memoriza en lugar de aprender
#
# Un árbol sin límite de profundidad puede memorizar cada ejemplo de
# entrenamiento. Va a dar un error de entrenamiento casi nulo y va a andar peor
# en datos nuevos. Eso es **sobreajuste** (*overfitting*), y se ve al ojo cuando
# graficás las dos curvas de error juntas.

# %%
profundidades = range(1, 21)
curva = []
for p in profundidades:
    modelo = DecisionTreeRegressor(max_depth=p, random_state=238).fit(Xr_ent, yr_ent)
    curva.append({
        "profundidad": p,
        "error en entrenamiento": mean_absolute_error(yr_ent, modelo.predict(Xr_ent)),
        "error en prueba": mean_absolute_error(yr_prueba, modelo.predict(Xr_prueba)),
    })
curva = pd.DataFrame(curva).set_index("profundidad")
mejor = int(curva["error en prueba"].idxmin())

fig, ax = plt.subplots()
ax.plot(curva.index, curva["error en entrenamiento"], color=SERIE[0],
        label="error en entrenamiento (datos que vio)")
ax.plot(curva.index, curva["error en prueba"], color=SERIE[1],
        label="error en prueba (datos nuevos)")
ax.axvline(mejor, color=ESTADO["grave"], linestyle="--", linewidth=1.4)
ax.text(mejor + 0.3, curva["error en prueba"].max() * 0.85,
        f"mejor profundidad: {mejor}", color=ESTADO["grave"], fontsize=9.5)
ax.set_xlabel("profundidad máxima del árbol")
ax.set_ylabel("MAE (%)")
ax.set_xticks(range(0, 21, 2))
ax.legend()
titular(ax, "¿Hasta dónde conviene dejar crecer el árbol?",
        "A partir de cierta profundidad el modelo mejora en lo que vio y empeora en lo nuevo.")
plt.show()

# %% [markdown]
# Las dos curvas separándose **es** el sobreajuste, dibujado. La curva azul sigue
# bajando porque el árbol memoriza mejor; la naranja se despega porque esa
# memoria no sirve para nada nuevo.
#
# > Un modelo más complejo no es un modelo mejor. La complejidad se elige mirando
# > el error **en datos que el modelo no vio**, nunca el de entrenamiento.

# %% [markdown]
# ---
# ## Bloque 6 — Ejercicios

# %% [markdown]
# ### Ejercicio A4.1 [B] — Clasificación o regresión
#
# Para cada problema, guardá `"clasificacion"` o `"regresion"` en el diccionario.

# %%
# TU CÓDIGO ACÁ
tipos = {
    "¿La bomba tiene que encenderse en la próxima hora?": "",
    "¿Cuántos watts va a consumir el enchufe mañana a las 21?": "",
    "¿El sensor está funcionando bien, regular o mal?": "",
    "¿Cuántos días faltan para que el CO2 supere el umbral?": "",
    "¿Este mensaje MQTT vino de mi nodo o de otro?": "",
}

# %%
_correctas = ["clasificacion", "regresion", "clasificacion", "regresion", "clasificacion"]
_dadas = [v.strip().lower() for v in tipos.values()]
check("Las cinco están bien clasificadas", _dadas == _correctas,
      "mirá el tipo de la respuesta: ¿es una categoría o un número?")
for (pregunta, dada), correcta in zip(tipos.items(), _correctas):
    if dada != correcta:
        print(f"      revisá: {pregunta}")

# %% [markdown]
# ### Ejercicio A4.2 [I] — Un modelo para el enchufe
#
# Entrená un clasificador que detecte el **compresor pegado** (columna
# `compresor_pegado` del conjunto del enchufe).
#
# Pasos:
# 1. Cargá `enchufe_consumo.csv` y limpiá `potencia_W` (rango 0 a 2200).
# 2. Construí al menos **tres características**, incluyendo una de ventana móvil
#    (por ejemplo el mínimo de las últimas 4 horas: si el compresor no corta, ese
#    mínimo nunca baja).
# 3. Partí **por tiempo**, 70 / 30.
# 4. Entrená un `DecisionTreeClassifier` con `max_depth=4`.
# 5. Guardá en `metricas_enchufe` un diccionario con `accuracy`, `precision` y
#    `recall` sobre el conjunto de prueba.

# %%
# TU CÓDIGO ACÁ
metricas_enchufe = {}

# %%
check("Calculaste las tres métricas",
      set(metricas_enchufe) == {"accuracy", "precision", "recall"})
if set(metricas_enchufe) == {"accuracy", "precision", "recall"}:
    check("La sensibilidad supera 0.5", metricas_enchufe["recall"] > 0.5,
          "si el recall es 0, tu modelo nunca dice que sí: probá class_weight='balanced'")
    check("La precisión supera 0.5", metricas_enchufe["precision"] > 0.5)
    print()
    for k, v in metricas_enchufe.items():
        print(f"   {k:12s} {v:.3f}")

# %% [markdown]
# ### Ejercicio A4.3 [I] — Ganarle al baseline
#
# Volvé al problema de regresión del riego. Tu tarea: **superar al baseline de
# persistencia**, que tiene el MAE que viste en la tabla de arriba.
#
# Podés cambiar las características, el modelo o sus parámetros. Guardá tu error
# final en `mi_mae` y la lista de características que usaste en `mis_caracteristicas`.
#
# Si no lo lográs, eso también es un resultado: anotalo. Un baseline que no se
# puede vencer es información valiosa sobre el problema, no un fracaso tuyo.

# %%
# TU CÓDIGO ACÁ
mis_caracteristicas = []
mi_mae = None

# %%
_mae_baseline = mean_absolute_error(yr_prueba, Xr_prueba["humedad_suelo_pct"])
print(f"   MAE del baseline de persistencia: {_mae_baseline:.3f} %")
check("Calculaste tu MAE", mi_mae is not None)
if mi_mae is not None:
    print(f"   MAE de tu modelo:                 {mi_mae:.3f} %")
    if mi_mae < _mae_baseline:
        print(f"\n   Le ganaste al baseline por {(_mae_baseline - mi_mae):.3f} puntos de humedad.")
    else:
        print("\n   Todavía no le ganás al baseline. Escribí abajo qué probaste;")
        print("   documentar un intento que no funcionó es una entrega válida.")

# %% [markdown]
# ### Ejercicio A4.4 [A] — La métrica de tu proyecto, justificada
#
# Sin verificación automática: es la entrega del módulo G-3.
#
# Completá esta tabla para **tu** proyecto, en la celda de texto:
#
# 1. ¿Cuál es el **falso positivo** en tu sistema, en palabras concretas? ¿Qué
#    pasa en el mundo real cuando ocurre?
# 2. ¿Cuál es el **falso negativo**, y qué pasa cuando ocurre?
# 3. ¿Cuál de los dos errores es más caro, y **cuánto** más caro? (Aproximá: "el
#    doble", "diez veces", "incomparable".)
# 4. Por lo tanto, la métrica que voy a priorizar es… porque…
# 5. El valor mínimo de esa métrica con el que consideraría que el sistema sirve
#    es… (un número, decidido **antes** de entrenar nada).
#
# El punto 5 es el más incómodo y el más importante. Fijar el umbral de éxito
# después de ver el resultado es hacerse trampa al solitario.

# %% [markdown]
# **Tu respuesta:** *(doble clic para editar)*
#
# 1. Falso positivo en mi proyecto:
# 2. Falso negativo en mi proyecto:
# 3. El más caro es:
# 4. Métrica que priorizo:
# 5. Umbral de éxito declarado de antemano:

# %% [markdown]
# ---
# ## Cierre del cuaderno A-4
#
# **Lo que quedó instalado en tu cabeza:**
#
# - `X` e `y`; el tipo de `y` decide si es clasificación o regresión.
# - Con series temporales se parte **por tiempo**. La partición al azar da
#   números mejores y falsos.
# - Un árbol de decisión aprende reglas que se pueden leer y discutir.
# - La exactitud engaña con clases desbalanceadas: un `return 0` puede sacar 90 %.
# - Todo resultado se compara contra un **baseline trivial**. Sin baseline, un
#   número no es un resultado.
# - Las dos curvas de error separándose son el sobreajuste, dibujado.
#
# **Checklist de entrega**
#
# - [ ] A4.1 con las cinco respuestas correctas.
# - [ ] El clasificador del enchufe con sus tres métricas (A4.2).
# - [ ] El intento de vencer al baseline, con su resultado, haya salido o no (A4.3).
# - [ ] Las cinco respuestas de A4.4, incluido el umbral de éxito declarado
#       **antes** de entrenar.
#
# **Sigue en:** `A5_Reglas_vs_modelo.ipynb` — donde ponemos frente a frente el
# sistema de reglas de A-1 y el modelo de este cuaderno, con la misma métrica.
