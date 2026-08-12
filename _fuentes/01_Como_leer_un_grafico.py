# %% [markdown]
# # Cuaderno 1 · Cómo leer un gráfico
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Arranque del cuatrimestre. Va después del Cuaderno 0 y antes de A-0. |
# | **Duración** | 100 minutos |
# | **Modalidad** | En clase, con discusión |
# | **Requisitos** | Ninguno. |
#
# ### Antes de arrancar
#
# A hacer gráficos nos enseñan. A **leerlos** no nos enseñó nadie, y sin embargo
# es lo que más vas a hacer: en el diario, en una reunión, en la ficha técnica de
# un sensor, y en tu propio proyecto cuando tengas que decidir dónde poner un
# umbral.
#
# Este cuaderno es de lectura, no de dibujo. Vas a mirar gráficos y contestar qué
# dicen. En el cuaderno **A-3** vas a aprender a hacerlos; acá aprendés a
# desconfiar de ellos, que es lo que después te va a hacer hacerlos bien.
#
# > Una advertencia antes de empezar: **casi todos los gráficos que te van a
# > engañar en tu vida no tienen ni un dato falso.** El engaño está en cómo se
# > presentan datos verdaderos. Por eso hace falta un método y no solo buena fe.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from estilo_grafico import (aplicar_estilo, titular, SERIE, ESTADO,
                            TINTA_SUAVE, TINTA_APAGADA)

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
# ## Bloque 1 — El protocolo de los cinco segundos (15 minutos)
#
# Cada vez que te ponen un gráfico adelante, pasá por estos cinco pasos **en este
# orden**, antes de sacar cualquier conclusión. Con práctica te lleva cinco
# segundos.
#
# | | Pregunta | Por qué importa |
# |---|---|---|
# | **1** | ¿Qué dice el título? ¿Es una **etiqueta** o una **afirmación**? | "Consumo mensual" no afirma nada. "El consumo se disparó" ya te está diciendo qué pensar. |
# | **2** | ¿Qué hay en cada eje y **en qué unidad**? | Sin unidad, un 40 puede ser grados, porcentaje o pesos. |
# | **3** | ¿Dónde **arranca** el eje vertical? | Si no arranca en cero y son barras, las alturas mienten. |
# | **4** | ¿Cuántas series hay y **cómo se distinguen**? | Si el único indicio es el color, hay gente que no lo ve. |
# | **5** | ¿Qué **no** está mostrando? | El paso que nadie hace, y el que más sirve. |
#
# ### El paso 5, que es el importante
#
# Un gráfico siempre es un recorte. Preguntate:
#
# - **¿Qué período eligieron, y por qué justo ese?** Un gráfico de tres meses
#   puede mostrar una tendencia que en tres años no existe.
# - **¿Cuántos datos hay detrás de cada punto?** Un promedio de 3 mediciones y uno
#   de 3000 se dibujan igual.
# - **¿Qué pasó con los datos que no dieron?** Si descartaron el 40 % de las
#   lecturas y no lo dicen, el gráfico es de otra cosa.
# - **¿Hay una comparación ausente?** Un modelo con 92 % de acierto parece
#   buenísimo hasta que ves que tirando una moneda cargada sacás 90 %.

# %% [markdown]
# ---
# ## Bloque 2 — Cada forma contesta una pregunta distinta (20 minutos)
#
# Vamos a tomar **exactamente los mismos datos** —el consumo del enchufe
# inteligente— y dibujarlos de cuatro formas. Fijate qué se ve y qué desaparece
# en cada una.

# %%
enchufe = pd.read_csv(DATOS / "enchufe_consumo.csv", parse_dates=["timestamp"])
enchufe.loc[(enchufe["potencia_W"] < 0) | (enchufe["potencia_W"] > 2200), "potencia_W"] = np.nan
serie = enchufe.dropna(subset=["potencia_W"]).set_index("timestamp")["potencia_W"]

fig, ejes = plt.subplots(2, 2, figsize=(11.5, 7.5))

# 1. Línea en el tiempo
ejes[0, 0].plot(serie.index, serie.values, color=SERIE[0], linewidth=0.6)
ejes[0, 0].set_title("A · Línea en el tiempo")
ejes[0, 0].set_ylabel("potencia (W)")
ejes[0, 0].tick_params(axis="x", labelrotation=20)

# 2. Barras por hora del día
por_hora = serie.groupby(serie.index.hour).mean()
ejes[0, 1].bar(por_hora.index, por_hora.values, color=SERIE[0], width=0.8)
ejes[0, 1].set_title("B · Barras por hora del día")
ejes[0, 1].set_xlabel("hora")
ejes[0, 1].set_ylabel("potencia media (W)")

# 3. Histograma
ejes[1, 0].hist(serie.values, bins=50, color=SERIE[0], edgecolor="white", linewidth=0.4)
ejes[1, 0].set_title("C · Histograma (distribución)")
ejes[1, 0].set_xlabel("potencia (W)")
ejes[1, 0].set_ylabel("cantidad de lecturas")

# 4. Dispersión contra la hora
ejes[1, 1].scatter(serie.index.hour + np.random.default_rng(1).uniform(-0.4, 0.4, len(serie)),
                   serie.values, s=4, alpha=0.12, color=SERIE[0], edgecolors="none")
ejes[1, 1].set_title("D · Dispersión: cada lectura, por hora")
ejes[1, 1].set_xlabel("hora")
ejes[1, 1].set_ylabel("potencia (W)")
ejes[1, 1].grid(axis="both")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Qué contesta cada uno (y qué esconde)
#
# | | Contesta bien | Se pierde |
# |---|---|---|
# | **A · línea** | ¿Cambió a lo largo del mes? ¿Hubo un episodio raro? | El patrón diario: son tantos puntos que se hace una mancha. |
# | **B · barras** | ¿A qué hora se concentra el consumo? | Que a las 21 hay días de 400 W y días de 50 W: el promedio los borra. |
# | **C · histograma** | ¿Cuál es el consumo habitual? ¿Hay valores raros? | **El tiempo entero.** No podés saber si los picos fueron todos el mismo día. |
# | **D · dispersión** | Las dos cosas: el patrón horario **y** cuánto varía. | Es la más difícil de leer para alguien no entrenado. |
#
# > **No existe "el gráfico correcto" de un conjunto de datos.** Existe el
# > gráfico correcto **para una pregunta**. Si cambiás la pregunta, cambia el
# > gráfico. Por eso el cuaderno A-3 va a empezar pidiéndote que escribas la
# > pregunta antes de dibujar nada.
#
# Y fijate en la fila del histograma: **perder el tiempo no es un defecto**, es lo
# que lo hace útil. Cada forma gana algo a cambio de tirar algo.

# %% [markdown]
# ---
# ## Bloque 3 — Leer una línea temporal: ponerle nombre a lo que ves (20 minutos)
#
# Cuando mirás una serie en el tiempo hay **cuatro cosas distintas** que pueden
# estar pasando, y conviene poder nombrarlas por separado.

# %%
dias = np.arange(120)
rng = np.random.default_rng(238)

componentes = {
    "1. Tendencia": 20 + 0.15 * dias,
    "2. Estacionalidad": 25 + 5 * np.sin(2 * np.pi * dias / 30),
    "3. Ruido": 25 + rng.normal(0, 2.5, 120),
    "4. Evento puntual": np.where((dias > 60) & (dias < 68), 42, 25) + rng.normal(0, 0.6, 120),
}

fig, ejes = plt.subplots(2, 2, figsize=(11, 6))
for eje, (nombre, valores) in zip(ejes.ravel(), componentes.items()):
    eje.plot(dias, valores, color=SERIE[0])
    eje.set_title(nombre)
    eje.set_ylim(15, 47)
    eje.set_xlabel("día")
plt.tight_layout()
plt.show()

# %% [markdown]
# | Qué es | Cómo se ve | Qué significa en tu proyecto |
# |---|---|---|
# | **Tendencia** | Sube o baja de forma sostenida | Algo está cambiando de fondo: el sensor se descalibra, la estación avanza, el grano se fermenta |
# | **Estacionalidad** | Sube y baja con un ritmo fijo | El ciclo día/noche, el ciclo semanal, el compresor de la heladera |
# | **Ruido** | Tiembla sin patrón | La precisión del sensor. **No significa nada**: no lo interpretes |
# | **Evento puntual** | Un escalón o un pico que empieza y termina | Pasó algo concreto: se apagó, se abrió la bolsa, se enchufó otra cosa |
#
# ### El error más común al leer series
#
# **Confundir ruido con señal.** Si el sensor tiene ±2 °C de precisión y ves que
# la temperatura "subió 1 °C", no subió nada: eso está dentro del temblor normal
# del instrumento.
#
# La regla práctica: **antes de interpretar un cambio, preguntate si es más grande
# que el ruido habitual de esa serie.** Si no lo es, no hay nada que interpretar.

# %%
# Los cuatro juntos, que es lo que ves en la realidad.
real = (20 + 0.15 * dias
        + 5 * np.sin(2 * np.pi * dias / 30)
        + np.where((dias > 60) & (dias < 68), 12, 0)
        + rng.normal(0, 2.0, 120))

fig, ax = plt.subplots()
ax.plot(dias, real, color=SERIE[0], linewidth=1, alpha=0.55, label="lo que medís")
ax.plot(dias, pd.Series(real).rolling(15, center=True).median(),
        color=SERIE[1], linewidth=2.4, label="mediana móvil de 15 días")
ax.set_xlabel("día")
ax.set_ylabel("temperatura (°C)")
ax.legend()
titular(ax, "¿Podés separar las cuatro cosas en esta serie?",
        "Hay tendencia, un ciclo de 30 días, un evento entre el día 60 y el 68, y ruido encima de todo.")
plt.show()

# %% [markdown]
# La línea naranja es un **suavizado**: una mediana móvil. Sirve justamente para
# eso, para sacar el ruido de encima y dejar ver lo demás.
#
# Ojo con una cosa: un suavizado **también borra información**. El evento del día
# 60 se ve más chico y más ancho de lo que fue. Cuando leas un gráfico suavizado,
# preguntá siempre **con qué ventana** lo suavizaron.

# %% [markdown]
# ---
# ## Bloque 4 — Por qué no alcanza con los números (20 minutos)
#
# Acá va la demostración más famosa de la estadística, y con razón.
#
# En 1973, un estadístico llamado Francis Anscombe armó **cuatro conjuntos de
# datos** que tienen prácticamente los mismos promedios, los mismos desvíos y la
# misma correlación. Si solo mirás los números, son idénticos.

# %%
anscombe = {
    "I":   ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II":  ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV":  ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}

resumen = pd.DataFrame({
    nombre: {
        "promedio de x": np.mean(x),
        "promedio de y": np.mean(y),
        "desvío de y": np.std(y, ddof=1),
        "correlación": np.corrcoef(x, y)[0, 1],
    }
    for nombre, (x, y) in anscombe.items()
}).round(2)

print("LOS CUATRO CONJUNTOS, EN NÚMEROS:\n")
print(resumen.to_string())
print("\n¿Conclusión? Son el mismo conjunto de datos. Cerramos y nos vamos.")

# %%
fig, ejes = plt.subplots(2, 2, figsize=(10, 7.5))
for eje, (nombre, (x, y)) in zip(ejes.ravel(), anscombe.items()):
    eje.scatter(x, y, s=90, color=SERIE[0], zorder=3)
    coeficientes = np.polyfit(x, y, 1)
    linea = np.array([2, 20])
    eje.plot(linea, coeficientes[0] * linea + coeficientes[1],
             color=ESTADO["critico"], linewidth=1.6)
    eje.set_xlim(2, 20)
    eje.set_ylim(2, 14)
    eje.set_title(f"Conjunto {nombre}")
    eje.grid(axis="both")

fig.suptitle("Los mismos números, dibujados", x=0.005, ha="left",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Qué acaba de pasar
#
# Los cuatro tienen la misma recta ajustada, y no se parecen en nada:
#
# | Conjunto | Qué es en realidad |
# |---|---|
# | **I** | Una relación lineal con ruido. La recta tiene sentido. |
# | **II** | Una **curva**. La recta es un disparate: sistemáticamente alta en los extremos y baja en el medio. |
# | **III** | Una relación lineal **perfecta** más **un solo dato equivocado**, que tuerce toda la recta. |
# | **IV** | Todos los puntos en la misma x salvo uno. Ese punto solo **define** la recta. Sin él, no hay relación posible. |
#
# > **La conclusión, que vale para toda la materia:** los resúmenes numéricos
# > —promedio, desvío, correlación, y también la exactitud de un modelo— **pueden
# > coincidir en situaciones completamente distintas**. Por eso se mira el gráfico
# > *además* del número, nunca en lugar de.
#
# Y fijate lo que pasa en los conjuntos III y IV: **un solo dato manda**. Eso es
# exactamente lo que vas a estar cazando en el cuaderno A-2 cuando busques
# lecturas fuera de rango. Un dato invalido no te corre el promedio un poquito:
# te puede dar vuelta la conclusión entera.

# %% [markdown]
# ---
# ## Bloque 5 — Leer una distribución: cuándo el promedio miente (15 minutos)
#
# El promedio es el número que más se usa y el que más se malinterpreta. Miremos
# por qué con los tiempos de respuesta de un nodo.

# %%
rng3 = np.random.default_rng(11)
# La mayoría de las respuestas son rápidas; algunas se cuelgan mucho.
tiempos = np.concatenate([
    rng3.normal(45, 8, 900),          # respuestas normales
    rng3.normal(400, 120, 60),        # reintentos por WiFi
    rng3.normal(1800, 300, 12),       # timeouts
])
tiempos = tiempos[tiempos > 0]

promedio, mediana = np.mean(tiempos), np.median(tiempos)
p95 = np.percentile(tiempos, 95)

# Dos paneles: el mismo dato entero y ampliado. La cola larga es tan larga que,
# si dibujamos todo junto, el grueso de las respuestas se aplasta contra el eje
# y no se ve nada. Eso, en sí mismo, es parte de la lección.
CORTE = 250
fuera = int((tiempos > CORTE).sum())

fig, (todo, ampliado) = plt.subplots(1, 2, figsize=(11.5, 4.2))

todo.hist(tiempos, bins=70, color=SERIE[0], edgecolor="white", linewidth=0.3)
todo.set_title("Todo el rango: no se ve nada")
todo.set_xlabel("tiempo de respuesta (ms)")
todo.set_ylabel("cantidad de respuestas")

ampliado.hist(tiempos, bins=60, range=(0, CORTE), color=SERIE[0],
              edgecolor="white", linewidth=0.3)
for valor, color in [(mediana, ESTADO["bien"]), (promedio, ESTADO["grave"])]:
    ampliado.axvline(valor, color=color, linewidth=2)
ampliado.set_xlim(0, CORTE)
ampliado.set_xlabel("tiempo de respuesta (ms)")
ampliado.set_title(f"Ampliado hasta {CORTE} ms  ({fuera} respuestas quedan afuera)",
                   fontsize=11)

# Las referencias van en un bloque aparte: si se rotulan sobre las líneas, se
# pisan entre ellas porque la mediana y el promedio están muy cerca.
texto = (f"mediana        {mediana:6.0f} ms\n"
         f"promedio       {promedio:6.0f} ms\n"
         f"percentil 95   {p95:6.0f} ms  (ni entra acá)")
ampliado.text(0.97, 0.95, texto, transform=ampliado.transAxes, ha="right", va="top",
              fontsize=9.5, family="monospace", color=TINTA_SUAVE)

fig.suptitle("¿Cuánto tarda en responder el nodo?", x=0.005, ha="left",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"mediana        {mediana:7.0f} ms   -> la mitad de las veces tarda menos que esto")
print(f"promedio       {promedio:7.0f} ms   -> lo corren unos pocos casos malísimos")
print(f"percentil 95   {p95:7.0f} ms   -> 1 de cada 20 veces tarda MÁS que esto")

# %% [markdown]
# ### Cómo leer esta forma
#
# Esta distribución tiene **cola larga a la derecha**: casi todo se acumula
# temprano y unos pocos casos se van lejísimos. Es la forma más común en
# ingeniería (tiempos de respuesta, consumos, duraciones de reparación).
#
# En una distribución así:
#
# - **El promedio está corrido hacia arriba** por los casos extremos. Está bien
#   calculado y describe mal a la mayoría.
# - **La mediana describe al caso típico.** La mitad está de cada lado.
# - **El percentil 95 describe la mala experiencia.** Es el número que importa si
#   te preguntan "¿cada cuánto se cuelga?".
#
# | Si te preguntan | Mirá |
# |---|---|
# | ¿Cómo anda normalmente? | la **mediana** |
# | ¿Cuánto consume en total el mes? | el **promedio** (por el total sí sirve) |
# | ¿Qué tan mal se pone cuando se pone mal? | el **percentil 95** o el máximo |
#
# > Cuando alguien te dé un solo número para describir un montón de mediciones,
# > **preguntá cuál de los tres es**. Si no sabe contestarte, no leyó sus datos.

# %% [markdown]
# ---
# ## Bloque 6 — El catálogo de engaños, desde el lado del que lee (15 minutos)
#
# Ocho trucos. Ninguno necesita un dato falso.

# %%
fig, ejes = plt.subplots(2, 2, figsize=(11, 7))
etiquetas = ["A", "B", "C", "D"]
valores = [438, 445, 452, 461]

# 1. Eje truncado
ejes[0, 0].bar(etiquetas, valores, color=ESTADO["critico"], width=0.6)
ejes[0, 0].set_ylim(430, 465)
ejes[0, 0].set_title("1 · Eje truncado: «se triplicó»")

# 2. El mismo, honesto
ejes[0, 1].bar(etiquetas, valores, color=SERIE[0], width=0.6)
ejes[0, 1].set_ylim(0, 500)
ejes[0, 1].set_title("El mismo dato desde cero: subió 5 %")

# 3. Recorte del período
largo = np.concatenate([np.linspace(100, 60, 40), np.linspace(60, 95, 20)])
ejes[1, 0].plot(largo, color=SERIE[0])
ejes[1, 0].axvspan(40, 59, color=ESTADO["critico"], alpha=0.13)
ejes[1, 0].text(41, 92, "el trozo que\nte muestran", color=ESTADO["critico"], fontsize=9)
ejes[1, 0].set_title("2 · Recorte del período: «viene subiendo»")

# 4. Burbujas: área contra radio
for i, (x, valor) in enumerate(zip([1, 2], [10, 20])):
    ejes[1, 1].scatter([x], [1], s=(valor ** 2) * 6, color=SERIE[0], alpha=0.75)
    ejes[1, 1].text(x, 1.6, f"{valor}", ha="center", color=TINTA_SUAVE)
ejes[1, 1].set_xlim(0.4, 2.6)
ejes[1, 1].set_ylim(0, 2.2)
ejes[1, 1].set_title("3 · Burbujas: el doble se ve como el cuádruple")
ejes[1, 1].set_yticks([])
ejes[1, 1].grid(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### La lista completa, para tener a mano
#
# | # | El truco | Cómo lo cazás |
# |---|---|---|
# | **1** | **Eje truncado**: barras que no arrancan en cero | Mirá el primer número del eje vertical. Si no es 0 y son barras, recalculá el porcentaje real. |
# | **2** | **Recorte del período**: eligen justo el tramo que conviene | Preguntá por qué empieza ahí. Pedí la serie completa. |
# | **3** | **Burbujas o íconos**: codifican el valor en el radio | Si el radio se duplica, el área se cuadruplica. El ojo lee área. |
# | **4** | **Doble eje**: dos escalas verticales | Cambiando los límites se fabrica cualquier apariencia de correlación. Pedí el número (el coeficiente), no el dibujo. |
# | **5** | **Eje invertido**: el eje va de mayor a menor | Se ve una curva subiendo cuando en realidad baja. Leé el orden de los números. |
# | **6** | **Escala logarítmica sin avisar** | Fijate si el eje va 1, 10, 100, 1000. Ahí una subida chica es enorme. |
# | **7** | **Torta con muchas porciones** | El ojo no compara ángulos. Con más de 5 categorías, pedí barras. |
# | **8** | **Falta la incertidumbre** | ¿Cuántas mediciones hay detrás de cada punto? ¿Cuánto varían? Sin eso, no se puede saber si la diferencia es real. |
#
# Y uno que no es un truco pero engaña igual:
#
# | **9** | **Correlación presentada como causa** | Que dos cosas suban juntas no significa que una cause la otra. Puede haber una tercera causa empujando a las dos —o pura casualidad. |

# %% [markdown]
# ---
# ## Bloque 7 — Ejercicios de lectura
#
# Ahora te toca a vos. Mirá cada gráfico y contestá **antes** de ejecutar la celda
# de verificación.

# %% [markdown]
# ### Ejercicio 1.1 [B] — Leé estos dos

# %%
rng4 = np.random.default_rng(3)
sem = np.arange(1, 13)
valores_a = 100 + rng4.normal(0, 1.2, 12)

fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 3.8))
izq.plot(sem, valores_a, color=SERIE[0], marker="o")
izq.set_ylim(97.5, 102.5)
izq.set_xlabel("semana")
izq.set_ylabel("consumo (kWh)")
izq.set_title("Gráfico 1")

der.plot(sem, valores_a, color=SERIE[0], marker="o")
der.set_ylim(0, 130)
der.set_xlabel("semana")
der.set_ylabel("consumo (kWh)")
der.set_title("Gráfico 2")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Pregunta:** los dos gráficos muestran exactamente los mismos datos. ¿Cuál es
# la lectura correcta?
#
# - `"a"` — El consumo tuvo variaciones importantes durante el trimestre.
# - `"b"` — El consumo se mantuvo prácticamente estable; lo que se ve en el
#   Gráfico 1 es ruido de medición.
# - `"c"` — El Gráfico 2 está mal hecho porque desperdicia espacio.

# %%
# TU CÓDIGO ACÁ
respuesta_1_1 = ""

# %%
if check("Respuesta correcta", respuesta_1_1.strip().lower() == "b",
         "mirá cuánto vale la variación total comparada con el valor absoluto"):
    print(f"\n   La variación total es de {valores_a.max() - valores_a.min():.1f} kWh "
          f"sobre {valores_a.mean():.0f}: menos del "
          f"{(valores_a.max() - valores_a.min()) / valores_a.mean() * 100:.0f} %.")
    print("   El Gráfico 1 no miente, pero invita a interpretar ruido como señal.")
    print("   La opción (c) es la trampa: el 'espacio desperdiciado' del Gráfico 2")
    print("   ES la información. Muestra que la variación es chica.")

# %% [markdown]
# ### Ejercicio 1.2 [B] — ¿Qué falta acá?

# %%
categorias = ["Modelo A", "Modelo B", "Modelo C"]
exactitud = [0.91, 0.93, 0.92]

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar(categorias, exactitud, color=SERIE[0], width=0.55)
ax.set_ylim(0.85, 0.95)
ax.set_ylabel("exactitud")
ax.set_title("Comparación de modelos")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Pregunta:** ¿cuál es el problema **más grave** de este gráfico?
#
# - `"a"` — El eje no arranca en cero.
# - `"b"` — Faltan los colores para distinguir los modelos.
# - `"c"` — No hay ninguna referencia contra la cual comparar: no sabemos qué
#   saca un modelo trivial, ni cuánta variación hay entre corridas.
# - `"d"` — El título es poco descriptivo.

# %%
# TU CÓDIGO ACÁ
respuesta_1_2 = ""

# %%
if check("Respuesta correcta", respuesta_1_2.strip().lower() == "c",
         "pensá qué te haría falta para saber si un 93 % es bueno o malo"):
    print("\n   El eje truncado (a) también es un problema, y el título (d) es")
    print("   mejorable. Pero son cosméticos al lado de (c): sin baseline, un")
    print("   93 % no significa nada. Si el 91 % de los casos son de una sola")
    print("   clase, un modelo que dice siempre lo mismo saca 91 %.")
    print("\n   Ese razonamiento es exactamente el del cuaderno A-4. Anotalo.")

# %% [markdown]
# ### Ejercicio 1.3 [I] — Nombrá lo que ves

# %%
silo = pd.read_csv(DATOS / "silobolsa_gas.csv", parse_dates=["timestamp"])
silo.loc[(silo["co2_ppm"] < 350) | (silo["co2_ppm"] > 1500), "co2_ppm"] = np.nan
gas = silo.dropna(subset=["co2_ppm"]).set_index("timestamp")["co2_ppm"]

fig, ax = plt.subplots()
ax.plot(gas.index, gas.values, color=SERIE[0], linewidth=0.7)
ax.set_ylabel("CO₂ (ppm)")
ax.set_title("CO₂ dentro de la silobolsa")
plt.show()

# %% [markdown]
# **Pregunta:** de las cuatro cosas del Bloque 3 (tendencia, estacionalidad,
# ruido, evento puntual), ¿cuál domina este gráfico?
#
# Guardá tu respuesta como texto: `"tendencia"`, `"estacionalidad"`, `"ruido"` o
# `"evento puntual"`.
#
# Y en `cuantos_eventos`, cuántos de esos episodios contás a simple vista.

# %%
# TU CÓDIGO ACÁ
respuesta_1_3 = ""
cuantos_eventos = 0

# %%
check("Identificaste el componente dominante",
      respuesta_1_3.strip().lower() == "evento puntual",
      "¿la línea sube de a poco todo el tiempo, o hay episodios que empiezan y terminan?")
if check("Contaste los episodios", cuantos_eventos in (3, 7),
         "hay episodios grandes y otros más chicos: contá los que veas"):
    if cuantos_eventos == 7:
        print("\n   Muy buen ojo: contaste los 3 grandes Y los 4 chicos.")
    else:
        print("\n   Bien: esos son los 3 grandes, los que llegan bien alto.")
        print("   Mirá de nuevo con atención: hay 4 más chicos que suben y se")
        print("   vuelven solos sin llegar arriba.")
    print("\n   Esos 4 episodios chicos son la razón por la que este problema NO")
    print("   se resuelve con un umbral y listo. Lo vas a trabajar en el A-4.")

# %% [markdown]
# ### Ejercicio 1.4 [I] — El promedio contra la mediana
#
# Volvé al histograma de tiempos de respuesta del Bloque 5.
#
# Un compañero escribe en el informe: *"el nodo responde en 89 ms en promedio,
# así que la comunicación es rápida"*. El número está bien calculado. La
# conclusión, no tanto.

# %%
# TU CÓDIGO ACÁ
# ¿Qué porcentaje de las respuestas tardó MENOS que el promedio?
porcentaje_bajo_el_promedio = None

# %%
_esperado = (tiempos < promedio).mean() * 100
if check("Calculaste el porcentaje",
         porcentaje_bajo_el_promedio is not None
         and abs(porcentaje_bajo_el_promedio - _esperado) < 1.0,
         "usá (tiempos < promedio).mean() * 100"):
    print(f"\n   El {_esperado:.0f} % de las respuestas tardó MENOS que el promedio.")
    print("   O sea: el 'promedio' no describe a casi nadie. Describe un punto")
    print("   entre la mayoría rápida y unos pocos casos malísimos.")
    print(f"\n   La frase correcta para el informe sería: 'la mitad de las")
    print(f"   respuestas tardó menos de {mediana:.0f} ms, pero 1 de cada 20")
    print(f"   superó los {p95:.0f} ms'.")

# %% [markdown]
# ### Ejercicio 1.5 [A] — Traé un gráfico de la calle
#
# Sin verificación automática. Es la entrega de este cuaderno.
#
# 1. Buscá un gráfico **fuera de esta materia**: en un diario, en una red social,
#    en la ficha técnica de un componente, en el informe de una empresa.
# 2. Pegalo (captura o enlace) en la celda de abajo.
# 3. Aplicale **los cinco pasos del Bloque 1**, uno por uno, por escrito.
# 4. Decí si encontraste alguno de los nueve trucos del Bloque 6, y cuál.
# 5. Escribí **la afirmación que el gráfico soporta de verdad**, en una oración.
#    Ojo: puede ser bastante más modesta que la que el gráfico sugiere.
#
# El punto 5 es la habilidad completa. Todo lo demás es preparación para eso.

# %% [markdown]
# **Tu análisis:** *(doble clic para editar)*
#
# **Gráfico elegido:** *(enlace o descripción)*
#
# **1. Título — ¿etiqueta o afirmación?**
#
# **2. Ejes y unidades:**
#
# **3. ¿Dónde arranca el eje vertical?**
#
# **4. Series y cómo se distinguen:**
#
# **5. Qué NO muestra:**
#
# **Trucos detectados:**
#
# **Lo que el gráfico soporta de verdad:**

# %% [markdown]
# ---
# ## Cierre del Cuaderno 1
#
# **Lo que te tenés que llevar:**
#
# 1. **Los cinco pasos, en orden.** Y sobre todo el quinto: qué *no* está
#    mostrando.
# 2. **Cada forma de gráfico gana algo a cambio de tirar algo.** No hay gráfico
#    correcto sin una pregunta.
# 3. **Tendencia, estacionalidad, ruido y evento** son cuatro cosas distintas.
#    El ruido no se interpreta.
# 4. **Anscombe:** cuatro conjuntos con los mismos números y formas opuestas. El
#    resumen numérico nunca reemplaza al gráfico, ni al revés.
# 5. **El promedio, la mediana y el percentil 95** contestan preguntas distintas.
#    Cuando te den un solo número, preguntá cuál es.
# 6. **Casi ningún gráfico engañoso tiene datos falsos.**
#
# **Checklist de entrega**
#
# - [ ] Las cuatro respuestas de opción múltiple correctas (1.1 a 1.4).
# - [ ] Un gráfico de afuera de la materia, analizado con los cinco pasos (1.5).
# - [ ] La afirmación que ese gráfico soporta de verdad, escrita en una oración.
#
# **Sigue en:** `A0_Entorno_y_herramientas.ipynb` — a partir de acá empieza el
# trabajo con tus propios datos.
