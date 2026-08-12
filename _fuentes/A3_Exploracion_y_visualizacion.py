# %% [markdown]
# # Anexo A-3 · Exploración y visualización que decide algo
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Módulo **F-3** del cuadernillo |
# | **Duración** | 120 minutos |
# | **Modalidad** | En equipo, sobre el proyecto propio |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Elegir el tipo de gráfico a partir de **la pregunta**, y no de lo que quede
#    más lindo.
# 2. Producir las cuatro figuras que tu informe F-5 necesita, con estilo
#    consistente y guardadas en disco.
# 3. Separar una **tendencia** del ruido con promedios móviles y líneas base.
# 4. Reconocer y desarmar tres formas habituales en que un gráfico miente sin
#    decir una sola mentira.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from estilo_grafico import (aplicar_estilo, titular, guardar,
                            SERIE, ESTADO, TINTA_APAGADA, TINTA_SUAVE)

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


# Los datos ya limpios: repetimos acá la limpieza de A-2 en forma condensada.
def cargar_limpio(archivo, columna, minimo, maximo, repeticiones=8):
    tabla = pd.read_csv(DATOS / archivo, parse_dates=["timestamp"])
    serie = tabla[columna]
    grupo = (serie != serie.shift()).cumsum()
    congelado = serie.groupby(grupo).transform("size") >= repeticiones
    invalido = (serie < minimo) | (serie > maximo) | serie.isna() | congelado
    invalido |= (tabla["timestamp"].diff() <= pd.Timedelta(0))
    tabla.loc[invalido, columna] = np.nan
    tabla = tabla.sort_values("timestamp").reset_index(drop=True)
    return tabla


silo = cargar_limpio("silobolsa_gas.csv", "co2_ppm", 350, 1500)
enchufe = cargar_limpio("enchufe_consumo.csv", "potencia_W", 0, 2200)
riego = cargar_limpio("riego_humedad.csv", "humedad_suelo_pct", 0, 100)

print(f"silobolsa {silo['co2_ppm'].notna().sum()} lecturas válidas de {len(silo)}")

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · El gráfico se elige desde la pregunta (25 minutos)
#
# El cuadernillo lo dice en una línea: *"un gráfico que se ve lindo pero no
# ayuda a decidir nada no cumple su función"*. La versión operativa de esa idea
# es un procedimiento de tres pasos, en este orden y no en otro:
#
# 1. **Escribí la pregunta.** Textual, con signos de interrogación.
# 2. **Elegí la forma** según qué tipo de trabajo hace el dato.
# 3. **Recién ahí** pensá en colores.
#
# La mayoría de los gráficos malos se hacen al revés: se elige el color y el tipo
# primero, y después se busca qué decir.
#
# ### Qué forma para qué trabajo
#
# | Si la pregunta es sobre… | La forma es… | Ejemplo de esta cohorte |
# |---|---|---|
# | **cambio en el tiempo** | línea | ¿el CO₂ viene subiendo esta semana? |
# | **comparar magnitudes entre categorías** | barras | ¿qué franja horaria consume más? |
# | **relación entre dos variables** | dispersión | ¿la temperatura explica el CO₂? |
# | **cómo se reparten los valores** | histograma | ¿en qué humedad vive la planta la mayor parte del tiempo? |
# | **un solo número que importa** | ningún gráfico: el número grande | 3 focos de fermentación en 90 días |
#
# Esa última fila es en serio. Si la respuesta es un número, **escribí el
# número**. Un gráfico de torta con un solo valor es peor que el valor.
#
# ### Cuatro reglas que no se negocian
#
# 1. **Un solo eje vertical.** Nunca dos escalas Y en el mismo gráfico. Es el
#    error más común y el más engañoso; lo demostramos abajo.
# 2. **La grilla y los ejes van apagados.** Los datos son lo que tiene que
#    saltar a la vista, no la decoración.
# 3. **Dos series o más llevan leyenda siempre.** El color solo no puede cargar
#    el significado: hay gente que no lo distingue.
# 4. **El color sigue a la entidad, no al puesto.** Si "riego" es naranja, es
#    naranja siempre, aunque cambie de posición al filtrar.

# %% [markdown]
# ---
# ## Bloque 2 — Práctica · Las cuatro figuras del informe (45 minutos)

# %% [markdown]
# ### Figura 1 — Línea en el tiempo: ¿el CO₂ viene subiendo?
#
# Con 4320 puntos, el gráfico crudo es una mancha. La solución no es graficar
# menos datos: es **agregar una capa de tendencia** encima del dato crudo, y
# dejar el crudo atenuado como contexto.

# %%
serie = silo.set_index("timestamp")["co2_ppm"]
diario = serie.resample("D").mean()
linea_base = serie.rolling("7D").median()   # mediana: no la corren los picos
UMBRAL = 1000

fig, ax = plt.subplots()
ax.plot(serie.index, serie.values, color=SERIE[0], alpha=0.25, linewidth=0.8,
        label="lectura cada 30 min")
ax.plot(linea_base.index, linea_base.values, color=SERIE[0], linewidth=2.2,
        label="línea base (mediana móvil de 7 días)")
ax.axhline(UMBRAL, color=ESTADO["critico"], linewidth=1.5, linestyle="--")
ax.text(serie.index[10], UMBRAL + 25, f"umbral de alerta · {UMBRAL} ppm",
        color=ESTADO["critico"], fontsize=9)

ax.set_ylabel("CO₂ (ppm)")
ax.legend(loc="upper left")
titular(ax, "¿El CO₂ dentro de la silobolsa viene subiendo?",
        "Tres episodios llevan el CO₂ por encima del umbral; entre ellos vuelve a ~450 ppm.")
plt.show()

# %% [markdown]
# Fijate qué hace ese gráfico y qué no. **No** muestra "el CO₂ a lo largo del
# tiempo": muestra que hay **tres episodios discretos** y que entre ellos el
# sistema vuelve a su línea base. Esa es una afirmación sobre el fenómeno, y se
# puede escribir en una oración. Ese es el estándar del módulo F-3: *"escribí una
# conclusión de una línea a partir del gráfico: qué te dice, no solo qué
# muestra"*.
#
# > **Por qué mediana móvil y no promedio móvil:** el promedio se va detrás de
# > cualquier pico. La mediana se queda quieta salvo que la mitad de la ventana
# > se haya movido. Para una línea base —que quiere decir *"cuánto es lo
# > normal"*— casi siempre querés la mediana.

# %% [markdown]
# ### Figura 2 — Barras: ¿a qué hora consume más el enchufe?
#
# Acá la pregunta es de comparación entre categorías (las horas del día), así
# que la forma es barras. Dos detalles que la hacen legible: las barras arrancan
# **siempre en cero** (una barra codifica magnitud por su largo, así que
# truncarla es mentir), y en lugar de rotular las 24 barras, se rotula
# **selectivamente** solo la que importa.

# %%
por_hora = enchufe.assign(hora=enchufe["timestamp"].dt.hour) \
                  .groupby("hora")["potencia_W"].mean()
hora_pico = int(por_hora.idxmax())

colores = [SERIE[0] if h != hora_pico else ESTADO["grave"] for h in por_hora.index]

fig, ax = plt.subplots()
ax.bar(por_hora.index, por_hora.values, color=colores, width=0.78)
ax.annotate(f"{por_hora.max():.0f} W a las {hora_pico}:00",
            xy=(hora_pico, por_hora.max()), xytext=(hora_pico - 6.5, por_hora.max() * 0.94),
            color=TINTA_SUAVE, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=TINTA_APAGADA, linewidth=1))

ax.set_xlabel("hora del día")
ax.set_ylabel("potencia media (W)")
ax.set_xticks(range(0, 24, 3))
ax.set_ylim(0, None)
titular(ax, "¿En qué franja horaria se concentra el consumo?",
        f"El pico está a las {hora_pico}:00; la madrugada es solo el ciclo de la heladera.")
plt.show()

# %% [markdown]
# ### Figura 3 — Dispersión: ¿la temperatura explica el CO₂?
#
# Cuando la pregunta es *"¿estas dos variables se mueven juntas?"*, la forma es
# dispersión. Con miles de puntos hay que bajar la opacidad, si no se forma una
# mancha sólida que no deja ver dónde se acumulan.
#
# Y hay algo más importante que el gráfico: **acompañarlo con un número**. El
# coeficiente de correlación de Pearson mide, entre −1 y 1, qué tan bien una
# recta describe la relación.

# %%
valido = silo.dropna(subset=["co2_ppm", "temperatura_C"])
correlacion = valido["co2_ppm"].corr(valido["temperatura_C"])

fig, ax = plt.subplots(figsize=(6.2, 5))
ax.scatter(valido["temperatura_C"], valido["co2_ppm"],
           s=9, alpha=0.18, color=SERIE[0], edgecolors="none")
ax.set_xlabel("temperatura (°C)")
ax.set_ylabel("CO₂ (ppm)")
ax.grid(axis="both")
titular(ax, "¿La temperatura explica el nivel de CO₂?",
        f"Correlación de Pearson: r = {correlacion:.2f}")
plt.show()

print(f"r = {correlacion:.3f}   ->   r² = {correlacion ** 2:.3f}")
print(f"La temperatura explica alrededor del {correlacion ** 2 * 100:.0f} % "
      f"de la variación del CO₂.")

# %% [markdown]
# > **Correlación no es causalidad, pero acá hay algo más específico que decir.**
# > En este caso *sabemos* que hay una relación causal, porque la fermentación
# > genera calor **y** genera CO₂: las dos variables suben juntas porque las
# > empuja la misma causa de fondo. La temperatura no *causa* el CO₂; los dos son
# > efectos del mismo proceso.
# >
# > Esa distinción tiene una consecuencia práctica directa para el cuaderno A-4:
# > si la temperatura sube junto con el CO₂, entonces **sirve como variable de
# > entrada para anticiparlo**, aunque no lo cause. Un modelo predictivo se
# > aprovecha de las correlaciones; solo un modelo causal necesita causas.

# %% [markdown]
# ### Figura 4 — Distribución: ¿en qué humedad vive la planta?
#
# El promedio de humedad no dice casi nada si la humedad oscila entre 30 y 75. La
# pregunta correcta es cómo se **reparten** los valores, y para eso el
# histograma.

# %%
humedad = riego["humedad_suelo_pct"].dropna()

fig, ax = plt.subplots()
ax.hist(humedad, bins=40, color=SERIE[0], edgecolor="white", linewidth=0.6)
ax.axvline(humedad.median(), color=ESTADO["grave"], linewidth=2)
ax.text(humedad.median() + 1, ax.get_ylim()[1] * 0.9,
        f"mediana {humedad.median():.0f} %", color=ESTADO["grave"], fontsize=9.5)
ax.axvspan(0, 30, color=ESTADO["critico"], alpha=0.10)
ax.text(31, ax.get_ylim()[1] * 0.6, "zona de estrés\n(por debajo de 30 %)",
        color=TINTA_SUAVE, fontsize=9)

ax.set_xlabel("humedad de suelo (%)")
ax.set_ylabel("cantidad de lecturas")
ax.grid(axis="y")
titular(ax, "¿En qué rango de humedad vive la planta la mayor parte del tiempo?",
        "La distribución es ancha y bastante plana: el controlador la pasea por toda la banda.")
plt.show()

# %% [markdown]
# ---
# ## Bloque 3 — Teoría · Tres formas de mentir sin mentir (25 minutos)
#
# Ninguno de los tres gráficos que siguen tiene un dato falso. Los tres inducen
# una conclusión equivocada. Conviene saber reconocerlos porque los vas a
# encontrar —y porque los vas a hacer sin querer.

# %% [markdown]
# ### Mentira 1 — El eje truncado
#
# Una barra codifica su valor por el **largo**. Si el eje no arranca en cero, el
# largo deja de ser proporcional al valor y la comparación visual es falsa.

# %%
semanas = ["sem 1", "sem 2", "sem 3", "sem 4"]
valores = [438, 445, 452, 461]

fig, (izq, der) = plt.subplots(1, 2, figsize=(10.5, 4))

izq.bar(semanas, valores, color=ESTADO["critico"], width=0.6)
izq.set_ylim(430, 465)
izq.set_title("«El CO₂ se disparó»", color=ESTADO["critico"])
izq.set_ylabel("CO₂ (ppm)")

der.bar(semanas, valores, color=SERIE[0], width=0.6)
der.set_ylim(0, 465)
der.set_title("Los mismos cuatro números, desde cero")
der.set_ylabel("CO₂ (ppm)")

fig.suptitle("Mentira 1 · el eje truncado", x=0.005, ha="left",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"La variación real es de {valores[-1] - valores[0]} ppm sobre {valores[0]}: "
      f"{(valores[-1] / valores[0] - 1) * 100:.1f} %.")

# %% [markdown]
# ### Mentira 2 — El doble eje
#
# Poner dos variables con escalas distintas en el mismo gráfico, cada una con su
# eje Y, permite **fabricar la apariencia de correlación que quieras**: alcanza
# con elegir los límites de cada eje. Y como los límites parecen un detalle
# técnico, nadie los mira.
#
# Abajo, los **mismos dos conjuntos de datos**, dibujados dos veces con distinto
# recorte de ejes.

# %%
dias = pd.date_range("2026-09-05", periods=60, freq="D")
rng = np.random.default_rng(7)
gas = 430 + np.cumsum(rng.normal(0, 6, 60))
# La temperatura acompaña al gas (los dos los empuja la fermentación) más ruido.
temperatura = 19 + 0.012 * (gas - gas.mean()) + rng.normal(0, 0.12, 60)


def con_margen(valores, factor=1.0):
    """Límites de eje que dejan la serie completa adentro, con un respiro."""
    bajo, alto = valores.min(), valores.max()
    respiro = (alto - bajo) * 0.08
    centro, medio_alto = (alto + bajo) / 2, (alto - bajo) / 2 + respiro
    return centro - medio_alto * factor, centro + medio_alto * factor


def panel_doble_eje(eje, lim_gas, lim_temp, titulo):
    eje.plot(dias, gas, color=SERIE[0], linewidth=1.8)
    eje.set_ylim(*lim_gas)
    eje.set_ylabel("CO₂ (ppm)", color=SERIE[0])
    eje.tick_params(axis="y", colors=SERIE[0])
    gemelo = eje.twinx()
    gemelo.plot(dias, temperatura, color=SERIE[1], linewidth=1.8)
    gemelo.set_ylim(*lim_temp)
    gemelo.set_ylabel("temperatura (°C)", color=SERIE[1])
    gemelo.tick_params(axis="y", colors=SERIE[1])
    gemelo.spines["top"].set_visible(False)
    eje.set_title(titulo)
    eje.set_xticks(dias[::25])
    eje.grid(False)


fig, (izquierda, derecha) = plt.subplots(1, 2, figsize=(11.5, 4))

# Truco A: cada eje ajustado al rango exacto de SU serie. Las dos curvas pasan a
# ocupar todo el alto del panel y sus formas se superponen casi exactamente.
panel_doble_eje(izquierda, con_margen(gas), con_margen(temperatura),
                "«Van juntas, clarísimo»")

# Truco B: al CO₂ le damos un eje cinco veces más ancho de lo necesario. La misma
# curva se aplasta contra el centro y parece que no pasara nada.
panel_doble_eje(derecha, con_margen(gas, factor=5), con_margen(temperatura),
                "«No tienen nada que ver»")

fig.suptitle("Mentira 2 · el doble eje: los mismos datos, dos conclusiones opuestas",
             x=0.005, ha="left", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"Correlación real entre las dos series: r = {np.corrcoef(gas, temperatura)[0, 1]:.2f}")
print("El número es el mismo en los dos paneles. Lo único que cambió fueron los")
print("límites de los ejes, que son justamente lo que nadie mira.")

# %% [markdown]
# Con estos datos, la relación **existe**: el coeficiente lo confirma. Así que el
# panel de la izquierda dice algo cierto y el de la derecha, algo falso.
#
# Pero ese no es el punto. El punto es que **vos, mirando cualquiera de los dos
# paneles, no tenías cómo saber cuál era cuál.** Un gráfico de doble eje se puede
# construir para respaldar la conclusión que uno ya quería, y el lector no tiene
# forma de auditarlo salvo leyendo los límites de los ejes, que nadie lee.
#
# > Por eso la regla no es "usá bien el doble eje": es **no lo uses**. Una técnica
# > cuya corrección depende de que el autor sea honesto no es una técnica, es una
# > promesa.

# %% [markdown]
# **La forma correcta** de comparar dos variables de escalas distintas en el
# tiempo son **dos paneles apilados que comparten el eje X**. Cada uno con su
# escala, honesta, y la comparación la hace el ojo sobre el eje temporal común.

# %%
fig, (p1, p2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

p1.plot(dias, gas, color=SERIE[0])
p1.set_ylabel("CO₂ (ppm)")
p1.set_title("La forma correcta: dos paneles, un eje temporal compartido")

p2.plot(dias, temperatura, color=SERIE[1])
p2.set_ylabel("temperatura (°C)")
p2.set_xlabel("fecha")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Mentira 3 — El promedio que esconde el evento
#
# Es la misma trampa del muestreo que vimos en A-2, ahora del lado del gráfico.
# Promediar por día es cómodo y suele ser lo correcto… salvo cuando lo que
# importa es exactamente el pico.

# %%
serie_enchufe = enchufe.set_index("timestamp")["potencia_W"]

fig, ax = plt.subplots()
ax.plot(serie_enchufe.resample("D").mean(), color=SERIE[0],
        label="promedio diario")
ax.plot(serie_enchufe.resample("D").max(), color=SERIE[1],
        label="máximo diario")
ax.set_ylabel("potencia (W)")
ax.legend(loc="upper left")
titular(ax, "¿Alcanza con el promedio diario para dimensionar la instalación?",
        "El promedio ronda los 60 W; los picos reales pasan de 400 W.")
plt.show()

# %% [markdown]
# Si dimensionás el cableado o el relé con el promedio, elegís un componente para
# 60 W cuando la instalación ve picos de más de 400 W. **Para dimensionar se usa
# el máximo; para estimar el costo de la energía se usa el promedio.** Cuál de
# los dos corresponde depende de la decisión que el gráfico tiene que apoyar, que
# es —otra vez— la pregunta del paso 1.

# %% [markdown]
# ---
# ## Bloque 4 — Ejercicios

# %% [markdown]
# ### Ejercicio A3.1 [B] — La pregunta antes que el gráfico
#
# Completá `mis_preguntas`: tres preguntas sobre **tu** proyecto que un gráfico
# podría contestar, cada una con la forma que le corresponde.
#
# Las formas válidas son: `"linea"`, `"barras"`, `"dispersion"`, `"histograma"`,
# `"ningun grafico"`.

# %%
# TU CÓDIGO ACÁ
mis_preguntas = [
    # {"pregunta": "¿...?", "forma": "linea", "por_que": "..."},
]

# %%
_formas = {"linea", "barras", "dispersion", "histograma", "ningun grafico"}
check("Escribiste tres preguntas", len(mis_preguntas) == 3)
check("Todas están redactadas como pregunta",
      bool(mis_preguntas) and all("?" in p["pregunta"] for p in mis_preguntas),
      "una pregunta lleva signos de interrogación; si no los tiene, es un título")
check("Las formas elegidas son válidas",
      bool(mis_preguntas) and all(p["forma"] in _formas for p in mis_preguntas))
check("Usaste al menos dos formas distintas",
      len({p["forma"] for p in mis_preguntas}) >= 2,
      "si las tres preguntas piden el mismo gráfico, probablemente son la misma pregunta")

# %% [markdown]
# ### Ejercicio A3.2 [I] — Tu figura de tendencia
#
# Armá la figura de línea temporal de **tu** proyecto, con estas tres capas:
#
# 1. El dato crudo, atenuado (`alpha` entre 0.2 y 0.3).
# 2. Una línea base: mediana móvil de una ventana que elijas y justifiques.
# 3. Una línea horizontal en tu umbral de decisión, con su rótulo.
#
# El título tiene que ser **la pregunta**, y el subtítulo **la respuesta**. Guardá
# la figura con `guardar(fig, "tendencia")`.

# %%
# TU CÓDIGO ACÁ
fig, ax = plt.subplots()

# ...

# %%
_ruta = Path("figuras") / "tendencia.png"
check("Guardaste la figura con guardar()", _ruta.exists(),
      "llamá a guardar(fig, 'tendencia') al final de la celda anterior")

# %% [markdown]
# ### Ejercicio A3.3 [I] — Desarmar el doble eje
#
# Te pasan este gráfico en una reunión. Reproducilo tal cual y después **armá la
# versión honesta** de los mismos datos (dos paneles apilados). Guardá esta
# segunda en `fig_honesta`.
#
# Después, en la celda de texto: ¿qué conclusión sugiere el primero y qué
# conclusión soporta el segundo?

# %%
_rng = np.random.default_rng(11)
_fechas = pd.date_range("2026-09-01", periods=45, freq="D")
_humedad = 55 - np.cumsum(_rng.normal(0.15, 0.6, 45))
_consumo = 60 + np.cumsum(_rng.normal(0.0, 3.0, 45))

# El gráfico sospechoso, para que lo veas:
_f, _a = plt.subplots(figsize=(8, 3.5))
_a.plot(_fechas, _humedad, color=SERIE[0])
_a.set_ylim(30, 60)
_g = _a.twinx()
_g.plot(_fechas, _consumo, color=SERIE[1])
_g.set_ylim(90, 40)          # ojo: el eje está INVERTIDO, además
_a.set_title("«Cuanto más baja la humedad, más sube el consumo»")
_a.grid(False)
plt.show()

# %%
# TU CÓDIGO ACÁ
fig_honesta = None

# %%
check("Armaste la figura honesta", fig_honesta is not None)
check("Tiene dos paneles",
      fig_honesta is not None and len(fig_honesta.axes) == 2,
      "usá plt.subplots(2, 1, sharex=True)")
_posiciones = ([tuple(a.get_position().bounds) for a in fig_honesta.axes]
               if fig_honesta is not None else [])
check("Ningún panel tiene eje gemelo",
      bool(_posiciones) and len(set(_posiciones)) == len(_posiciones),
      "un twinx() ocupa exactamente la misma posición que su panel: eso es lo que hay que evitar")
print(f"\n   Correlación real: r = {np.corrcoef(_humedad, _consumo)[0, 1]:.2f}")

# %% [markdown]
# **Tu respuesta:** *(doble clic para editar)*
#
# - El primer gráfico sugiere que…
# - El segundo soporta que…
# - El truco específico que usa el primero es…

# %% [markdown]
# ### Ejercicio A3.4 [A] — La conclusión de una línea
#
# Sin verificación automática: es la entrega del módulo F-3.
#
# Para **cada** figura que produjiste en A3.2 y A3.3, escribí abajo una
# conclusión de **una sola oración** que cumpla las tres condiciones:
#
# 1. Afirma algo sobre el **fenómeno**, no sobre el gráfico. (Mal: "se observa
#    una línea creciente". Bien: "el consumo crece unos 3 W por día".)
# 2. Incluye al menos **un número**.
# 3. Es algo que alguien podría **discutir** mirando los mismos datos.
#
# La tercera condición es la que más cuesta. Si tu conclusión no se puede
# discutir, probablemente no dijiste nada.

# %% [markdown]
# **Tus conclusiones:** *(doble clic para editar)*
#
# 1.
# 2.

# %% [markdown]
# ---
# ## Cierre del cuaderno A-3
#
# **Lo que quedó instalado en tu cabeza:**
#
# - El orden es: pregunta → forma → color. Nunca al revés.
# - Una línea base con mediana móvil separa lo normal de lo excepcional mejor
#   que un promedio.
# - Los tres engaños clásicos —eje truncado, doble eje, promedio que tapa el
#   pico— no requieren ningún dato falso.
# - Para dimensionar se usa el máximo; para costear, el promedio. La decisión
#   que el gráfico apoya determina el estadístico.
#
# **Checklist de entrega**
#
# - [ ] Tres preguntas propias con su forma justificada (A3.1).
# - [ ] La figura de tendencia guardada en `figuras/tendencia.png` (A3.2).
# - [ ] La versión honesta del gráfico de doble eje, con la explicación del
#       truco (A3.3).
# - [ ] Dos conclusiones de una línea que cumplan las tres condiciones (A3.4).
#
# **Sigue en:** `A4_Primer_modelo_supervisado.ipynb`
