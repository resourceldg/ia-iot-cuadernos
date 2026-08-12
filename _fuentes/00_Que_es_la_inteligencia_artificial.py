# %% [markdown]
# # Cuaderno 0 · ¿De qué hablamos cuando hablamos de inteligencia artificial?
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Arranque del cuatrimestre. Va antes que todo lo demás. |
# | **Duración** | 100 minutos |
# | **Modalidad** | En clase, con discusión |
# | **Requisitos** | Ninguno. Ni matemática, ni haber visto IA nunca. |
#
# ### Antes de arrancar
#
# Este cuaderno no tiene fórmulas ni te pide que programes casi nada. Es para
# contestar la pregunta que probablemente tengas dando vueltas: **¿qué es esto
# en realidad?**
#
# Vas a salir de acá sabiendo qué es la IA, qué no es, de dónde salió, y —lo más
# importante para la materia— **dónde entra en el nodo que estás armando**.
#
# Hay algo que conviene decir de entrada: **no hace falta que te guste la IA para
# esta materia.** Hace falta que la entiendas lo suficiente como para decidir
# cuándo usarla y, sobre todo, cuándo no. Eso es lo que hace un técnico.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from estilo_grafico import aplicar_estilo, titular, SERIE, ESTADO, TINTA_SUAVE

aplicar_estilo()


def check(descripcion, condicion, pista=""):
    if condicion:
        print(f"  [OK]     {descripcion}")
    else:
        print(f"  [REVISAR] {descripcion}" + (f"\n            Pista: {pista}" if pista else ""))
    return bool(condicion)


print("Listo. Ejecutá las celdas con Shift+Enter.")

# %% [markdown]
# ---
# ## Bloque 1 — Arranquemos con algo que ya sabés hacer (20 minutos)
#
# Antes de definir nada, hagamos un experimento.
#
# Tenés un sensor de humedad en una maceta y querés que el sistema decida solo
# cuándo regar. Sabés hacerlo: es un `if`. Escribilo.

# %%
# Diez mediciones y lo que un jardinero decidió en cada caso.
mediciones = pd.DataFrame({
    "humedad": [72, 65, 41, 28, 55, 19, 33, 80, 24, 48],
    "regar":   [ 0,  0,  0,  1,  0,  1,  0,  0,  1,  0],
})
mediciones.T

# %%
# TU TURNO: cambiá el número hasta que la regla acierte las diez.
UMBRAL = 50

prediccion = (mediciones["humedad"] < UMBRAL).astype(int)
aciertos = (prediccion == mediciones["regar"]).sum()
print(f"Con umbral {UMBRAL}: acertás {aciertos} de 10")

# %% [markdown]
# Fácil, ¿no? Con un umbral entre 29 y 32 acertás las diez. **Escribiste un
# sistema que decide, y no necesitaste ninguna inteligencia artificial.** Eso es
# importante: la enorme mayoría de los problemas se resuelven así, y está bien.
#
# Ahora complicamos apenas un poco. Resulta que el jardinero también mira la
# temperatura: con calor riega antes, porque el agua se evapora rápido.

# %%
mediciones2 = pd.DataFrame({
    "humedad":     [20, 22, 30, 38, 44, 60, 28, 33, 40, 44, 50, 62, 25, 48],
    "temperatura": [17, 19, 18, 20, 17, 19, 34, 33, 35, 33, 34, 35, 25, 25],
    "regar":       [ 1,  1,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  1,  0],
})
total = len(mediciones2)
objetivo = mediciones2["regar"]


def mejor_umbral(columna, desde, hasta):
    """Prueba TODOS los umbrales posibles sobre una sola columna."""
    opciones = []
    for u in range(desde, hasta):
        for sentido in ("<", ">"):
            pred = ((mediciones2[columna] < u) if sentido == "<"
                    else (mediciones2[columna] > u)).astype(int)
            opciones.append(((pred == objetivo).sum(), u, sentido))
    return max(opciones)


for columna, desde, hasta in [("humedad", 15, 90), ("temperatura", 15, 40)]:
    aciertos, u, sentido = mejor_umbral(columna, desde, hasta)
    print(f"El mejor umbral usando SOLO {columna:12s} es «{columna} {sentido} {u}» "
          f"y acierta {aciertos} de {total}.")

print(f"\nProbamos todos los umbrales posibles de las dos variables por separado.")
print(f"Ninguno llega a {total}.")

# %% [markdown]
# Y ahí está el punto. **El problema no es que seas mal programador: ninguna
# regla sobre una sola variable puede acertar todos los casos.** Fijate en las
# dos filas con humedad 44: en una hay que regar y en la otra no. Con el mismo
# número. Lo que las diferencia es la temperatura.
#
# Con dos variables todavía podrías encontrar la regla a mano, con paciencia. Con
# ocho, ni loco.
#
# Miremos por qué:

# %%
fig, ax = plt.subplots(figsize=(6.4, 5))
for valor, color, marca, etiqueta in [(0, SERIE[0], "o", "no regó"),
                                      (1, SERIE[1], "s", "regó")]:
    sub = mediciones2[mediciones2["regar"] == valor]
    ax.scatter(sub["humedad"], sub["temperatura"], s=140, color=color,
               marker=marca, label=etiqueta, zorder=3)
ax.set_xlabel("humedad del suelo (%)")
ax.set_ylabel("temperatura (°C)")
ax.grid(axis="both")
ax.legend()
titular(ax, "¿Podés separar los cuadrados de los círculos con UNA línea vertical?",
        "Una línea vertical es exactamente lo que hace un umbral sobre la humedad sola.")
plt.show()

# %% [markdown]
# Una línea vertical no alcanza: hay cuadrados y círculos a la misma altura de
# humedad. Lo que sí separa es una frontera **escalonada o inclinada**, que use
# las dos variables a la vez — algo como *"regar si la humedad está muy baja, o
# si está media pero hace calor"*.
#
# Y acá viene la idea central de toda la materia, en una oración:
#
# > **En vez de escribir vos la regla, le das a la computadora los ejemplos
# > resueltos y le pedís que encuentre ella la regla.**
#
# Eso es aprendizaje automático. Nada más. Miralo hacer:

# %%
from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(max_depth=3, random_state=238)
modelo.fit(mediciones2[["humedad", "temperatura"]], mediciones2["regar"])

aciertos = (modelo.predict(mediciones2[["humedad", "temperatura"]])
            == mediciones2["regar"]).sum()
print(f"La máquina acierta {aciertos} de {total}.\n")

from sklearn.tree import export_text
print("Y esta es la regla que encontró sola:")
print(export_text(modelo, feature_names=["humedad", "temperatura"], decimals=0))

# %% [markdown]
# **Leé esa regla.** No es magia, no es una caja negra, no es un cerebro. Son tres
# `if` anidados, del mismo tipo que vos escribiste al principio, solo que **los
# encontró probando combinaciones en vez de que los pensaras vos**.
#
# Y decodificada al castellano dice: *si la humedad está muy baja, regar; si no,
# regar sólo cuando haga calor y la humedad no esté alta.* Que es, palabra por
# palabra, lo que hace un jardinero con experiencia.
#
# Si algún día alguien te dice que la inteligencia artificial es incomprensible,
# acordate de esta celda.

# %% [markdown]
# ---
# ## Bloque 2 — Qué es la IA, y sobre todo qué no es (20 minutos)
#
# ### La definición
#
# > **Inteligencia artificial** es el conjunto de técnicas para que una máquina
# > haga tareas que, si las hiciera una persona, diríamos que requieren
# > inteligencia.
#
# Fijate lo raro de esa definición: **no habla de la máquina, habla de nosotros.**
# Define la IA por lo que a nosotros nos parece difícil. Y eso trae una
# consecuencia divertida:
#
# ### El efecto IA
#
# Cada vez que algo funciona bien, deja de llamarse inteligencia artificial y
# pasa a llamarse "el programa".
#
# | Cuando salió era "IA" | Hoy le decimos |
# |---|---|
# | El corrector ortográfico | el corrector |
# | El GPS que calcula la ruta más corta | el Waze |
# | El filtro de spam del correo | la carpeta de correo no deseado |
# | El reconocimiento de la patente en el peaje | la cámara del peaje |
# | El desbloqueo del celular con la cara | el desbloqueo |
#
# Todo eso fue IA de punta en su momento. **Nada de eso te parece inteligente
# hoy.** Dentro de diez años lo mismo va a pasar con lo que hoy nos asombra.
#
# ### Las muñecas rusas
#
# Los términos se usan como sinónimos y no lo son. Van uno adentro del otro:
#
# ```
# ┌─────────────────────────────────────────────────────────┐
# │ INTELIGENCIA ARTIFICIAL                                 │
# │ Cualquier técnica para que la máquina "decida bien".    │
# │ Incluye los sistemas de reglas de toda la vida.         │
# │                                                         │
# │   ┌───────────────────────────────────────────────┐     │
# │   │ APRENDIZAJE AUTOMÁTICO (machine learning)     │     │
# │   │ La máquina encuentra la regla a partir de     │     │
# │   │ ejemplos, en lugar de que vos la escribas.    │     │
# │   │                                               │     │
# │   │   ┌─────────────────────────────────────┐     │     │
# │   │   │ APRENDIZAJE PROFUNDO (deep learning)│     │     │
# │   │   │ Aprendizaje automático con redes    │     │     │
# │   │   │ neuronales de muchas capas.         │     │     │
# │   │   │                                     │     │     │
# │   │   │   ┌───────────────────────────┐     │     │     │
# │   │   │   │ MODELOS DE LENGUAJE       │     │     │     │
# │   │   │   │ ChatGPT y compañía.       │     │     │     │
# │   │   │   │ Un rincón de todo esto.   │     │     │     │
# │   │   │   └───────────────────────────┘     │     │     │
# │   │   └─────────────────────────────────────┘     │     │
# │   └───────────────────────────────────────────────┘     │
# └─────────────────────────────────────────────────────────┘
# ```
#
# El árbol que corriste recién es aprendizaje automático y **no** es aprendizaje
# profundo. Anda perfecto igual. La mayor parte de lo que vas a hacer en tu
# proyecto vive en los dos anillos de afuera.
#
# ### Cinco cosas que se dicen y no son
#
# | Se dice | Qué pasa en realidad |
# |---|---|
# | *"La IA piensa"* | Calcula. Encuentra patrones estadísticos en números. No hay nadie adentro. |
# | *"La IA entiende lo que le decís"* | Predice qué palabra sigue. Que el resultado tenga sentido no significa que haya comprensión. |
# | *"La IA es objetiva porque es matemática"* | Aprende de datos que juntó gente. Si los datos vienen torcidos, la salida sale torcida — **y encima con apariencia de neutral**. |
# | *"La IA es nueva"* | El primer paper es de 1950. El perceptrón, de 1958. Lo nuevo es que hay datos y placas de video baratas. |
# | *"La IA va a reemplazar a los técnicos"* | Nadie que sepa poner un sensor en una silobolsa a las 6 de la mañana está en riesgo. Lo que cambia son las herramientas, como cambió con el multímetro digital. |

# %% [markdown]
# ---
# ## Bloque 3 — Cinco momentos, para ubicarte (15 minutos)
#
# La historia importa por una razón práctica: **esto ya se sobrevendió dos veces,
# y las dos veces terminó mal.** Saberlo te da el escepticismo que necesitás para
# trabajar.
#
# ### 1950 — Turing cambia la pregunta
#
# Alan Turing publica *"Computing Machinery and Intelligence"*. Arranca
# preguntando "¿pueden pensar las máquinas?", se da cuenta de que la pregunta no
# se puede contestar, y la reemplaza por una que sí: **¿podría una máquina hacerse
# pasar por una persona en una conversación escrita?**
#
# Ese movimiento —cambiar una pregunta filosófica por una prueba medible— es el
# mismo que vas a hacer vos cuando en el cuaderno A-4 tengas que elegir una
# métrica. No es casualidad.
#
# ### 1956 — Le ponen el nombre
#
# En un taller de verano en Dartmouth, un grupo de investigadores acuña el
# término *artificial intelligence*. Calcularon que un grupo de diez personas
# podía avanzar mucho en dos meses. Estaban entusiasmados.
#
# ### 1958 — El perceptrón, y la primera vez que se les fue la mano
#
# Frank Rosenblatt construye el **perceptrón**: la primera neurona artificial que
# aprende de ejemplos. Es, literalmente, la misma idea que vas a programar en el
# cuaderno A-6.
#
# La prensa de la época anunció que esa máquina iba a poder caminar, hablar, ver,
# escribir y reproducirse. **No hizo nada de eso.**
#
# ### 1969 y el invierno
#
# Minsky y Papert publican un libro que demuestra que un perceptrón no puede
# resolver un problema tan pequeño como el "o exclusivo" (lo vas a ver, y a
# resolver, en A-6). La plata para investigar se cortó. A ese período se lo llama
# **el invierno de la IA**, y hubo dos.
#
# El deshielo llegó en 1986, cuando se popularizó la **retropropagación**: la
# forma de entrenar redes de varias capas. Que también vas a programar en A-6.
#
# ### 2012 — El cambio real
#
# Una red neuronal profunda gana por lejos un concurso de reconocimiento de
# imágenes. La técnica no era nueva. Lo nuevo eran dos cosas: **muchísimos datos
# etiquetados y placas de video para entrenar**.
#
# Esa es la lección que vale la pena llevarse: **lo que destrabó la IA moderna no
# fue una idea genial, fueron datos y fierros.** Y por eso el cuaderno A-2, que
# parece el más aburrido porque habla de limpiar datos, es en realidad el más
# importante.
#
# ### 2017 hasta hoy
#
# Aparece una arquitectura llamada *transformer* y, sobre ella, los modelos de
# lenguaje. En el cuaderno A-7 vas a entrenar uno propio, chiquito, en tu
# computadora, y vas a ver que hace exactamente lo mismo que los grandes.
#
# > **Moraleja de los 75 años:** cada vez que escuches "esto lo cambia todo",
# > preguntá qué métrica mejoró, cuánto, y comparado con qué. Es la misma
# > pregunta que te va a pedir esta materia en cada entrega.

# %% [markdown]
# ---
# ## Bloque 4 — El vocabulario mínimo (10 minutos)
#
# Quince palabras. Con estas quince entendés cualquier conversación técnica sobre
# el tema.

# %%
glosario = pd.DataFrame([
    ("dato", "Una medición. Un número con unidad y momento.",
     "una lectura de 43 % de humedad a las 14:30"),
    ("dataset / conjunto", "Muchos datos juntos y ordenados en una tabla.",
     "los 4320 registros del silobolsa"),
    ("característica (feature)", "Una columna que usás como entrada.",
     "humedad, temperatura, hora del día"),
    ("etiqueta (label)", "La respuesta correcta que querés predecir.",
     "¿hubo que regar? sí / no"),
    ("modelo", "La regla que la máquina encontró.",
     "el árbol que corriste en el Bloque 1"),
    ("entrenar", "El proceso de encontrar esa regla a partir de ejemplos.",
     "modelo.fit(...)"),
    ("predecir / inferir", "Usar el modelo ya entrenado con un caso nuevo.",
     "modelo.predict(...)"),
    ("clasificación", "Predecir una categoría.",
     "¿riesgo de fermentación? sí / no"),
    ("regresión", "Predecir un número.",
     "¿cuánta humedad va a haber en 3 horas?"),
    ("supervisado", "Aprender de ejemplos que traen la respuesta correcta.",
     "todo lo de esta materia"),
    ("baseline", "Lo más tonto que se puede hacer. Hay que ganarle.",
     "predecir que mañana pasa lo mismo que hoy"),
    ("métrica", "El número con el que decidís si funciona.",
     "de cada 10 alertas, cuántas eran de verdad"),
    ("sobreajuste", "El modelo memorizó los ejemplos en vez de aprender.",
     "anda perfecto con lo visto y falla con lo nuevo"),
    ("red neuronal", "Un modelo hecho de capas de sumas y funciones simples.",
     "lo que programás en A-6, en 30 líneas"),
    ("alucinación", "Cuando el modelo inventa algo verosímil y falso.",
     "un dato técnico inventado con total seguridad"),
], columns=["término", "en criollo", "ejemplo de esta materia"])

pd.set_option("display.max_colwidth", 60)
glosario

# %% [markdown]
# ---
# ## Bloque 5 — Dónde entra la IA en tu nodo (15 minutos)
#
# Esta es la parte que más te sirve para el proyecto. Hay **cuatro lugares**
# donde podría entrar, y son muy distintos entre sí.

# %%
lugares = pd.DataFrame([
    ("1. En el sensor", "Filtrar ruido, descartar lecturas imposibles",
     "Casi nunca hace falta IA. Un promedio móvil y un rango alcanzan.", "A-2"),
    ("2. En el nodo", "Decidir si actúa o alerta",
     "Reglas, casi siempre. Un ESP32 no corre un modelo.", "A-1, A-5"),
    ("3. En el servidor", "Buscar patrones en el histórico",
     "Acá sí, si tenés meses de datos y el patrón no es obvio.", "A-4, A-5"),
    ("4. En la interfaz", "Redactar informes, traducir, explicar",
     "Un modelo de lenguaje sirve. Nunca como fuente de datos técnicos.", "A-7, A-8"),
], columns=["dónde", "para qué", "¿conviene IA?", "cuaderno"])
lugares

# %% [markdown]
# ### Ojo con el lugar 2, que es donde todos quieren ponerla
#
# Es tentador decir "le pongo IA al nodo". El problema es físico: un ESP32 tiene
# unos cientos de kilobytes de memoria. El modelo de lenguaje más chico que vas a
# ver en el cuaderno A-8 ocupa **538 megabytes**. Es unas mil veces más de lo que
# entra.
#
# Eso no significa que no haya nada que hacer. Significa que la jugada correcta
# es otra, y es la que trabaja el cuaderno A-5: **entrenás el modelo en la
# computadora, le leés la regla que encontró, y esa regla —que son cuatro
# líneas— la escribís en el firmware.** El nodo termina siendo simple, rápido y
# entendible, pero la regla que ejecuta salió de mirar meses de datos.
#
# Esa es, probablemente, la idea más útil de todo el anexo para tu carrera.

# %% [markdown]
# ---
# ## Bloque 6 — Lo que la IA no puede, demostrado (15 minutos)
#
# No te lo voy a pedir de fe. Vamos a construir el problema a propósito y ver qué
# pasa.
#
# **La situación:** queremos detectar si una planta necesita agua. Juntamos los
# datos en dos tandas: las plantas sanas las medimos un lunes a la mañana, y las
# plantas secas las medimos un jueves a la tarde. Sin querer, el número de día de
# la semana quedó en la tabla.

# %%
rng = np.random.default_rng(238)
n = 200

datos = pd.DataFrame({
    "humedad": np.concatenate([rng.normal(60, 8, n), rng.normal(25, 8, n)]),
    "temperatura": np.concatenate([rng.normal(22, 3, n), rng.normal(23, 3, n)]),
    # La columna traicionera: quedó ahí sin que nadie lo pensara.
    "dia_de_medicion": np.concatenate([np.ones(n), np.full(n, 4)]),
    "necesita_agua": np.concatenate([np.zeros(n), np.ones(n)]).astype(int),
})

modelo_trampa = DecisionTreeClassifier(max_depth=2, random_state=238)
X = datos[["humedad", "temperatura", "dia_de_medicion"]]
modelo_trampa.fit(X, datos["necesita_agua"])

print(f"Acierto: {(modelo_trampa.predict(X) == datos['necesita_agua']).mean():.1%}  "
      f"— ¡perfecto!\n")
print("Pero mirá en qué se apoyó:")
print(export_text(modelo_trampa,
                  feature_names=["humedad", "temperatura", "dia_de_medicion"],
                  decimals=0))

# %% [markdown]
# **El modelo aprendió a mirar el calendario.**
#
# Con 100 % de acierto, en un informe eso se vería espectacular. Y es basura: el
# día de la semana no tiene absolutamente nada que ver con la sed de una planta.
# El día que lo pongas a andar un martes, no va a funcionar.
#
# Esto tiene nombre: el modelo encontró un **atajo**. No aprendió el fenómeno,
# aprendió una casualidad de cómo se juntaron los datos.
#
# > **Y acá está el punto que quiero que te lleves de este cuaderno entero:** la
# > máquina no tiene forma de saber que el día de la semana es una tontería. Vos
# > sí. **La máquina encuentra correlaciones; el criterio para saber cuáles tienen
# > sentido lo ponés vos.** Ese criterio es tu trabajo, y no se automatiza.
#
# Por eso en el cuaderno A-4 vas a leer siempre las reglas que aprendió el
# modelo. No es un ejercicio de curiosidad: es control de calidad.

# %% [markdown]
# ### Las otras tres limitaciones, en corto
#
# | Limitación | Qué significa para tu proyecto |
# |---|---|
# | **Aprende lo que hay en los datos, incluidos los prejuicios** | Si tus datos vienen de una sola maceta, en un solo mes, el modelo sabe de esa maceta en ese mes. Nada más. |
# | **No sabe cuándo no sabe** | Un modelo contesta siempre, con la misma cara, aunque le des un caso que nunca vio. No te avisa. |
# | **No se hace responsable** | Si tu sistema riega de más y se pudre la planta, el responsable sos vos. "Lo decidió el modelo" no es una respuesta técnica ni legal. |

# %% [markdown]
# ---
# ## Bloque 7 — Ejercicios

# %% [markdown]
# ### Ejercicio 0.1 [B] — Tu propia definición
#
# Escribí en `mi_definicion` qué es la inteligencia artificial, **con tus
# palabras**, en no más de dos oraciones. No copies la del cuaderno.
#
# Tiene que poder entenderla alguien de tu familia que no sepa nada de esto.

# %%
# TU CÓDIGO ACÁ
mi_definicion = ""

# %%
_prohibidas = ["conjunto de técnicas para que una máquina haga tareas"]
check("Escribiste una definición", len(mi_definicion.split()) >= 15)
check("No copiaste la del cuaderno",
      all(p not in mi_definicion.lower() for p in _prohibidas))
check("No usaste la palabra 'piensa' ni 'entiende'",
      "piensa" not in mi_definicion.lower() and "entiende" not in mi_definicion.lower(),
      "son justo las dos palabras que el Bloque 2 desarma")

# %% [markdown]
# ### Ejercicio 0.2 [B] — ¿Reglas o aprendizaje?
#
# Para cada sistema, decidí si por dentro es más probable que use **reglas
# escritas a mano** (`"reglas"`) o **aprendizaje a partir de ejemplos**
# (`"aprendizaje"`).
#
# No hay trampa: pensá si un programador podría escribir todos los casos.

# %%
# TU CÓDIGO ACÁ
sistemas = {
    "El semáforo de tu esquina": "",
    "El corrector del teclado del celular": "",
    "La alarma de humo de tu casa": "",
    "El reconocimiento facial que desbloquea el celular": "",
    "El termostato de un aire acondicionado": "",
    "La sección 'quizás conozcas a' de una red social": "",
}

# %%
_correctas = ["reglas", "aprendizaje", "reglas", "aprendizaje", "reglas", "aprendizaje"]
_dadas = [v.strip().lower() for v in sistemas.values()]
if check("Las seis están bien clasificadas", _dadas == _correctas,
         "¿podría un programador escribir TODOS los casos a mano?"):
    print("\n   La pauta: si el fenómeno se puede describir con pocas condiciones")
    print("   claras (hay humo / hace calor / está en rojo), son reglas. Si hay")
    print("   millones de casos posibles y ninguno igual al otro (una cara, una")
    print("   palabra mal escrita), hace falta aprendizaje.")
else:
    for (nombre, dada), correcta in zip(sistemas.items(), _correctas):
        if dada != correcta:
            print(f"      revisá: {nombre}")

# %% [markdown]
# ### Ejercicio 0.3 [I] — Cazá el atajo
#
# Igual que el ejemplo de la planta y el día de la semana, en el conjunto de abajo
# hay **una característica traicionera**: una columna que predice perfecto por una
# casualidad de cómo se juntaron los datos, y que no serviría en la realidad.
#
# Entrená un árbol, leé sus reglas, y guardá el nombre de la columna culpable en
# `columna_traicionera`.

# %%
rng2 = np.random.default_rng(7)
m = 150
maquinas = pd.DataFrame({
    "vibracion_mm_s": np.concatenate([rng2.normal(2.0, 0.6, m), rng2.normal(4.5, 1.2, m)]),
    "temperatura_C":  np.concatenate([rng2.normal(45, 6, m), rng2.normal(52, 8, m)]),
    "horas_de_uso":   np.concatenate([rng2.normal(1200, 300, m), rng2.normal(1400, 350, m)]),
    "id_de_sensor":   np.concatenate([np.full(m, 17), np.full(m, 23)]),
    "va_a_fallar":    np.concatenate([np.zeros(m), np.ones(m)]).astype(int),
})

# TU CÓDIGO ACÁ
columna_traicionera = ""

# %%
check("Encontraste la columna traicionera",
      columna_traicionera.strip() == "id_de_sensor",
      "entrená un árbol con export_text y fijate por cuál columna corta primero")
if columna_traicionera.strip() == "id_de_sensor":
    print("\n   Las máquinas sanas se midieron con el sensor 17 y las falladas con")
    print("   el 23. El modelo aprendió a leer el número de serie del sensor.")
    print("   Con 100 % de acierto, y completamente inútil.")

# %% [markdown]
# ### Ejercicio 0.4 [I] — Tu nodo y los cuatro lugares
#
# Volvé a la tabla del Bloque 5. Para **tu** proyecto, completá:

# %%
# TU CÓDIGO ACÁ
mi_nodo = {
    "que_mide": "",
    "que_decide_hoy": "",
    "lugar_donde_podria_entrar_ia": 0,   # 1, 2, 3 o 4
    "hace_falta": "",                    # "si", "no" o "todavia no se"
    "por_que": "",
}

# %%
check("Describiste qué mide tu nodo", len(mi_nodo["que_mide"].split()) >= 5)
check("Describiste qué decide hoy", len(mi_nodo["que_decide_hoy"].split()) >= 5)
check("Elegiste uno de los cuatro lugares", mi_nodo["lugar_donde_podria_entrar_ia"] in (1, 2, 3, 4))
check("Contestaste si hace falta",
      mi_nodo["hace_falta"].strip().lower() in ("si", "no", "todavia no se"))
check("Justificaste en al menos dos oraciones", mi_nodo["por_que"].count(".") >= 2)

if mi_nodo["hace_falta"].strip().lower() == "no":
    print("\n   Contestar que NO hace falta es una respuesta correcta y valiente.")
    print("   La mayoría de los proyectos de este nivel se resuelven con reglas,")
    print("   y saber demostrarlo con números es exactamente lo que pide el")
    print("   módulo G-2. Guardá esta respuesta: la vas a citar en el cuaderno A-5.")

# %% [markdown]
# ### Ejercicio 0.5 [A] — El mito que se te cayó
#
# Sin verificación automática.
#
# De las cinco frases del Bloque 2 que "se dicen y no son", **elegí la que vos
# creías** y escribí abajo:
#
# 1. Cuál era y de dónde te la habías llevado (una película, un video, alguien
#    que la dijo con seguridad).
# 2. Qué te hizo cambiar de idea.
# 3. Una cosa que ahora mirarías distinto cuando alguien te venda "una solución
#    con inteligencia artificial".
#
# Si sentís que no creías ninguna, entonces escribí sobre algo que **sigas sin
# tener claro**. Esa respuesta vale igual, y es más útil para la clase.

# %% [markdown]
# **Tu respuesta:** *(doble clic acá para escribir)*
#
# 1.
# 2.
# 3.

# %% [markdown]
# ---
# ## Cierre del Cuaderno 0
#
# **Las cinco cosas que te tenés que llevar:**
#
# 1. **Aprendizaje automático es que la máquina encuentre la regla en vez de que
#    la escribas vos.** Lo viste pasar en el Bloque 1: dos `if`, encontrados solos.
# 2. **No hay nadie adentro.** Calcula, encuentra patrones. Que el resultado
#    parezca inteligente no cambia lo que está pasando.
# 3. **Lo que destrabó la IA moderna fueron datos y placas de video**, no una idea
#    genial. Por eso el cuaderno de limpiar datos es el más importante.
# 4. **La máquina encuentra correlaciones; el criterio lo ponés vos.** El modelo
#    que miraba el día de la semana acertaba el 100 %.
# 5. **En un nodo IoT casi nunca va un modelo.** Va una regla — que puede haber
#    salido de mirar un modelo.
#
# **Checklist de entrega**
#
# - [ ] Tu definición propia (0.1).
# - [ ] Los seis sistemas clasificados (0.2).
# - [ ] La columna traicionera encontrada (0.3).
# - [ ] Tu nodo ubicado en uno de los cuatro lugares, con justificación (0.4).
# - [ ] El texto sobre el mito que se te cayó (0.5).
#
# **Sigue en:** `01_Como_leer_un_grafico.ipynb` — porque de acá en adelante casi
# todo lo que tengas que decidir lo vas a decidir mirando un gráfico, y a leerlos
# no nos enseñó nadie.
