# %% [markdown]
# # Anexo A-8 · Un modelo de lenguaje real, corriendo en tu máquina *(opcional)*
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Ampliación de A-7. No es obligatorio. |
# | **Duración** | 90 minutos |
# | **Requisitos** | Conexión a internet **una vez** (≈ 270 MB de descarga) y unos 2 GB de RAM libres |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Descargar y ejecutar un modelo de lenguaje real **enteramente en tu
#    computadora**, sin cuenta, sin API y sin mandar un solo dato afuera.
# 2. Reconocer en él las mismas piezas que programaste en A-7: tokenizador,
#    contexto, temperatura, muestreo.
# 3. Medir su velocidad y su consumo, y sacar conclusiones sobre dónde puede y
#    dónde no puede vivir un modelo así.
# 4. **Verificar experimentalmente que alucina**, y explicar por qué eso no es una
#    falla sino el mecanismo.
#
# ### Antes de arrancar: por qué esto es opcional
#
# Este es el único cuaderno del anexo que necesita descargar algo. Si en el aula
# no hay conexión o las máquinas no dan, se puede saltear sin perder continuidad:
# todo lo conceptual ya está en A-7.
#
# El modelo que vamos a usar es **SmolLM2-135M-Instruct**, de HuggingFace: 135
# millones de parámetros, unos 270 MB en disco. Es de los más chicos que existen
# entre los que saben seguir instrucciones. Elegimos justo ese, y no uno bueno, a
# propósito: **con un modelo chico los defectos se ven, y los defectos son lo que
# hay que aprender a ver.**

# %% [markdown]
# ---
# ## Bloque 1 — Instalación (15 minutos)
#
# Hacen falta dos bibliotecas que no vienen con el resto del anexo. Si no las
# tenés, corré esta celda una sola vez. Puede tardar varios minutos.
#
# La versión de PyTorch para CPU es bastante más liviana que la de GPU y alcanza
# de sobra para un modelo de este tamaño.

# %%
def check(descripcion, condicion, pista=""):
    if condicion:
        print(f"  [OK]     {descripcion}")
    else:
        print(f"  [REVISAR] {descripcion}" + (f"\n            Pista: {pista}" if pista else ""))
    return bool(condicion)


# Descomentá y ejecutá SOLO si te falta alguna de las dos.
# import sys
# !{sys.executable} -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# !{sys.executable} -m pip install transformers

try:
    import torch
    import transformers
    print(f"torch        {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"GPU disponible: {torch.cuda.is_available()}")
    LISTO = True
except ImportError as e:
    print(f"Falta instalar: {e.name}")
    print("Descomentá las líneas de pip de arriba y ejecutá esta celda.")
    LISTO = False

# %% [markdown]
# ---
# ## Bloque 2 — Cargar el modelo (15 minutos)
#
# La primera vez esto descarga el modelo y lo guarda en una caché local
# (`~/.cache/huggingface`). Las veces siguientes carga de disco y **ya no
# necesita internet**.

# %%
import time

MODELO = "HuggingFaceTB/SmolLM2-135M-Instruct"

if LISTO:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    inicio = time.time()
    tokenizador = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32)
    modelo.eval()

    parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Cargado en {time.time() - inicio:.1f} s")
    print(f"Parámetros:        {parametros:,}")
    print(f"Vocabulario:       {len(tokenizador):,} tokens")
    print(f"Contexto máximo:   {modelo.config.max_position_embeddings:,} tokens")
    print(f"Memoria del modelo: {parametros * 4 / 1e6:.0f} MB en float32")

# %% [markdown]
# ### Comparalo con el de A-7
#
# | | A-7 (tuyo) | A-8 (este) | Un modelo comercial |
# |---|---|---|---|
# | Parámetros | ~1 600 | ~135 000 000 | ~500 000 000 000 |
# | Vocabulario | 27 caracteres | ~49 000 subpalabras | ~200 000 subpalabras |
# | Contexto | 3 caracteres | 8 192 tokens | 200 000 tokens |
# | Arquitectura | 1 capa oculta | transformer, 30 capas | transformer, cientos de capas |
#
# El de este cuaderno tiene **ochenta mil veces** más parámetros que el tuyo, y
# aun así es unas **cuatro mil veces** más chico que uno comercial. Está mucho más
# cerca del tuyo que del de ellos.

# %% [markdown]
# ---
# ## Bloque 3 — El tokenizador (20 minutos)
#
# En A-7 cada carácter era un token. Los modelos reales usan **subpalabras**:
# trozos aprendidos estadísticamente del corpus de entrenamiento. Miremos qué
# hace con castellano técnico.

# %%
if LISTO:
    frases = [
        "El sensor de humedad del suelo mide",
        "The soil moisture sensor measures",
        "microcontrolador",
        "silobolsa",
        "co2_ppm > 1000",
    ]
    for frase in frases:
        ids = tokenizador(frase)["input_ids"]
        piezas = [tokenizador.decode([i]) for i in ids]
        print(f"{len(ids):3d} tokens  {frase!r}")
        print(f"            {' | '.join(piezas)}\n")

# %% [markdown]
# ### Mirá con atención la comparación entre las dos primeras frases
#
# Dicen lo mismo, y **la versión en castellano usa más tokens que la inglesa**.
# La razón es que el corpus de entrenamiento del tokenizador es mayoritariamente
# en inglés, así que las palabras inglesas frecuentes tienen un token propio
# mientras que las castellanas se parten en pedazos.
#
# Eso tiene tres consecuencias muy concretas, y ninguna es teórica:
#
# 1. **Cuesta más caro.** Los servicios de IA cobran por token. Un texto en
#    castellano puede salir bastante más que el mismo texto en inglés.
# 2. **Entra menos texto.** Si el contexto es de 8192 tokens, en castellano entra
#    menos documento que en inglés.
# 3. **Anda peor.** Menos tokens dedicados a un idioma es, en general, menos
#    capacidad para ese idioma.
#
# > Esto se llama **sesgo de tokenización**, y es una de las formas más
# > silenciosas de desigualdad en la infraestructura de IA. No aparece en ninguna
# > interfaz, pero está ahí en cada consulta que hacés en castellano.

# %% [markdown]
# ---
# ## Bloque 4 — Generar, y medir cuánto tarda (20 minutos)
#
# El modelo es *instruct*: fue ajustado para seguir instrucciones en formato de
# conversación. Ese formato se arma con `apply_chat_template`.

# %%
def responder(pregunta, max_tokens=100, temperatura=0.7, semilla=238):
    """Le hace una pregunta al modelo y devuelve la respuesta y las métricas."""
    torch.manual_seed(semilla)
    mensajes = [{"role": "user", "content": pregunta}]
    texto = tokenizador.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True)
    entrada = tokenizador(texto, return_tensors="pt")

    inicio = time.time()
    with torch.no_grad():
        salida = modelo.generate(
            **entrada, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperatura, top_p=0.9,
            pad_token_id=tokenizador.eos_token_id)
    duracion = time.time() - inicio

    generados = salida[0][entrada["input_ids"].shape[1]:]
    respuesta = tokenizador.decode(generados, skip_special_tokens=True)
    return respuesta, {"tokens": len(generados), "segundos": round(duracion, 1),
                       "tokens_por_segundo": round(len(generados) / duracion, 1)}


if LISTO:
    respuesta, metricas = responder("Explain in one sentence what a soil moisture sensor does.")
    print(respuesta)
    print(f"\n{metricas}")

# %% [markdown]
# ### La velocidad importa más de lo que parece
#
# Anotá los tokens por segundo que te dio. Ese número decide qué se puede
# construir con esto:
#
# | Velocidad | Qué se puede hacer |
# |---|---|
# | > 20 tok/s | Conversación fluida |
# | 5 – 20 tok/s | Usable con paciencia; sirve para procesamiento por lotes |
# | < 5 tok/s | Solo procesamiento en segundo plano, nunca interactivo |
#
# Y ojo con el contexto: **este modelo no va a correr en un ESP32 ni en una
# Raspberry Pi Zero**. 135 millones de parámetros en float32 son unos 540 MB solo
# para los pesos. Una Raspberry Pi 4 o 5 con 4 GB puede; un microcontrolador,
# jamás. Esa es la razón por la que el cuaderno A-5 insiste tanto en derivar
# reglas simples para el nodo.

# %% [markdown]
# ---
# ## Bloque 5 — Verificar la alucinación (20 minutos)
#
# En A-7 vimos alucinaciones a escala de palabras inventadas. Acá las vamos a ver
# en su forma peligrosa: **oraciones bien formadas, con tono seguro, y falsas.**
#
# Le vamos a preguntar cosas técnicas y verificables de nuestro propio dominio.

# %%
if LISTO:
    preguntas = [
        "In one sentence: what does an MQ-135 sensor measure?",
        "In one sentence: what is the MQTT protocol used for?",
        "En una oracion: que mide un sensor de humedad de suelo?",
        "In one sentence: what is the maximum current of an ESP32 GPIO pin?",
    ]
    for pregunta in preguntas:
        respuesta, metricas = responder(pregunta, max_tokens=60)
        print(f"P: {pregunta}")
        print(f"R: {respuesta.strip()}")
        print(f"   ({metricas['tokens']} tokens en {metricas['segundos']} s)\n")

# %% [markdown]
# ### Ahora leelas como técnico, no como usuario
#
# Tomate el trabajo de **verificar cada respuesta** contra lo que sabés y contra
# los datasheets. Es muy probable que encuentres:
#
# - Respuestas correctas en lo general y erradas en el detalle numérico.
# - Respuestas que mezclan dos conceptos parecidos.
# - Respuestas en inglés a preguntas en castellano.
# - Respuestas que suenan perfectas y son directamente falsas.
#
# Todas están escritas con exactamente el mismo tono de seguridad. **No hay
# ninguna señal en el texto que distinga una respuesta correcta de una inventada**,
# porque para el modelo son la misma operación: elegir tokens probables.
#
# > Este es el ejercicio más importante del cuaderno, y por eso lo hacemos con un
# > modelo chico: acá el error salta a la vista. En un modelo comercial el mismo
# > mecanismo produce errores mucho más raros y mucho más difíciles de detectar,
# > que es exactamente lo que los vuelve peligrosos.
# >
# > La regla profesional es simple: **un modelo de lenguaje sirve para redactar,
# > reformular, resumir y traducir texto que vos ya tenés; no sirve como fuente de
# > datos técnicos.** Para eso están los datasheets.

# %% [markdown]
# ---
# ## Bloque 6 — Para qué SÍ sirve en un proyecto de IoT (10 minutos)
#
# Después de todo lo anterior, la pregunta razonable es para qué usarlo. Hay usos
# buenos, y tienen una característica en común: **el modelo transforma texto que
# ya existe, en lugar de inventarlo.**

# %%
if LISTO:
    resumen_de_datos = """
Sensor: CO2 en silobolsa. Periodo: 90 dias, muestreo cada 30 minutos.
Lecturas totales: 4320. Descartadas por invalidas: 147 (3.4%).
Linea base: 450 ppm. Episodios sobre 1000 ppm: 3.
Regla de alerta: CO2 > 872 ppm o (CO2 > 750 y subida en 24h > 171 ppm).
Resultado sobre datos de prueba: sensibilidad 0.975, precision 0.927.
"""
    pedido = ("Rewrite the following technical notes as a short formal paragraph "
              "for a report. Do not add any information that is not present.\n"
              + resumen_de_datos)
    respuesta, metricas = responder(pedido, max_tokens=140, temperatura=0.3)
    print(respuesta.strip())
    print(f"\n{metricas}")

# %% [markdown]
# ### La diferencia clave, y el límite que acabás de ver
#
# En este pedido el modelo **no tiene que saber nada**: toda la información está
# en el texto que le dimos. Su trabajo es reordenarla y redactarla. Ahí es donde
# un modelo de lenguaje es genuinamente útil y verificable, porque podés
# comprobar línea por línea que no agregó nada.
#
# Ahora bien, mirá qué te devolvió. Con toda probabilidad **no reescribió nada**:
# copió las notas casi textuales, quizás les puso un encabezado, y se cortó a
# mitad de camino. Reformular un texto conservando la información es una tarea
# más difícil de lo que parece, y 135 millones de parámetros no alcanzan.
#
# Eso también es un resultado, y de los útiles: **la tarea es la correcta, el
# modelo es demasiado chico para ella.** Con un modelo de 3000 millones de
# parámetros el mismo pedido sale bien. La conclusión no es "los modelos no
# sirven para redactar", es "elegí el tamaño según la tarea, y comprobá el
# resultado en lugar de suponerlo".
#
# **Verificar no es opcional**, es parte del uso.
#
# | Uso | ¿Sirve? | Por qué |
# |---|---|---|
# | Redactar el informe a partir de tus notas | Sí | La información la ponés vos |
# | Traducir la documentación al inglés | Sí | Transforma texto existente |
# | Sugerir nombres de variables y funciones | Sí | Bajo riesgo si te equivocás |
# | Explicarte un mensaje de error | Con cuidado | Verificalo contra la documentación |
# | Decirte el pinout del ESP32 | **No** | Para eso está el datasheet |
# | Decidir el umbral de tu alerta | **No** | Para eso están tus datos y el cuaderno A-5 |
# | Correr adentro del nodo | **No** | No entra, ni cerca |

# %% [markdown]
# ---
# ## Bloque 7 — Ejercicios

# %% [markdown]
# ### Ejercicio A8.1 [B] — Medir el sesgo de tokenización
#
# Tomá cinco términos técnicos de tu proyecto y sus equivalentes en inglés.
# Contá los tokens de cada uno y guardá en `sesgo` un diccionario
# `{"castellano": total_tokens_es, "ingles": total_tokens_en}`.

# %%
# TU CÓDIGO ACÁ
terminos = [
    # ("humedad de suelo", "soil moisture"),
]
sesgo = {}

# %%
check("Elegiste al menos cinco pares", len(terminos) >= 5)
check("Contaste los tokens de los dos idiomas", set(sesgo) == {"castellano", "ingles"})
if set(sesgo) == {"castellano", "ingles"} and sesgo["ingles"]:
    print(f"\n   castellano: {sesgo['castellano']} tokens")
    print(f"   inglés:     {sesgo['ingles']} tokens")
    print(f"   El castellano cuesta un {(sesgo['castellano'] / sesgo['ingles'] - 1) * 100:.0f} % más.")

# %% [markdown]
# ### Ejercicio A8.2 [I] — Cazar alucinaciones, sistemáticamente
#
# Preparte **ocho preguntas técnicas de tu proyecto cuya respuesta correcta
# conozcas** (del datasheet, del apunte, de tu propia medición). Hacéselas al
# modelo y clasificá cada respuesta:
#
# - `"correcta"`
# - `"parcial"` (bien en general, mal en el detalle)
# - `"falsa"`
#
# Guardá el resultado en `auditoria`: una lista de diccionarios con las claves
# `pregunta`, `respuesta_del_modelo`, `respuesta_correcta` y `veredicto`.

# %%
# TU CÓDIGO ACÁ
auditoria = []

# %%
check("Auditaste al menos ocho preguntas", len(auditoria) >= 8)
_claves = {"pregunta", "respuesta_del_modelo", "respuesta_correcta", "veredicto"}
check("Cada entrada tiene las cuatro claves",
      bool(auditoria) and all(set(a) == _claves for a in auditoria))
if auditoria and all(set(a) == _claves for a in auditoria):
    import pandas as pd
    _cuenta = pd.Series([a["veredicto"] for a in auditoria]).value_counts()
    print("\n   Veredictos:", _cuenta.to_dict())
    _tasa = _cuenta.get("correcta", 0) / len(auditoria)
    print(f"   Tasa de acierto: {_tasa:.0%}")
    print("\n   Pregunta para el informe: con esta tasa de acierto, ¿usarías este")
    print("   modelo como fuente en tu documentación técnica?")

# %% [markdown]
# ### Ejercicio A8.3 [I] — La temperatura, otra vez
#
# En A-7 la temperatura controlaba la invención de palabras. Acá controla la
# invención de hechos.
#
# Hacé **la misma pregunta técnica** con temperaturas `0.1`, `0.7` y `1.5`,
# cinco veces cada una. Guardá en `variabilidad` un diccionario
# `{temperatura: cantidad_de_respuestas_distintas}`.
#
# Después contestá: si un sistema tiene que dar siempre la misma respuesta a la
# misma pregunta, ¿qué temperatura corresponde? ¿Y eso garantiza que la respuesta
# sea correcta?

# %%
# TU CÓDIGO ACÁ
variabilidad = {}

# %%
check("Probaste las tres temperaturas", set(variabilidad) == {0.1, 0.7, 1.5})
if set(variabilidad) == {0.1, 0.7, 1.5}:
    check("A mayor temperatura, más variabilidad",
          variabilidad[1.5] >= variabilidad[0.1])
    for t in sorted(variabilidad):
        print(f"   T = {t}: {variabilidad[t]} respuestas distintas de 5")

# %% [markdown]
# **Tu respuesta:** *(doble clic para editar)*
#
# - Para respuestas reproducibles corresponde una temperatura…
# - Pero eso **no** garantiza que sean correctas, porque…

# %% [markdown]
# ### Ejercicio A8.4 [A] — La política de uso de tu equipo
#
# Sin verificación automática.
#
# Escribí, en no más de media carilla, la **política de uso de asistentes de IA**
# de tu equipo para el proyecto integrador. Tiene que decir:
#
# 1. **Para qué SÍ** lo van a usar, con ejemplos concretos del proyecto.
# 2. **Para qué NO**, con el motivo técnico (no "porque está prohibido").
# 3. **Cómo verifican** lo que el modelo produce, antes de incorporarlo.
# 4. **Cómo lo declaran** en el informe final. Un texto asistido por IA que se
#    presenta como propio es un problema de honestidad; uno declarado y verificado
#    no lo es.
# 5. **Qué datos del proyecto nunca salen del equipo**, y por qué. Acá enganchá
#    con lo que trabajaste en F-4 sobre privacidad del dato: mandar telemetría a
#    un servicio externo es una decisión con consecuencias, no un detalle
#    operativo.
#
# El punto 5 es el que conecta este cuaderno con el resto de la materia. Un modelo
# local como el de acá tiene una ventaja que ningún servicio en la nube puede
# igualar: **los datos no se van a ningún lado.**

# %% [markdown]
# **La política de tu equipo:** *(doble clic para editar)*
#
# **1. Para qué sí:**
#
# **2. Para qué no:**
#
# **3. Cómo verificamos:**
#
# **4. Cómo lo declaramos:**
#
# **5. Datos que no salen del equipo:**

# %% [markdown]
# ---
# ## Si querés ir más lejos
#
# | Herramienta | Para qué |
# |---|---|
# | **Ollama** (`ollama.com`) | La forma más simple de correr modelos locales. Un comando y anda. |
# | **llama.cpp** | Motor en C++, muy eficiente en CPU. Es lo que usa Ollama por debajo. |
# | **Formato GGUF** | Modelos cuantizados: los pesos pasan de 32 a 4 bits. Ocupan hasta 8 veces menos y andan bastante más rápido, con una pérdida de calidad chica. |
# | **Qwen2.5-0.5B-Instruct** | El escalón siguiente a este: cuatro veces más grande, bastante mejor en castellano. |
#
# La **cuantización** merece un párrafo porque es el concepto que hace viable
# todo esto. Un peso guardado en 32 bits se puede guardar en 4 bits con una
# pérdida de precisión que, empíricamente, casi no afecta la calidad de las
# respuestas. Ese solo cambio es lo que permite que modelos que necesitaban un
# servidor entren hoy en una notebook.

# %% [markdown]
# ---
# ## Cierre del cuaderno A-8
#
# **Lo que quedó instalado en tu cabeza:**
#
# - Un modelo de lenguaje real corre en una notebook común, sin cuenta y sin
#   mandar datos a ningún lado.
# - Adentro tiene exactamente las mismas piezas que el de A-7. Lo que cambia es la
#   escala, la arquitectura y el ajuste posterior.
# - El castellano cuesta más tokens que el inglés. Eso es plata, contexto y
#   calidad, y no aparece en ninguna interfaz.
# - Las respuestas correctas y las inventadas se escriben con el mismo tono de
#   seguridad. No hay señal en el texto que las distinga.
# - Para un nodo IoT, un modelo así no entra. Para redactar el informe del nodo,
#   sirve.
#
# **Checklist de entrega**
#
# - [ ] La medición del sesgo de tokenización (A8.1).
# - [ ] La auditoría de ocho preguntas con su tasa de acierto (A8.2).
# - [ ] La tabla de variabilidad por temperatura (A8.3).
# - [ ] La política de uso de IA de tu equipo, con los cinco puntos (A8.4).
#
# **Con esto cerrás el anexo.** Volvé al cuadernillo del Trayecto F: ya tenés
# todas las piezas para armar el informe del módulo F-5.
