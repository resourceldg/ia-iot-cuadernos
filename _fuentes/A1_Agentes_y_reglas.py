# %% [markdown]
# # Anexo A-1 · Agentes inteligentes y sistemas basados en reglas
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Módulo **F-0** del cuadernillo · Bloques 1 y 2 del programa anual |
# | **Duración** | 120 minutos |
# | **Modalidad** | En equipo, sobre el proyecto propio |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Escribir la especificación **PEAS** de tu nodo y clasificarlo entre los
#    cinco tipos de agente de Russell y Norvig, con argumento.
# 2. Implementar un agente reactivo simple y uno basado en modelo, y **mostrar
#    con datos** por qué el segundo es mejor en tu proyecto.
# 3. Escribir un motor de inferencia de reglas IF-THEN con encadenamiento hacia
#    adelante, y usarlo para diagnosticar fallas de un nodo IoT.
# 4. Explicar por qué un sistema de reglas es *explicable* y en qué se diferencia
#    eso de un modelo aprendido.

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


print("Listo.")

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · Qué es exactamente un agente (20 minutos)
#
# La definición de Russell y Norvig, que es la que usa el programa de la
# materia, dice:
#
# > Un **agente** es cualquier entidad que percibe su entorno a través de
# > **sensores** y actúa sobre ese entorno mediante **actuadores**, buscando
# > maximizar una **medida de desempeño**.
#
# Hay tres cosas en esa definición que conviene no pasar por alto:
#
# 1. **No dice "inteligente" en el sentido de aprender.** Un termostato de dos
#    posiciones es un agente. Que sea un agente *bueno* es otra discusión.
# 2. **Hay una medida de desempeño.** Sin un criterio de éxito declarado no se
#    puede decir si el agente funciona. Esto reaparece, idéntico, cuando en el
#    módulo G-3 haya que elegir una métrica.
# 3. **El entorno es parte del problema.** El mismo programa es un agente
#    excelente en un entorno y un desastre en otro.
#
# ### La especificación PEAS
#
# Antes de programar nada, un agente se describe con cuatro campos. Es el
# equivalente, en IA, a la ficha de decisión del módulo H-1.
#
# | Letra | Campo | Pregunta que responde |
# |---|---|---|
# | **P** | Performance (desempeño) | ¿Cómo mido si el agente lo está haciendo bien? |
# | **E** | Environment (entorno) | ¿Sobre qué actúa? ¿Qué más hay ahí? |
# | **A** | Actuators (actuadores) | ¿Qué puede modificar del mundo? |
# | **S** | Sensors (sensores) | ¿Qué puede percibir del mundo? |
#
# **PEAS del riego automático de la cohorte:**
#
# | | |
# |---|---|
# | **P** | La planta nunca baja de humedad crítica y se usa la menor cantidad de agua posible |
# | **E** | Una maceta o cantero, a la intemperie, con temperatura variable durante el día |
# | **A** | Un relé que abre o cierra la bomba |
# | **S** | Sensor de humedad de suelo y sensor de temperatura |
#
# ### Los cinco tipos de agente
#
# | Tipo | Qué lo define | Ejemplo en esta cohorte |
# |---|---|---|
# | Reactivo simple | Decide **solo** con la percepción actual | Regar si humedad < 30 % |
# | Reactivo basado en modelo | Guarda **estado interno** del mundo | Regar considerando si ya venía regando (histéresis) |
# | Basado en objetivos | Tiene una **meta** explícita y planifica hacia ella | Mantener la humedad entre 40 % y 60 % todo el día |
# | Basado en utilidad | Compara alternativas con una **función de utilidad** | Balancear agua gastada contra estrés de la planta |
# | De aprendizaje | **Mejora con la experiencia** | Ajustar el umbral según cómo respondió la planta |
#
# > Casi todos los proyectos de segundo año son, en su primera versión,
# > reactivos simples. Eso está bien. Lo que **no** está bien es no darse cuenta
# > de que lo son.

# %% [markdown]
# ---
# ## Bloque 2 — Práctica · El agente como código (30 minutos)
#
# Un agente, en Python, es un objeto con tres métodos que se corresponden uno a
# uno con las tres partes de la definición. Lo escribimos así, explícito, para
# que la estructura quede visible.

# %%
class Agente:
    """Estructura mínima de un agente: percibe, decide, actúa."""

    def __init__(self, nombre):
        self.nombre = nombre
        self.bitacora = []   # historial de lo que hizo, para poder evaluarlo

    def percibir(self, entorno):
        """Toma del entorno lo que este agente puede sentir."""
        raise NotImplementedError

    def decidir(self, percepcion):
        """Devuelve la acción a ejecutar. Acá vive toda la inteligencia."""
        raise NotImplementedError

    def actuar(self, accion, entorno):
        """Aplica la acción sobre el entorno."""
        raise NotImplementedError

    def paso(self, entorno):
        """Un ciclo completo percepción -> decisión -> actuación."""
        percepcion = self.percibir(entorno)
        accion = self.decidir(percepcion)
        self.actuar(accion, entorno)
        self.bitacora.append({"percepcion": percepcion, "accion": accion})
        return accion


# %% [markdown]
# ### Agente 1 — Reactivo simple
#
# Decide únicamente con la lectura actual. Un solo umbral, nada de memoria.

# %%
class RiegoReactivoSimple(Agente):
    def __init__(self, umbral=30.0):
        super().__init__("Reactivo simple")
        self.umbral = umbral

    def percibir(self, entorno):
        return {"humedad": entorno["humedad_suelo_pct"]}

    def decidir(self, percepcion):
        if percepcion["humedad"] < self.umbral:
            return "REGAR"
        return "ESPERAR"

    def actuar(self, accion, entorno):
        entorno["bomba"] = (accion == "REGAR")


# %% [markdown]
# ### Agente 2 — Reactivo basado en modelo
#
# Este guarda **estado interno**: recuerda si la bomba ya estaba encendida. Eso
# le permite usar dos umbrales distintos —uno para arrancar y otro, más alto,
# para cortar—, lo que en control se llama **histéresis**.
#
# También lleva un modelo mínimo del mundo: si hace calor, la humedad se va a ir
# más rápido, así que conviene arrancar antes.

# %%
class RiegoConModelo(Agente):
    def __init__(self, umbral_encender=32.0, umbral_apagar=55.0):
        super().__init__("Basado en modelo")
        self.umbral_encender = umbral_encender
        self.umbral_apagar = umbral_apagar
        self.regando = False          # <- el estado interno
        self.ultima_humedad = None    # <- modelo del mundo: hacia dónde va

    def percibir(self, entorno):
        return {"humedad": entorno["humedad_suelo_pct"],
                "temperatura": entorno["temperatura_C"]}

    def decidir(self, percepcion):
        humedad = percepcion["humedad"]

        # Modelo del mundo: con calor, la evaporación es más rápida, así que se
        # adelanta el arranque unos puntos.
        umbral = self.umbral_encender
        if percepcion["temperatura"] > 25.0:
            umbral += 4.0

        if self.regando:
            # Ya está regando: solo corta cuando llega bien arriba.
            if humedad >= self.umbral_apagar:
                self.regando = False
                return "CORTAR"
            return "SEGUIR_REGANDO"

        if humedad < umbral:
            self.regando = True
            return "REGAR"
        return "ESPERAR"

    def actuar(self, accion, entorno):
        entorno["bomba"] = accion in ("REGAR", "SEGUIR_REGANDO")


# %% [markdown]
# ### El entorno tiene que reaccionar, si no la comparación no vale nada
#
# Acá hay una trampa metodológica en la que es fácil caer: **no se puede evaluar
# un agente pasándole datos ya grabados**. Si le doy al agente la humedad que
# registró el nodo real, esa humedad ya refleja las decisiones del controlador
# que estaba puesto, no las del agente que estoy probando. Su bomba no cambia
# nada; está actuando sobre una película.
#
# Para comparar en serio hace falta un **entorno simulado** que responda: si la
# bomba se enciende, la humedad tiene que subir de verdad. Eso es lo que hace la
# clase de abajo. La temperatura sí la tomamos del dataset real, porque es una
# variable exógena: el riego no la modifica.

# %%
class Maceta:
    """Modelo físico mínimo del suelo, para que el agente actúe sobre algo.

    Cada paso equivale a 30 minutos. La humedad baja por evaporación (más
    rápido cuanto más calor hace) y sube si la bomba está encendida.
    """

    def __init__(self, humedad_inicial, temperaturas, semilla=238):
        self.temperaturas = temperaturas
        self.rng = np.random.default_rng(semilla)
        self.t = 0
        self.humedad_real = humedad_inicial
        self.entorno = {"bomba": False}
        self.historial = []
        self._publicar()

    def _publicar(self):
        """Lo que el agente puede percibir: la lectura del sensor, con ruido."""
        lectura = self.humedad_real + self.rng.normal(0, 0.3)
        self.entorno["humedad_suelo_pct"] = lectura
        self.entorno["temperatura_C"] = self.temperaturas[self.t]

    def avanzar(self):
        bomba = self.entorno["bomba"]
        temperatura = self.temperaturas[self.t]
        if bomba:
            self.humedad_real = min(self.humedad_real + 6.0, 95.0)
        else:
            evaporacion = 0.22 + 0.035 * max(temperatura - 15.0, 0.0)
            self.humedad_real = max(self.humedad_real - evaporacion, 0.0)
        self.historial.append({"humedad": self.humedad_real, "bomba": bomba})
        self.t += 1
        self._publicar()


def simular(agente, temperaturas, humedad_inicial=60.0):
    maceta = Maceta(humedad_inicial, temperaturas)
    for _ in range(len(temperaturas) - 1):
        agente.paso(maceta.entorno)
        maceta.avanzar()
    historial = pd.DataFrame(maceta.historial)
    arranques = int(((historial["bomba"]) & (~historial["bomba"].shift(1, fill_value=False))).sum())
    return {
        "arranques de la bomba": arranques,
        "pasos con bomba encendida (agua)": int(historial["bomba"].sum()),
        "humedad promedio": round(historial["humedad"].mean(), 1),
        "humedad mínima alcanzada": round(historial["humedad"].min(), 1),
    }


# %%
riego = pd.read_csv(DATOS / "riego_humedad.csv", parse_dates=["timestamp"])
temperaturas = riego["temperatura_C"].to_numpy()[:1000]

comparacion = pd.DataFrame({
    "Reactivo simple": simular(RiegoReactivoSimple(), temperaturas),
    "Basado en modelo": simular(RiegoConModelo(), temperaturas),
})
comparacion

# %% [markdown]
# ### Qué muestra este experimento
#
# El agente reactivo simple **arranca la bomba unas cuatro veces más**. Cada
# arranque es un golpe de corriente sobre el relé y sobre el motor; en control
# esto se llama **chattering** (repiqueteo) y es una causa concreta de que el
# hardware se muera antes de tiempo.
#
# Ahora mirá la fila del agua: **es prácticamente la misma en los dos casos**.
# Eso no es casualidad ni un error de la simulación, y vale la pena entender por
# qué: en régimen permanente, el agua que entra tiene que igualar al agua que se
# evapora. La evaporación la fija la temperatura, no el controlador. **El
# controlador no decide cuánta agua se gasta; decide en cuántas tandas se
# gasta.**
#
# Esa es una conclusión que sale de correr el experimento, no de la intuición. Si
# hubiéramos supuesto que "regar más seguido gasta más agua" habríamos escrito
# algo razonable y falso.

# %% [markdown]
# ### ¿Y entonces qué elige uno cuando elige el umbral?
#
# Barramos el umbral de corte y miremos las tres métricas juntas.

# %%
barrido = pd.DataFrame({
    f"apagar en {u} %": simular(RiegoConModelo(umbral_encender=32.0, umbral_apagar=u), temperaturas)
    for u in [38, 45, 55, 65, 80]
})
barrido

# %% [markdown]
# El barrido deja ver el intercambio real:
#
# - **Arranques:** bajan fuerte a medida que sube el umbral de corte.
# - **Agua:** se mantiene casi constante, por la razón física de recién.
# - **Humedad promedio:** sube. La planta vive en una banda más húmeda.
#
# Así que la decisión no es "más o menos agua", sino: *¿cuántos ciclos de relé
# estoy dispuesto a gastar, y en qué banda de humedad quiero que viva la planta?*
# Un umbral de corte muy alto casi no castiga el relé, pero mantiene la tierra
# húmeda todo el tiempo, y para muchas especies eso significa pudrición de raíz.
#
# > **La pregunta "¿cuál agente es mejor?" no tiene respuesta sin la P de PEAS.**
# > Un agente no es bueno o malo en abstracto: es bueno o malo respecto de una
# > medida de desempeño declarada. Por eso el ejercicio A1.1 te va a pedir un
# > número, y no una intención.
#
# Y fijate que el agente 2 no es más "inteligente" en ningún sentido místico.
# Solo tiene una variable más: `self.regando`. Ese es, literalmente, todo el
# salto entre un tipo de agente y el siguiente.

# %% [markdown]
# ---
# ## Bloque 3 — Teoría · Sistemas basados en reglas (25 minutos)
#
# Antes de que el aprendizaje automático fuera práctico, la IA aplicada se hacía
# con **sistemas expertos**: bases de reglas escritas por una persona que sabe
# del dominio. Hoy siguen siendo la mejor opción cuando se dan tres condiciones,
# que son exactamente las de la mayoría de los proyectos de esta materia:
#
# 1. El conocimiento es **explícito** (un técnico lo puede escribir).
# 2. Hay **pocos datos históricos** (un cuatrimestre no alcanza para entrenar).
# 3. Se necesita **explicabilidad** (hay que poder justificar cada alerta).
#
# ### Anatomía de un sistema de reglas
#
# | Pieza | Qué es |
# |---|---|
# | **Base de hechos** | Lo que el sistema sabe ahora mismo. Empieza con las percepciones. |
# | **Base de reglas** | Las reglas SI-ENTONCES escritas por el experto. |
# | **Motor de inferencia** | El programa que aplica reglas sobre hechos y deriva hechos nuevos. |
#
# ### Encadenamiento hacia adelante
#
# El motor recorre las reglas; cada regla cuya condición se cumple **dispara** y
# agrega hechos nuevos. Con esos hechos nuevos vuelve a recorrer todas las
# reglas. Sigue así hasta que una pasada completa no agrega nada: eso se llama
# **punto fijo**, y ahí terminó la inferencia.
#
# La consecuencia interesante es que **las reglas se encadenan solas**: la regla
# que concluye `sensor_sospechoso` habilita a la regla que concluye
# `diagnostico = sensor_desconectado`, sin que nadie las haya conectado a mano.

# %%
class Regla:
    """Una regla de producción: SI condicion ENTONCES conclusion."""

    def __init__(self, nombre, condicion, conclusion, explicacion):
        self.nombre = nombre
        self.condicion = condicion      # función hechos -> bool
        self.conclusion = conclusion    # dict con los hechos que agrega
        self.explicacion = explicacion  # texto en castellano, para la traza

    def aplica(self, hechos):
        try:
            return self.condicion(hechos)
        except KeyError:
            # Si la regla necesita un hecho que todavía no existe, no aplica.
            return False


class MotorDeInferencia:
    """Encadenamiento hacia adelante hasta punto fijo, con traza."""

    def __init__(self, reglas, max_pasadas=20):
        self.reglas = reglas
        self.max_pasadas = max_pasadas

    def inferir(self, hechos_iniciales):
        hechos = dict(hechos_iniciales)
        traza = []
        disparadas = set()

        for pasada in range(1, self.max_pasadas + 1):
            algo_nuevo = False
            for regla in self.reglas:
                if regla.nombre in disparadas:
                    continue                      # cada regla dispara una vez
                if not regla.aplica(hechos):
                    continue
                # Solo cuenta como nuevo si cambia o agrega algún hecho.
                nuevos = {k: v for k, v in regla.conclusion.items()
                          if hechos.get(k) != v}
                if not nuevos:
                    disparadas.add(regla.nombre)
                    continue
                hechos.update(nuevos)
                disparadas.add(regla.nombre)
                traza.append({"pasada": pasada, "regla": regla.nombre,
                              "agrego": nuevos, "porque": regla.explicacion})
                algo_nuevo = True
            if not algo_nuevo:
                break                             # punto fijo alcanzado

        return hechos, traza


def mostrar_traza(traza):
    if not traza:
        print("   (ninguna regla disparó)")
        return
    for paso in traza:
        agregado = ", ".join(f"{k} = {v}" for k, v in paso["agrego"].items())
        print(f"   pasada {paso['pasada']} · {paso['regla']}")
        print(f"      porque: {paso['porque']}")
        print(f"      concluye: {agregado}")


print("Motor listo.")

# %% [markdown]
# ### Un sistema experto de diagnóstico de nodo IoT
#
# Este es el ejercicio que pide el Bloque 2 del programa anual: *diagnóstico en
# sistemas tecnológicos*. Las reglas están escritas desde la experiencia de
# armar nodos, no desde un dataset.

# %%
reglas_diagnostico = [
    Regla("R1_sin_alimentacion",
          lambda h: not h["nodo_responde"] and not h["led_encendido"],
          {"diagnostico": "sin alimentación", "confianza": "alta", "resuelto": True},
          "el nodo no responde y además no tiene ni el LED de power encendido"),

    Regla("R2_firmware_colgado",
          lambda h: not h["nodo_responde"] and h["led_encendido"],
          {"sospecha_firmware": True},
          "hay alimentación (LED encendido) pero el nodo no contesta"),

    Regla("R3_firmware_confirmado",
          lambda h: h.get("sospecha_firmware") and not h["responde_tras_reset"],
          {"diagnostico": "firmware colgado o corrupto", "confianza": "alta", "resuelto": True},
          "sigue sin responder incluso después de un reset"),

    Regla("R4_problema_red",
          lambda h: h["nodo_responde"] and not h["llegan_datos_al_servidor"],
          {"sospecha_red": True},
          "el nodo funciona localmente pero al servidor no le llega nada"),

    Regla("R5_wifi",
          lambda h: h.get("sospecha_red") and not h["wifi_asociado"],
          {"diagnostico": "el nodo no se asocia a la red WiFi", "confianza": "alta", "resuelto": True},
          "hay problema de red y el nodo ni siquiera está asociado al AP"),

    Regla("R6_broker",
          lambda h: h.get("sospecha_red") and h["wifi_asociado"],
          {"diagnostico": "WiFi ok, falla la publicación (broker o credenciales MQTT)",
           "confianza": "media", "resuelto": True},
          "hay WiFi pero los datos no llegan: el problema está más arriba en la pila"),

    Regla("R7_sensor_congelado",
          lambda h: h["nodo_responde"] and h["llegan_datos_al_servidor"]
                    and h["lectura_constante_horas"] >= 3,
          {"diagnostico": "sensor desconectado o congelado", "confianza": "media", "resuelto": True},
          "llegan datos pero el valor no cambia hace horas: el sensor no está midiendo"),

    Regla("R8_todo_bien",
          lambda h: h["nodo_responde"] and h["llegan_datos_al_servidor"]
                    and h["lectura_constante_horas"] < 3,
          {"diagnostico": "sin falla detectada", "confianza": "alta", "resuelto": True},
          "el nodo responde, publica y el sensor varía como corresponde"),
]

motor = MotorDeInferencia(reglas_diagnostico)

# %%
casos = {
    "Caso A — el nodo está muerto": {
        "nodo_responde": False, "led_encendido": False, "responde_tras_reset": False,
        "wifi_asociado": False, "llegan_datos_al_servidor": False, "lectura_constante_horas": 0,
    },
    "Caso B — prende pero no arranca": {
        "nodo_responde": False, "led_encendido": True, "responde_tras_reset": False,
        "wifi_asociado": False, "llegan_datos_al_servidor": False, "lectura_constante_horas": 0,
    },
    "Caso C — anda pero no publica": {
        "nodo_responde": True, "led_encendido": True, "responde_tras_reset": True,
        "wifi_asociado": True, "llegan_datos_al_servidor": False, "lectura_constante_horas": 0,
    },
    "Caso D — publica siempre lo mismo": {
        "nodo_responde": True, "led_encendido": True, "responde_tras_reset": True,
        "wifi_asociado": True, "llegan_datos_al_servidor": True, "lectura_constante_horas": 7,
    },
}

for titulo, hechos in casos.items():
    conclusiones, traza = motor.inferir(hechos)
    print(titulo)
    print(f"   DIAGNÓSTICO: {conclusiones.get('diagnostico', 'no concluyente')} "
          f"(confianza {conclusiones.get('confianza', '-')})")
    mostrar_traza(traza)
    print()

# %% [markdown]
# ### Esto es lo que significa "explicable"
#
# Mirá la salida del **Caso B**: el sistema no solo dijo *firmware colgado*, sino
# que mostró las dos reglas que encadenó para llegar ahí y por qué cada una se
# activó. Podés discutir con el diagnóstico, corregir una regla, agregar una
# nueva.
#
# En el cuaderno A-4 vamos a entrenar un árbol de decisión que también da un
# diagnóstico, pero cuya justificación es *"los datos históricos dicen que sí"*.
# Los dos enfoques son válidos; **son válidos para cosas distintas**. Esa
# elección, con argumento, es exactamente lo que pide el módulo G-2.

# %% [markdown]
# ---
# ## Bloque 4 — Ejercicios

# %% [markdown]
# ### Ejercicio A1.1 [B] — PEAS de tu nodo
#
# Completá el diccionario con la especificación PEAS de **tu proyecto**. Escribí
# oraciones completas, no palabras sueltas. El campo `desempeno` tiene que ser
# medible: "que funcione bien" no sirve; "que la humedad nunca baje de 25 %"
# sirve.

# %%
# TU CÓDIGO ACÁ
peas = {
    "proyecto": "",
    "desempeno": "",
    "entorno": "",
    "actuadores": "",
    "sensores": "",
}

# %%
check("Los cinco campos están completos",
      all(isinstance(v, str) and len(v.strip()) > 0 for v in peas.values()),
      "no dejes ningún campo vacío")
check("La medida de desempeño es concreta",
      len(peas["desempeno"].split()) >= 6 and any(c.isdigit() for c in peas["desempeno"]),
      "una medida de desempeño casi siempre tiene un número adentro")

# %% [markdown]
# ### Ejercicio A1.2 [B] — Clasificar el agente
#
# Guardá en `tipo_de_agente` uno de estos cinco textos exactos, según qué es
# **hoy** tu nodo (no lo que te gustaría que fuera):
#
# `"reactivo simple"`, `"reactivo basado en modelo"`, `"basado en objetivos"`,
# `"basado en utilidad"`, `"de aprendizaje"`
#
# Y en `justificacion`, dos oraciones explicando por qué.

# %%
# TU CÓDIGO ACÁ
tipo_de_agente = ""
justificacion = ""

# %%
_validos = {"reactivo simple", "reactivo basado en modelo", "basado en objetivos",
            "basado en utilidad", "de aprendizaje"}
check("El tipo elegido es uno de los cinco", tipo_de_agente.strip().lower() in _validos)
check("La justificación tiene al menos dos oraciones",
      justificacion.count(".") >= 2,
      "explicá qué percibe, qué decide y si guarda o no estado interno")

# %% [markdown]
# ### Ejercicio A1.3 [I] — Tu agente con histéresis
#
# Escribí la clase `MiAgente`, subclase de `Agente`, que implemente la lógica de
# decisión de **tu** proyecto con al menos un estado interno.
#
# Si tu proyecto es el **enchufe**: encendé una alerta cuando la potencia supere
# 250 W y apagala recién cuando baje de 150 W.
# Si es el **silobolsa**: alerta cuando el CO₂ pase de 800 ppm, se apaga por
# debajo de 600 ppm.
#
# El método `decidir` tiene que devolver `"ALERTA"` o `"NORMAL"`.

# %%
# TU CÓDIGO ACÁ
class MiAgente(Agente):
    def __init__(self):
        super().__init__("Mi agente")
        # definí acá tu estado interno

    def percibir(self, entorno):
        pass

    def decidir(self, percepcion):
        pass

    def actuar(self, accion, entorno):
        pass


# %%
# Verificación: la histéresis se prueba con una secuencia que sube y baja.
# Se le pasa una señal que cruza los umbrales y se controla que NO repiquetee.
_secuencia = [100, 200, 260, 300, 240, 200, 180, 160, 140, 120, 260, 100]
try:
    _agente = MiAgente()
    _acciones = []
    for _valor in _secuencia:
        _entorno = {"potencia_W": _valor, "co2_ppm": _valor * 4,
                    "humedad_suelo_pct": _valor / 4, "temperatura_C": 20.0}
        _acciones.append(_agente.paso(_entorno))
except Exception as e:
    _acciones = []
    print(f"  [REVISAR] El agente todavía no corre: {type(e).__name__}: {e}")

if set(_acciones) == {"ALERTA", "NORMAL"}:
    _cambios = sum(1 for i in range(1, len(_acciones)) if _acciones[i] != _acciones[i - 1])
    check("El agente devuelve solo ALERTA o NORMAL", True)
    check("Hay histéresis: no cambia de estado apenas cruza el umbral",
          _cambios <= 4,
          f"tu agente cambió de estado {_cambios} veces; con dos umbrales tendrían que ser 4 o menos")
    print("\n   Secuencia de acciones:", " ".join(a[0] for a in _acciones))
elif _acciones:
    check("El agente devuelve solo ALERTA o NORMAL", False,
          f"decidir() devolvió {sorted(set(map(str, _acciones)))}; tienen que ser esas dos cadenas exactas")

# %% [markdown]
# ### Ejercicio A1.4 [I] — Cinco reglas para tu proyecto
#
# El programa anual pide **al menos 5 reglas de producción** para un problema
# dado. Escribilas para tu proyecto usando la clase `Regla`, y armá el motor.
#
# Requisitos:
# - Al menos 5 reglas.
# - Al menos una regla tiene que **encadenarse** con otra: es decir, usar en su
#   condición un hecho que otra regla concluyó (como `sospecha_firmware` en el
#   ejemplo de arriba).
# - Todas las reglas necesitan su texto de `explicacion`.

# %%
# TU CÓDIGO ACÁ
mis_reglas = [
    # Regla("R1_...", lambda h: ..., {"...": ...}, "porque ..."),
]

mis_hechos_de_prueba = {
    # los hechos iniciales con los que vas a probar el motor
}

# %%
if not mis_reglas:
    print("  [REVISAR] Todavía no escribiste ninguna regla.")
else:
    _motor = MotorDeInferencia(mis_reglas)
    _conclusiones, _traza = _motor.inferir(mis_hechos_de_prueba)
    _hechos_derivados = set(_conclusiones) - set(mis_hechos_de_prueba)
    _hechos_usados_en_condiciones = sum(
        1 for paso in _traza if paso["pasada"] > 1)

    check("Escribiste al menos 5 reglas", len(mis_reglas) >= 5)
    check("Todas tienen explicación",
          all(isinstance(r.explicacion, str) and len(r.explicacion) > 10 for r in mis_reglas))
    check("El motor derivó hechos nuevos", len(_hechos_derivados) > 0,
          "revisá que los hechos iniciales activen al menos una condición")
    check("Hay encadenamiento (alguna regla disparó en la pasada 2 o posterior)",
          _hechos_usados_en_condiciones >= 1,
          "una regla tiene que concluir un hecho que otra regla use como condición")
    print()
    mostrar_traza(_traza)

# %% [markdown]
# ### Ejercicio A1.5 [A] — Cuándo las reglas no alcanzan
#
# Sin verificación automática. Respondé en la celda de texto de abajo:
#
# 1. Escribí una situación **concreta de tu proyecto** en la que un sistema de
#    reglas daría una respuesta equivocada o insuficiente.
# 2. Explicá qué le falta a las reglas en ese caso: ¿es que no conocés el umbral
#    correcto? ¿que el umbral depende de otras variables? ¿que el patrón cambia
#    con el tiempo?
# 3. Ese diagnóstico es, textualmente, la respuesta a la pregunta del módulo G-2:
#    *"¿mi problema tiene un patrón que valga la pena que un modelo aprenda?"*.
#    Anticipá tu respuesta acá; en el cuaderno A-5 la vas a poner a prueba con
#    una métrica.

# %% [markdown]
# **Tu respuesta:**
#
# *(escribí acá, haciendo doble clic sobre esta celda)*
#
# 1.
# 2.
# 3.

# %% [markdown]
# ---
# ## Cierre del cuaderno A-1
#
# **Lo que quedó instalado en tu cabeza:**
#
# - Un agente se especifica con PEAS *antes* de programarse. La medida de
#   desempeño no es opcional.
# - La diferencia entre un agente reactivo simple y uno basado en modelo es una
#   variable de estado, y esa variable tiene consecuencias físicas medibles
#   sobre tu hardware.
# - Un motor de inferencia hacia adelante aplica reglas hasta punto fijo, y las
#   reglas se encadenan solas.
# - Un sistema de reglas es explicable: podés mostrar la traza que lleva a cada
#   conclusión. Guardate esa idea para compararla con el árbol de A-4.
#
# **Checklist de entrega**
#
# - [ ] PEAS de tu proyecto, con desempeño medible (A1.1).
# - [ ] Tipo de agente elegido y justificado (A1.2).
# - [ ] `MiAgente` corriendo con histéresis verificada (A1.3).
# - [ ] Al menos 5 reglas propias, con encadenamiento y traza visible (A1.4).
# - [ ] La respuesta escrita del A1.5, que después vas a citar en G-2.
#
# **Sigue en:** `A2_Ciclo_de_vida_del_dato.ipynb`
