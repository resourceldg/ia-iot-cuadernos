# Notas de cátedra — qué esperar de cada cuaderno

Uso interno. Resultados verificados sobre los datos generados con semilla 238,
con numpy 2.5, pandas 3.0, scikit-learn 1.9 y matplotlib 3.11.

---

## Idea rectora del anexo

El primer cuatrimestre quedó en fundamentos sin práctica. Este anexo apunta a lo
contrario: **cada concepto se demuestra corriendo código, y varias demostraciones
terminan en un resultado que contradice la expectativa.** Eso es deliberado. La
competencia que se busca formar no es "saber usar scikit-learn", es **comparar
contra un baseline y reportar lo que salió**.

Si hay que recortar por tiempo, el orden de prioridad es:

1. **A-2** (dato) y **A-4** (métricas y baseline) — irrenunciables.
2. **A-5** — tiene la idea más aprovechable para la carrera.
3. **A-1**, **A-3** — cubren F-0 y F-3 directamente.
4. **A-6**, **A-7** — teoría faltante; se pueden dar como unidad aparte.
5. **A-8** — opcional.

---

## Resultados que van a aparecer, cuaderno por cuaderno

### A-0 · Entorno

Sin sorpresas. La única fricción esperable es el `externally-managed-environment`
de pip en Debian/Ubuntu recientes: hay que usar el entorno virtual.

El ejercicio A0.4 (contar lecturas imposibles) da **58 de 4320**. La pista no
revela el número; el resumen solo se imprime si acertaron.

---

### A-1 · Agentes y reglas

**El simulador es el punto pedagógico.** Se explica explícitamente por qué no se
puede evaluar un agente contra datos grabados: la humedad registrada ya refleja
las decisiones del controlador que estaba puesto.

Resultados de la simulación (1000 pasos):

| | Reactivo simple | Basado en modelo |
|---|---|---|
| Arranques de la bomba | 49 | 13 |
| Pasos con bomba encendida | 49 | 52 |
| Humedad promedio | ~34 | ~44 |

**El agua es prácticamente la misma.** Ese resultado suele sorprender y es
físicamente correcto: en régimen permanente el agua que entra iguala a la que se
evapora, y la evaporación la fija la temperatura, no el controlador. El barrido
de umbral lo confirma: los arranques bajan de 33 a 6 mientras el agua no se
mueve.

Vale la pena detenerse ahí. Es el primer momento del anexo en que **la intuición
razonable ("regar más seguido gasta más agua") resulta falsa y el experimento lo
demuestra.**

El motor de inferencia diagnostica los cuatro casos correctamente y muestra el
encadenamiento en el Caso B (R2 → R3) y el Caso C (R4 → R6).

---

### A-2 · Ciclo de vida del dato

**El demo de Nyquist** barre todas las fases posibles de muestreo. Con muestreo
cada 15, 30 o 60 minutos el mismo pico de 300 W se lee entre 40 y 300 W según en
qué minuto arrancó el nodo. Ese rango es el argumento, no el promedio.

**Detección sobre la silobolsa:** 58 fuera de rango, 26 faltantes, 59 congeladas
en 3 tramos, 5 marcas de tiempo fuera de orden.

**El resultado contraintuitivo de este cuaderno** está en la comparación
antes/después de limpiar:

| | antes | después |
|---|---|---|
| promedio | 546.6 | 530.0 |
| desvío | 299.4 | 213.0 |
| mínimo | −294.5 | 368.0 |
| máximo | 3179.4 | 1386.6 |

**El promedio casi no se movió.** Quien mire solo el promedio concluye que los
datos sucios "no afectaban tanto". Lo que se derrumbó son los extremos, y de ahí
sale la demostración que cierra el bloque: un umbral fijado como el 80 % del
máximo histórico da **2543 ppm con datos sucios (0 alertas en todo el histórico)**
contra **1109 ppm con datos limpios (205 alertas)**. El 1 % de los datos
inutilizaba el sistema entero.

---

### A-3 · Exploración y visualización

Correlación temperatura–CO₂ en la silobolsa: **r = 0.39** (r² = 0.15). Conviene
señalar que la relación es causalmente indirecta: la fermentación produce calor
*y* CO₂; la temperatura no causa el CO₂. Sirve igual como predictor, y esa
distinción prepara A-4.

Las tres "mentiras" están construidas para que el estudiante las reproduzca:

- **Eje truncado:** cuatro valores con 5.3 % de variación real.
- **Doble eje:** dos series con r = 0.97, presentadas como "van juntas" y "no
  tienen nada que ver" cambiando solo los límites. El argumento no es que el
  doble eje se use mal, es que **su corrección depende de la honestidad del
  autor**, y por eso no se usa.
- **Promedio que tapa el pico:** promedio diario ~60 W contra máximos de más de
  400 W. Se conecta con dimensionar el relé.

El módulo `estilo_grafico.py` trae una paleta verificada para daltonismo. El
orden de los colores no es cosmético: cada par consecutivo fue medido. Si alguien
la cambia, que respete el orden.

---

### A-4 · Primer modelo supervisado

**Lo más importante de este cuaderno: la honestidad sobre la fuga temporal.**

En el problema de clasificación de la silobolsa, la partición al azar y la
temporal dan casi lo mismo (0.984 contra 0.983). El texto **no lo esconde**: lo
explica (el fenómeno es lento y las características describen estado, no
instante) y avisa que en el problema de regresión sí se va a ver.

Y se ve: MAE temporal 2.075 % contra 1.734 % al azar — la partición al azar
reporta un error 16 % menor del real.

El argumento final es el correcto: **no se puede saber de antemano cuánto va a
mentir la partición al azar en un problema dado**, y el protocolo correcto cuesta
una línea.

**Métricas de clasificación (prueba temporal, 10.2 % de positivos):**

| | Árbol (prof. 3) | "Siempre digo que no" |
|---|---|---|
| exactitud | 0.983 | 0.898 |
| precisión | 0.871 | 0.000 |
| sensibilidad | 0.975 | 0.000 |
| F1 | 0.920 | 0.000 |

**Regresión (predecir humedad a 3 h):**

| | MAE | RMSE |
|---|---|---|
| Baseline: persistencia | 4.503 | 9.748 |
| Regresión lineal | **5.064** | 8.953 |
| Árbol de regresión | 2.075 | 5.261 |

**La regresión lineal pierde contra el baseline.** Es el resultado más valioso
del cuaderno y conviene detenerse: un modelo entrenado, con cuatro
características, anda peor que repetir el último valor. La humedad es una sierra
y una recta no la representa. Quien no calcula el baseline reporta un MAE de 5 %
como logro.

La curva de sobreajuste tiene su mínimo en profundidad 9 y el error de
entrenamiento llega a cero en la 19.

---

### A-5 · Reglas contra modelo

Umbral óptimo por F1 **elegido sobre entrenamiento**: 870 ppm. Se insiste en el
punto: elegirlo mirando prueba es exactamente lo que el cuaderno critica.

**Resultados sobre prueba:**

| | Regla 1 (umbral) | Regla 2 (nivel+tendencia, a ojo) | Árbol |
|---|---|---|---|
| precisión | 0.883 | 0.824 | 0.871 |
| sensibilidad | 0.898 | 0.831 | 0.975 |
| F1 | 0.891 | 0.827 | 0.920 |

**La Regla 2 pierde contra la Regla 1.** Tiene más condiciones y peores números,
porque sus dos parámetros fueron elegidos a ojo mientras que el umbral de la
Regla 1 salió de un barrido. La moraleja —*los parámetros se buscan, no se
adivinan*— vale para todo el anexo.

**Y el resultado que cierra el cuaderno:**

| | Árbol (en la PC) | Regla derivada (en el ESP32) |
|---|---|---|
| precisión | 0.871 | **0.927** |
| sensibilidad | 0.975 | 0.975 |
| F1 | 0.920 | **0.950** |

**La regla escrita a mano le gana al árbol del que salió.** No es una
transcripción literal: se podaron las ramas que terminan en hojas de la misma
clase y se le agregó un piso de nivel con sentido físico. Esa poda funciona como
regularización.

El texto advierte explícitamente que la diferencia es chica, que hay tres
episodios en el conjunto, y que **hay que reportarlo como observación sobre este
conjunto y no como ley**. Conviene reforzarlo en clase.

Este bloque es el que más rinde en esta carrera: usar el modelo para descubrir la
regla, y desplegar la regla en el firmware.

---

### A-6 · Red neuronal desde cero

Verificación de gradiente: diferencia relativa **5.25e-09**. Si a algún
estudiante le da por encima de 1e-5, tiene una derivada mal.

XOR resuelto con 4 neuronas ocultas en 3000 épocas.

**Ejercicio A6.2 — atención acá.** La tasa de éxito sobre 8 semillas:

| neuronas ocultas | éxito |
|---|---|
| 1 | 0 / 8 |
| 2 | 4 / 8 |
| 3 | 8 / 8 |
| 4 | 8 / 8 |

Con 2 neuronas funciona **la mitad de las veces**. La versión original del
ejercicio pedía "el mínimo" y era una trampa: con la semilla por defecto, 2
funciona y 3 falla. Se rediseñó para pedir la tasa de éxito sobre varias
semillas, y la conclusión —mínimos locales, y **un resultado de una sola corrida
no es un resultado**— es mejor que la pregunta original.

**Normalización:** sin normalizar, F1 = 0.000 (la sigmoide se satura y el
gradiente muere). Normalizada, F1 = 0.931. La diferencia es total, no de matiz.

**Red contra árbol:** F1 0.931 contra 0.920. La red gana por nada, y el cuaderno
lo dice: con cinco características a mano sobre 90 días, no es el problema que
justifica una red.

---

### A-7 · Modelo de lenguaje diminuto

Corpus de **730 palabras**, vocabulario de 27 símbolos.

| modelo | perplejidad en validación |
|---|---|
| Al azar | 27.0 |
| Bigramas (contexto 1) | 9.8 |
| Red neuronal (contexto 3) | 8.3 |

El modelo tiene **1661 parámetros** y entrena en unos 15 segundos en CPU.

**Parada temprana:** el mejor punto está cerca de la época 1100 de 2500. Sin
parada temprana el modelo termina **peor que la tabla de bigramas**. Vale
mostrarlo: es la continuación directa de la curva de sobreajuste de A-4, ahora
con una técnica que hace algo al respecto.

La generación por temperatura es el material más aprovechable para hablar de
alucinaciones. A T = 0.7 salen cosas como *luna, lacion, filado, repara*: palabras
que no existen pero podrían existir. **La alucinación no es una falla del
mecanismo, es el mecanismo**: probable y verdadero son propiedades distintas.

---

### A-8 · LLM local (opcional)

Verificado con SmolLM2-135M-Instruct, torch 2.13 CPU, transformers 5.15.

- Carga en ~2 s desde caché; primera vez descarga ~270 MB.
- 134.5M parámetros, vocabulario 49 152, contexto 8192, 538 MB en float32.
- Velocidad medida: **~29 tokens/s** en CPU (16 núcleos, sin GPU).

**Sesgo de tokenización, medido:**

| frase | tokens |
|---|---|
| "El sensor de humedad del suelo mide" | 11 |
| "The soil moisture sensor measures" | 5 |
| "microcontrolador" | 3 (micro · control · ador) |
| "silobolsa" | 4 (sil · ob · ols · a) |

**Las alucinaciones salen redondas**, que es justamente por qué se eligió un
modelo chico. En la corrida de verificación:

- *"What does an MQ-135 sensor measure?"* → **"measures the altitude and velocity
  of the air around it, providing pilots with situational awareness"**. Falso de
  punta a punta, escrito con total seguridad.
- *"What is MQTT used for?"* → correcto en general, "mesh network" es incorrecto.
  Un caso de "parcial" de manual.
- La pregunta en castellano produce salida degenerada, con una definición
  circular y un "Año de especialidad: 2017" de la nada.
- *"Maximum current of an ESP32 GPIO pin"* → dice 100 mA "porque usa un selector
  digital de 8 bits". El valor está mal (el absoluto es ~40 mA) y la explicación
  es un sinsentido.

**Ojo con el Bloque 6:** la tarea de reescribir notas técnicas **no sale bien**.
El modelo copia las notas casi textuales y se corta. Eso está contemplado en el
texto y se usa como lección: la tarea es la correcta, el modelo es demasiado
chico. No hay que presentarlo como si funcionara.

---

## Errores de instalación que van a aparecer

| Error | Solución |
|---|---|
| `externally-managed-environment` | Falta el entorno virtual. |
| `ModuleNotFoundError: estilo_grafico` | Jupyter no arrancó en `notebooks/`. |
| `FileNotFoundError` con los `.csv` | Ídem. |
| Gráficos en blanco | Falta `%matplotlib inline` en versiones viejas de Jupyter, o hay que reiniciar el kernel. |
| A-8 muy lento | Es normal en máquinas con pocos núcleos. Bajar `max_new_tokens`. |
