# %% [markdown]
# # Anexo A-6 · Una red neuronal desde cero, con NumPy y nada más
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Profundización teórica del Bloque 4 · base para A-7 |
# | **Duración** | 150 minutos |
# | **Modalidad** | Individual o de a dos |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Explicar qué es una neurona artificial en términos de dos operaciones, y por
#    qué sin la segunda toda la red se derrumba a una recta.
# 2. Demostrar con código el límite del perceptrón (XOR) y superarlo agregando una
#    capa.
# 3. Implementar **descenso por gradiente y retropropagación** en NumPy, sin
#    ningún framework, y verificar que las derivadas están bien.
# 4. Entrenar la red sobre los datos de la silobolsa, normalizando como
#    corresponde, y compararla honestamente contra el árbol de A-4.
#
# ### Antes de arrancar
#
# Nada de este cuaderno necesita GPU, ni internet, ni PyTorch. Todo corre en
# segundos en cualquier notebook con NumPy. La idea es justamente esa: **una red
# neuronal no es una caja mágica, son treinta líneas de álgebra**. Después vas a
# usar frameworks, y vas a saber qué están haciendo.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from estilo_grafico import aplicar_estilo, titular, SERIE, ESTADO, TINTA_SUAVE, SECUENCIAL

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


rng = np.random.default_rng(238)
print("Listo.")

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · Qué hace una neurona (25 minutos)
#
# Una neurona artificial hace **dos** cosas, en este orden:
#
# **1. Una combinación lineal de sus entradas.** Cada entrada tiene un peso; se
# multiplican, se suman, y se le agrega un término independiente llamado *sesgo*
# (*bias*):
#
# $$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
#
# Si te suena a la ecuación de una recta, es porque es exactamente eso, en varias
# dimensiones. Los pesos son la pendiente y el sesgo es la ordenada al origen.
#
# **2. Una función de activación, que no es lineal.**
#
# $$a = f(z)$$
#
# ### Por qué la segunda parte es imprescindible
#
# Este es *el* punto que casi nunca se explica bien, así que vale la pena
# demostrarlo en lugar de afirmarlo.
#
# Si `f` fuera lineal (o directamente no estuviera), entonces apilar capas no
# serviría de nada: la composición de dos funciones lineales es otra función
# lineal. Una red de 50 capas sin activación tiene exactamente el mismo poder que
# una sola neurona.

# %%
# Demostración: dos capas lineales seguidas equivalen a UNA capa lineal.
W1 = rng.normal(size=(3, 4))
W2 = rng.normal(size=(4, 2))
x = rng.normal(size=(5, 3))

dos_capas = (x @ W1) @ W2
una_sola = x @ (W1 @ W2)          # la matriz equivalente, precalculada

print("Diferencia máxima entre 'dos capas' y 'una capa equivalente':",
      np.abs(dos_capas - una_sola).max())
print("\nEs cero (salvo error de redondeo). Sin activación no lineal,")
print("agregar capas no agrega absolutamente ninguna capacidad.")

# %% [markdown]
# ### Las tres activaciones que vas a ver
#
# | Nombre | Fórmula | Dónde se usa |
# |---|---|---|
# | **sigmoide** | $\sigma(z) = 1/(1+e^{-z})$ | salida de un clasificador binario: aplasta cualquier número a (0, 1), o sea a una probabilidad |
# | **tanh** | $\tanh(z)$ | capas ocultas; parecida a la sigmoide pero centrada en cero |
# | **ReLU** | $\max(0, z)$ | capas ocultas en redes grandes; barata de calcular |

# %%
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


z = np.linspace(-6, 6, 400)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(z, sigmoide(z), color=SERIE[0], label="sigmoide")
ax.plot(z, np.tanh(z), color=SERIE[1], label="tanh")
ax.plot(z, np.maximum(0, z), color=SERIE[2], label="ReLU")
ax.axhline(0, color="#c3c2b7", linewidth=0.8)
ax.axvline(0, color="#c3c2b7", linewidth=0.8)
ax.set_ylim(-1.5, 3)
ax.set_xlabel("z (la suma ponderada)")
ax.set_ylabel("a (la salida de la neurona)")
ax.legend()
titular(ax, "Las tres funciones de activación que vas a encontrar")
plt.show()

# %% [markdown]
# ---
# ## Bloque 2 — Teoría y práctica · El límite del perceptrón (25 minutos)
#
# Un **perceptrón** es una sola neurona: una combinación lineal y una activación.
# Geométricamente, lo único que puede hacer es **trazar una recta** (o un plano) y
# decir "de este lado sí, de este lado no".
#
# Eso alcanza para muchísimos problemas. Pero hay uno, sencillísimo, que no puede
# resolver: la función **XOR** (*o exclusivo*): verdadero cuando **una sola** de
# las dos entradas es verdadera.
#
# | x₁ | x₂ | XOR |
# |---|---|---|
# | 0 | 0 | 0 |
# | 0 | 1 | 1 |
# | 1 | 0 | 1 |
# | 1 | 1 | 0 |
#
# Probá de dibujar esos cuatro puntos y separarlos con **una sola recta**. No se
# puede: los dos "sí" están en diagonal, y los dos "no" también.
#
# > Este resultado, publicado por Minsky y Papert en 1969, frenó la investigación
# > en redes neuronales durante más de una década: se conoce como el primer
# > *invierno de la IA*. Lo que faltaba era agregar una capa en el medio y saber
# > cómo entrenarla.
#
# Y no es una curiosidad académica: **XOR es exactamente la forma de muchísimos
# problemas de IoT**. "Alertar si la humedad está baja *o* la temperatura está
# alta, pero no si están las dos" es un XOR. Ninguna combinación de umbrales
# lineales lo resuelve.

# %%
X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor = np.array([[0.], [1.], [1.], [0.]])

fig, ax = plt.subplots(figsize=(4.6, 4.4))
for clase, color, marca, etiqueta in [(0, SERIE[0], "o", "XOR = 0"),
                                      (1, SERIE[1], "s", "XOR = 1")]:
    puntos = X_xor[y_xor.ravel() == clase]
    ax.scatter(puntos[:, 0], puntos[:, 1], s=220, color=color,
               marker=marca, label=etiqueta, zorder=3)
for recta in [0.5, 0.8]:
    ax.plot([-0.3, 1.3], [recta + 0.3, recta - 0.6], color=ESTADO["critico"],
            linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.grid(axis="both")
ax.legend(loc="center")
titular(ax, "Probá separar los círculos de los cuadrados con una recta")
plt.show()

# %% [markdown]
# ---
# ## Bloque 3 — Teoría · Cómo aprende una red (30 minutos)
#
# Entrenar es ajustar los pesos para que la red se equivoque menos. Eso son tres
# ingredientes:
#
# ### 1. Una función de pérdida: cuánto nos estamos equivocando
#
# Para clasificación binaria se usa la **entropía cruzada binaria**:
#
# $$L = -\frac{1}{n}\sum \left[ y \log(p) + (1-y)\log(1-p) \right]$$
#
# En criollo: si la respuesta correcta es 1 y la red dijo 0.99, el castigo es
# casi nulo. Si dijo 0.01, el castigo es enorme. **Castiga la confianza
# equivocada mucho más que la duda.**
#
# ### 2. El gradiente: hacia dónde mover cada peso
#
# El gradiente de la pérdida respecto de un peso es la derivada: dice cuánto
# cambia el error si movés ese peso un poquito. Si la derivada es positiva,
# aumentar el peso empeora las cosas, así que hay que bajarlo. De ahí el
# **descenso** por gradiente:
#
# $$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$
#
# donde $\eta$ (eta) es la **tasa de aprendizaje**: cuán grande es cada paso.
#
# ### 3. Retropropagación: el gradiente de TODAS las capas
#
# En una red con capas, el gradiente de la primera capa depende del de la
# segunda. La **retropropagación** (*backpropagation*) no es más que aplicar la
# regla de la cadena del cálculo, de atrás para adelante, reutilizando lo ya
# calculado.
#
# Suena imponente. Son cuatro líneas de NumPy.

# %% [markdown]
# ### La red completa, entera, sin nada escondido

# %%
class RedNeuronal:
    """Red de una capa oculta, escrita a mano. Clasificación binaria.

    Arquitectura:  entradas -> capa oculta (tanh) -> salida (sigmoide)
    """

    def __init__(self, n_entradas, n_ocultas, semilla=238):
        generador = np.random.default_rng(semilla)
        # Inicialización de Xavier: la escala depende del tamaño de la capa.
        # Si arrancaran todos en cero, todas las neuronas aprenderían lo mismo.
        self.W1 = generador.normal(0, np.sqrt(1.0 / n_entradas), (n_entradas, n_ocultas))
        self.b1 = np.zeros((1, n_ocultas))
        self.W2 = generador.normal(0, np.sqrt(1.0 / n_ocultas), (n_ocultas, 1))
        self.b2 = np.zeros((1, 1))
        self.historial = []

    # ---------------- paso hacia adelante ----------------
    def adelante(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.p = sigmoide(self.z2)
        return self.p

    def perdida(self, p, y):
        p = np.clip(p, 1e-9, 1 - 1e-9)     # evita log(0)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # ---------------- paso hacia atrás ----------------
    def atras(self, X, y):
        n = X.shape[0]
        # Derivada de la entropía cruzada compuesta con la sigmoide: se simplifica
        # a esta resta. Es una de las razones por las que se usan juntas.
        dz2 = (self.p - y) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1 ** 2)     # derivada de tanh
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    # ---------------- entrenamiento ----------------
    def entrenar(self, X, y, epocas=2000, tasa=0.5, cada=100, verboso=True):
        for epoca in range(1, epocas + 1):
            p = self.adelante(X)
            L = self.perdida(p, y)
            dW1, db1, dW2, db2 = self.atras(X, y)

            self.W1 -= tasa * dW1
            self.b1 -= tasa * db1
            self.W2 -= tasa * dW2
            self.b2 -= tasa * db2

            self.historial.append(L)
            if verboso and (epoca % cada == 0 or epoca == 1):
                print(f"  época {epoca:5d}   pérdida {L:.5f}")
        return self

    def predecir(self, X, umbral=0.5):
        return (self.adelante(X) > umbral).astype(int)


print("Red definida: 4 líneas hacia adelante, 6 hacia atrás.")

# %% [markdown]
# ### Antes de confiar: verificar el gradiente
#
# El error más común al escribir retropropagación a mano es equivocarse en una
# derivada. Lo peor es que **la red igual entrena**, solo que peor, y nunca te
# enterás.
#
# Hay una forma de verificarlo que conviene conocer: comparar el gradiente
# analítico (el que calculamos con la regla de la cadena) contra el gradiente
# **numérico**, obtenido moviendo cada peso un poquito y midiendo cuánto cambia
# la pérdida. Si coinciden hasta la sexta cifra, las derivadas están bien.

# %%
def verificar_gradiente(red, X, y, epsilon=1e-6):
    red.adelante(X)
    dW1_analitico = red.atras(X, y)[0]

    dW1_numerico = np.zeros_like(red.W1)
    for i in range(red.W1.shape[0]):
        for j in range(red.W1.shape[1]):
            original = red.W1[i, j]
            red.W1[i, j] = original + epsilon
            mas = red.perdida(red.adelante(X), y)
            red.W1[i, j] = original - epsilon
            menos = red.perdida(red.adelante(X), y)
            red.W1[i, j] = original
            dW1_numerico[i, j] = (mas - menos) / (2 * epsilon)

    diferencia = (np.abs(dW1_analitico - dW1_numerico).max()
                  / max(np.abs(dW1_analitico).max(), 1e-12))
    return diferencia


red_prueba = RedNeuronal(n_entradas=2, n_ocultas=4)
error_relativo = verificar_gradiente(red_prueba, X_xor, y_xor)
print(f"Diferencia relativa entre gradiente analítico y numérico: {error_relativo:.2e}")
check("Las derivadas de la retropropagación están bien", error_relativo < 1e-5,
      "por debajo de 1e-5 se considera correcto")

# %% [markdown]
# ---
# ## Bloque 4 — Práctica · Resolver el XOR (20 minutos)

# %%
red_xor = RedNeuronal(n_entradas=2, n_ocultas=4)
red_xor.entrenar(X_xor, y_xor, epocas=3000, tasa=0.8, cada=500)

print("\nPredicciones finales:")
for entrada, esperado, probabilidad in zip(X_xor, y_xor.ravel(), red_xor.adelante(X_xor).ravel()):
    print(f"  {entrada.astype(int)} -> esperado {int(esperado)}   "
          f"la red dice {probabilidad:.4f}")

check("La red resolvió el XOR",
      bool((red_xor.predecir(X_xor).ravel() == y_xor.ravel()).all()))

# %% [markdown]
# ### Mirá la frontera que aprendió
#
# El perceptrón solo podía trazar una recta. La red con una capa oculta traza una
# frontera curva, porque **combina varias rectas** —una por cada neurona
# oculta— a través de la no linealidad.

# %%
paso = 0.01
xx, yy = np.meshgrid(np.arange(-0.3, 1.31, paso), np.arange(-0.3, 1.31, paso))
malla = np.c_[xx.ravel(), yy.ravel()]
zz = red_xor.adelante(malla).reshape(xx.shape)

fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.4))

izq.plot(red_xor.historial, color=SERIE[0])
izq.set_xlabel("época")
izq.set_ylabel("pérdida")
izq.set_title("La pérdida bajando durante el entrenamiento")

der.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.75)
der.contour(xx, yy, zz, levels=[0.5], colors=["#0b0b0b"], linewidths=1.6)
for clase, marca in [(0, "o"), (1, "s")]:
    puntos = X_xor[y_xor.ravel() == clase]
    der.scatter(puntos[:, 0], puntos[:, 1], s=200, marker=marca,
                facecolors="white", edgecolors="#0b0b0b", linewidths=2, zorder=3)
der.set_xlabel("x₁")
der.set_ylabel("x₂")
der.grid(False)
der.set_title("La frontera de decisión que aprendió")

plt.tight_layout()
plt.show()

# %% [markdown]
# La línea negra es la frontera de decisión. **No es una recta**, y por eso puede
# dejar los dos círculos de un lado y los dos cuadrados del otro. Eso es
# exactamente lo que el perceptrón no podía hacer, y lo único que hizo falta
# fueron cuatro neuronas en el medio.

# %% [markdown]
# ---
# ## Bloque 5 — Práctica · La red sobre los datos de la silobolsa (30 minutos)
#
# Ahora el problema real. Y acá aparece algo que con árboles no hacía falta y con
# redes es **obligatorio**: la normalización.
#
# Las características de la silobolsa tienen escalas muy distintas: el CO₂ está
# en cientos, la temperatura en decenas, la pendiente puede ser negativa. Una red
# neuronal suma todo eso con pesos: si una variable es mil veces más grande que
# otra, domina la suma y las demás no se enteran de que existen.
#
# La receta es la **estandarización**: a cada columna se le resta su promedio y se
# la divide por su desvío. Todas quedan centradas en 0 con desvío 1.
#
# > **Ojo con el detalle que arruina el experimento:** el promedio y el desvío se
# > calculan **solo sobre el conjunto de entrenamiento**, y con esos mismos números
# > se transforma el de prueba. Si los calculás sobre todo el conjunto, estás
# > filtrando información del futuro hacia el pasado. Es fuga de datos otra vez,
# > con otro disfraz.

# %%
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

X_ent = datos[CARACTERISTICAS].iloc[:corte].to_numpy()
X_pru = datos[CARACTERISTICAS].iloc[corte:].to_numpy()
y_ent = datos["riesgo_24h"].iloc[:corte].to_numpy().reshape(-1, 1).astype(float)
y_pru = datos["riesgo_24h"].iloc[corte:].to_numpy().reshape(-1, 1).astype(float)

# Estandarización, con los estadísticos del ENTRENAMIENTO solamente.
promedio = X_ent.mean(axis=0)
desvio = X_ent.std(axis=0)
Xn_ent = (X_ent - promedio) / desvio
Xn_pru = (X_pru - promedio) / desvio

print("Escala de las características ANTES de normalizar:")
print(pd.DataFrame({"promedio": promedio, "desvío": desvio},
                   index=CARACTERISTICAS).round(1).to_string())
print(f"\nDESPUÉS: promedio {Xn_ent.mean():.3f}, desvío {Xn_ent.std():.3f}")

# %% [markdown]
# ### Con normalización y sin ella, lado a lado

# %%
from sklearn.metrics import f1_score, precision_score, recall_score

resultados = {}
for etiqueta, Xe, Xp in [("SIN normalizar", X_ent, X_pru),
                         ("Normalizada", Xn_ent, Xn_pru)]:
    red = RedNeuronal(n_entradas=len(CARACTERISTICAS), n_ocultas=8)
    red.entrenar(Xe, y_ent, epocas=3000, tasa=0.5, verboso=False)
    pred = red.predecir(Xp).ravel()
    resultados[etiqueta] = {
        "pérdida final": red.historial[-1],
        "precisión": precision_score(y_pru.ravel(), pred, zero_division=0),
        "sensibilidad": recall_score(y_pru.ravel(), pred, zero_division=0),
        "F1": f1_score(y_pru.ravel(), pred, zero_division=0),
    }
    if etiqueta == "Normalizada":
        red_final = red

pd.DataFrame(resultados).round(3)

# %% [markdown]
# La diferencia no es de matiz. Sin normalizar, la red arranca con sumas del
# orden de los cientos, la sigmoide se satura (queda pegada en 0 o en 1), el
# gradiente se hace casi cero y **el entrenamiento no avanza**. Es el problema del
# *gradiente que se desvanece*, en su versión más básica.
#
# > Con árboles esto no pasa, porque un árbol solo compara valores contra
# > umbrales y no le importa la escala. Es una de las razones por las que
# > conviene empezar por árboles cuando uno recién arranca.

# %%
fig, ax = plt.subplots()
ax.plot(red_final.historial, color=SERIE[0], label="red normalizada")
ax.set_xlabel("época")
ax.set_ylabel("pérdida (entropía cruzada)")
ax.legend()
titular(ax, "¿La red está aprendiendo o se quedó trabada?",
        "Una curva que baja y se aplana indica que convergió. Una curva plana desde el arranque indica un problema.")
plt.show()

# %% [markdown]
# ### Y ahora, la comparación que importa

# %%
from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier(max_depth=3, random_state=238)
arbol.fit(X_ent, y_ent.ravel())
pred_arbol = arbol.predict(X_pru)
pred_red = red_final.predecir(Xn_pru).ravel()

final = pd.DataFrame({
    nombre: {
        "precisión": precision_score(y_pru.ravel(), pred, zero_division=0),
        "sensibilidad": recall_score(y_pru.ravel(), pred, zero_division=0),
        "F1": f1_score(y_pru.ravel(), pred, zero_division=0),
    }
    for nombre, pred in [("Árbol de decisión (A-4)", pred_arbol),
                         ("Red neuronal (este cuaderno)", pred_red)]
}).round(3)
final

# %% [markdown]
# ### La conclusión incómoda, que es la correcta
#
# Con este problema y estos datos, la red **no le saca ventaja apreciable al
# árbol** — y el árbol tiene tres ventajas que la red no va a tener nunca acá:
# se lee, se transcribe a C, y se entrena en milisegundos.
#
# Eso no significa que las redes neuronales no sirvan. Significa que **sirven
# cuando el problema tiene la forma que las hace necesarias**: muchas variables
# que interactúan de manera complicada, muchísimos datos, y relaciones que ningún
# umbral captura. Cinco características construidas a mano sobre 90 días de
# datos no es ese caso.
#
# > **Lo que ganaste en este cuaderno no es un clasificador mejor. Es entender
# > qué hay adentro.** Cuando en A-7 aparezca un modelo de lenguaje, vas a
# > reconocer las mismas piezas: combinación lineal, activación, pérdida,
# > gradiente. No hay nada más.

# %% [markdown]
# ---
# ## Bloque 6 — Ejercicios

# %% [markdown]
# ### Ejercicio A6.1 [B] — La tasa de aprendizaje
#
# Entrená la red sobre el XOR con cuatro tasas distintas y guardá en
# `perdidas_finales` un diccionario `{tasa: pérdida_final}` para las tasas
# `0.001`, `0.1`, `1.0` y `50.0`, con 2000 épocas cada una.
#
# Después contestá en la celda de texto qué pasa en cada extremo.

# %%
# TU CÓDIGO ACÁ
perdidas_finales = {}

# %%
check("Probaste las cuatro tasas", set(perdidas_finales) == {0.001, 0.1, 1.0, 50.0})
if set(perdidas_finales) == {0.001, 0.1, 1.0, 50.0}:
    check("Con tasa 0.001 la red casi no aprende",
          perdidas_finales[0.001] > 0.5,
          "con pasos tan chicos, 2000 épocas no alcanzan para llegar a ningún lado")
    check("Con tasa 1.0 la red aprende bien", perdidas_finales[1.0] < 0.1)
    print()
    for t in sorted(perdidas_finales):
        print(f"   tasa {t:>7} -> pérdida final {perdidas_finales[t]:.5f}")

# %% [markdown]
# **Tu respuesta:** *(doble clic para editar)*
#
# - Con una tasa muy chica pasa que…
# - Con una tasa muy grande pasa que…
# - La analogía que se me ocurre para explicarlo es…

# %% [markdown]
# ### Ejercicio A6.2 [I] — ¿Cuántas neuronas ocultas hacen falta?
#
# El XOR se resolvió con 4 neuronas ocultas. La pregunta obvia es cuántas hacen
# falta como mínimo. La pregunta correcta es otra, y este ejercicio te va a
# mostrar por qué.
#
# En teoría, **dos** neuronas ocultas alcanzan: el XOR se puede escribir como
# `(x1 OR x2) Y NO (x1 AND x2)`, o sea la combinación de dos rectas, una por
# neurona.
#
# Pero el entrenamiento arranca de pesos al azar, y de dónde arranca cambia a
# dónde llega. Así que no midas si funciona **una vez**: medí con qué frecuencia
# funciona.
#
# Construí `tasa_de_exito`: un diccionario `{n_ocultas: proporción_de_éxitos}`
# para 1, 2, 3 y 4 neuronas ocultas, entrenando con **las semillas 0 a 7**
# (`RedNeuronal(2, n, semilla=s)`), 3000 épocas y tasa 0.8. Un "éxito" es
# clasificar bien los cuatro casos.

# %%
# TU CÓDIGO ACÁ
tasa_de_exito = {}

# %%
check("Probaste las cuatro arquitecturas", set(tasa_de_exito) == {1, 2, 3, 4})
if set(tasa_de_exito) == {1, 2, 3, 4}:
    check("Con 1 neurona oculta nunca funciona", tasa_de_exito[1] == 0.0,
          "con una sola neurona oculta la red vuelve a ser, en la práctica, un perceptrón")
    check("Con 2 neuronas funciona sólo a veces", 0.0 < tasa_de_exito[2] < 1.0)
    check("Con 3 o más funciona siempre",
          tasa_de_exito[3] == 1.0 and tasa_de_exito[4] == 1.0)
    print()
    for n in sorted(tasa_de_exito):
        barra = "#" * int(tasa_de_exito[n] * 20)
        print(f"   {n} neurona(s) oculta(s): {tasa_de_exito[n]:5.1%}  {barra}")

# %% [markdown]
# ### Lo que muestra ese resultado
#
# Con dos neuronas ocultas la red **puede** resolver el XOR, pero la mitad de las
# veces se queda trabada en un **mínimo local**: un rincón del espacio de pesos
# donde cualquier cambio chico empeora la pérdida, aunque exista una solución
# mucho mejor en otro lado. El descenso por gradiente es un algoritmo miope; ve
# la pendiente donde está parado, no el mapa completo.
#
# Con tres o cuatro neuronas hay caminos de sobra para escaparse, y siempre
# llega.
#
# > **La conclusión práctica es contraintuitiva:** conviene darle a la red **más
# > capacidad de la mínima necesaria**, no por precisión, sino por **confiabilidad
# > del entrenamiento**. "El mínimo que puede funcionar" y "el mínimo que funciona
# > siempre" son dos números distintos, y el que importa es el segundo.
# >
# > Y hay una lección metodológica de fondo: **un resultado de una sola corrida
# > no es un resultado.** Si hubieras probado una vez con cada tamaño, te habrías
# > llevado una conclusión distinta según la semilla que te tocara.

# %% [markdown]
# ### Ejercicio A6.3 [I] — El XOR de tu proyecto
#
# Escribí un problema de **tu** proyecto que tenga forma de XOR: una condición que
# se cumple cuando **una** de dos cosas pasa, pero no cuando pasan las dos.
#
# Armá `X_mio` (4 filas, 2 columnas) e `y_mio` (4 filas, 1 columna) con los cuatro
# casos, entrenalos y verificá que la red los aprende.
#
# Ejemplos por proyecto:
# - **Riego:** regar si el suelo está seco **o** hace mucho calor, pero no si
#   están las dos (porque con suelo seco y mucho calor conviene esperar a la
#   noche para no evaporar el agua).
# - **Enchufe:** alertar si el consumo es alto **o** es horario nocturno, pero no
#   si son las dos (consumo alto de noche es la heladera, es normal).

# %%
# TU CÓDIGO ACÁ
X_mio = None
y_mio = None
descripcion_del_problema = ""

# %%
check("Definiste las cuatro combinaciones",
      X_mio is not None and np.asarray(X_mio).shape == (4, 2))
check("Definiste las cuatro respuestas",
      y_mio is not None and np.asarray(y_mio).shape == (4, 1))
check("Describiste el problema en palabras",
      len(descripcion_del_problema.split()) >= 15)
if X_mio is not None and y_mio is not None:
    _r = RedNeuronal(2, 4).entrenar(np.asarray(X_mio, dtype=float),
                                    np.asarray(y_mio, dtype=float),
                                    epocas=3000, tasa=0.8, verboso=False)
    check("La red aprendió tu problema",
          bool((_r.predecir(np.asarray(X_mio, dtype=float)).ravel()
                == np.asarray(y_mio).ravel()).all()))

# %% [markdown]
# ### Ejercicio A6.4 [A] — Agregar una segunda capa oculta
#
# Escribí `RedProfunda`, con **dos** capas ocultas en lugar de una. Vas a tener
# que:
#
# 1. Agregar `W2`, `b2` para la segunda capa oculta y renombrar la de salida.
# 2. Extender `adelante()` con un paso más.
# 3. Extender `atras()` propagando el gradiente una capa más hacia atrás.
# 4. **Verificar el gradiente**, como hicimos arriba. Si no lo verificás, no sabés
#    si está bien.
#
# Después compará contra la red de una capa sobre el XOR: ¿mejora? ¿empeora?
# ¿tarda más? Anotá lo que observes.

# %%
# TU CÓDIGO ACÁ
class RedProfunda:
    pass


# %% [markdown]
# **Tu observación:** *(doble clic para editar)*
#
# Con dos capas ocultas, sobre el XOR, observé que…

# %% [markdown]
# ---
# ## Cierre del cuaderno A-6
#
# **Lo que quedó instalado en tu cabeza:**
#
# - Una neurona es una combinación lineal más una activación no lineal. Sin la
#   segunda, apilar capas no agrega nada: lo demostramos con dos matrices.
# - El perceptrón no puede resolver el XOR, y el XOR es la forma de muchos
#   problemas reales de IoT.
# - Entrenar es: pérdida, gradiente, descenso. La retropropagación es la regla de
#   la cadena aplicada de atrás para adelante.
# - Las derivadas escritas a mano **se verifican** contra el gradiente numérico.
# - Las redes exigen normalizar; los árboles no. Y en problemas chicos, el árbol
#   suele ganar.
#
# **Checklist de entrega**
#
# - [ ] La verificación de gradiente en `[OK]`.
# - [ ] Las cuatro tasas de aprendizaje con su explicación (A6.1).
# - [ ] La tasa de éxito por tamaño de capa oculta, con la conclusión sobre
#       mínimos locales (A6.2).
# - [ ] El XOR de tu propio proyecto, aprendido por la red (A6.3).
# - [ ] La red de dos capas con su gradiente verificado (A6.4).
#
# **Sigue en:** `A7_Modelo_de_lenguaje_diminuto.ipynb` — donde entrenamos, en tu
# propia máquina y en menos de un minuto, un modelo de lenguaje.
