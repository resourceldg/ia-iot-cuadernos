# %% [markdown]
# # Anexo A-7 · Un modelo de lenguaje entrenado en tu propia máquina
#
# **Desarrollo de Sistemas de Inteligencia Artificial** — Trayecto F · 2.º año
# ISFT N.º 238 · Prof. Lucas Daniel Gómez · Ciclo lectivo 2026
#
# | | |
# |---|---|
# | **Vinculación** | Continuación de A-6 · marco conceptual para A-8 |
# | **Duración** | 150 minutos |
# | **Modalidad** | Individual o de a dos |
#
# ### Al terminar este cuaderno vas a poder
#
# 1. Explicar qué hace un modelo de lenguaje en una sola oración, y por qué esa
#    oración alcanza para entender a ChatGPT.
# 2. Entrenar dos modelos de lenguaje **en tu computadora, sin internet**: uno de
#    conteos y uno neuronal, y compararlos con una métrica.
# 3. Generar texto y controlar la generación con la **temperatura**.
# 4. Medir la calidad de un modelo de lenguaje con **perplejidad**.
# 5. Dimensionar honestamente la distancia entre esto y un modelo comercial, con
#    números.
#
# ### Antes de arrancar
#
# Todo lo de este cuaderno corre en **segundos** en cualquier notebook, sin GPU,
# sin descargar nada. Los modelos que vamos a entrenar tienen unos pocos miles de
# parámetros. Los modelos comerciales tienen cientos de miles de millones.
#
# Esa diferencia es enorme en escala y **nula en concepto**: hacen exactamente lo
# mismo, con las mismas piezas que programaste en A-6.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from estilo_grafico import aplicar_estilo, titular, SERIE, ESTADO, TINTA_SUAVE, SECUENCIAL

aplicar_estilo()
rng = np.random.default_rng(238)


def check(descripcion, condicion, pista=""):
    if condicion:
        print(f"  [OK]     {descripcion}")
    else:
        print(f"  [REVISAR] {descripcion}" + (f"\n            Pista: {pista}" if pista else ""))
    return bool(condicion)


print("Listo.")

# %% [markdown]
# ---
# ## Bloque 1 — Teoría · Qué es un modelo de lenguaje (20 minutos)
#
# La definición completa cabe en un renglón:
#
# > Un **modelo de lenguaje** es un sistema que, dado un texto, asigna una
# > probabilidad a cada símbolo que podría venir después.
#
# Eso es todo. No hay nada más. Todo lo que hace un asistente conversacional
# —responder, resumir, traducir, escribir código— sale de aplicar esa operación
# una y otra vez, agregando cada vez el símbolo elegido al final del texto y
# volviendo a preguntar.
#
# ### Lo que eso implica, y conviene tener claro desde el principio
#
# | Consecuencia | Por qué |
# |---|---|
# | **No "sabe" cosas: modela texto** | Aprende qué símbolos suelen seguir a cuáles, no hechos sobre el mundo |
# | **Puede inventar con total seguridad** | Una secuencia falsa puede ser altísimamente probable si "suena" a las que vio |
# | **No razona paso a paso salvo que el texto lo haga** | Por eso funciona pedirle que "piense en voz alta" |
# | **Su límite es el corpus** | Lo que no estaba en los datos de entrenamiento, no está |
#
# Vas a poder verificar las cuatro con el modelito que entrenamos acá abajo. Eso
# es lo valioso de hacerlo a escala chica: **las propiedades se ven a simple
# vista.**

# %% [markdown]
# ### El corpus
#
# Un modelo de lenguaje necesita texto. Vamos a usar el vocabulario técnico de la
# carrera más un vocabulario general de taller: son palabras con estructura
# interna clara (prefijos, sufijos, raíces), que es justo lo que un modelo de
# caracteres puede aprender.
#
# Va sin tildes ni eñes a propósito, para que el vocabulario quede chico y la
# tabla de bigramas se pueda mirar entera en una sola imagen.

# %%
CORPUS = """
sensor actuador microcontrolador transductor potenciometro amperimetro voltimetro
termistor fotorresistor higrometro barometro acelerometro magnetometro caudalimetro
resistencia capacitor inductor diodo transistor relevador optoacoplador regulador
placa circuito conductor aislante soldadura protoboard conector borne terminal
alimentacion tension corriente potencia energia consumo bateria cargador fuente
firmware software hardware middleware controlador biblioteca compilador interprete
programa funcion variable constante parametro argumento estructura arreglo puntero
registro memoria almacenamiento procesador nucleo hilo proceso interrupcion temporizador
protocolo paquete trama cabecera carga direccion puerto socket enlace pasarela
servidor cliente broker suscriptor publicador topico mensaje evento notificacion
red nodo enrutador conmutador antena cobertura alcance latencia ancho banda
seguridad confidencialidad integridad disponibilidad autenticidad autorizacion credencial
contrasena certificado cifrado descifrado firma resumen vulnerabilidad amenaza riesgo
mitigacion auditoria trazabilidad respaldo recuperacion aislamiento privilegio
dato medicion muestra muestreo frecuencia periodo resolucion precision exactitud
calibracion deriva ruido filtrado promedio mediana desviacion tendencia anomalia
modelo entrenamiento validacion prueba caracteristica objetivo prediccion inferencia
clasificacion regresion agrupamiento aprendizaje supervisado neurona capa peso
gradiente perdida optimizacion sobreajuste generalizacion metrica sensibilidad
automatizacion monitoreo telemetria tablero alerta umbral histeresis realimentacion
instalacion mantenimiento diagnostico reparacion documentacion informe presentacion
abrir aceite acelerar aceptar acero aclarar acomodar acompanar aconsejar
acordar activar actual acumular adaptar adelante ademas admitir adoptar
advertir afectar afirmar agregar agua ahorrar aire ajustar alambre alcanzar
alcohol alejar alerta alimentar almacenar alto altura aluminio amarillo
ambiente amplificar analizar ancho anillo anotar antes anterior anular apagar
aparato aparecer aplicar apoyar aprender apretar aprobar aprovechar archivo
arena armar arranque arriba articulo asegurar asignar aspecto atender atraso
aumentar automatico avanzar averiguar avisar ayudar azul bajar balance banda
barra base bastidor bloque bobina bomba borrar botella boton brazo brillo
bronce bucle buscar cable cadena caida caja calcular calentar calibrar calidad
calor cambiar camino campo canal cantidad capa capaz carbono carga carpeta
carrera cascara caudal causa cepillo cerrar chapa choque cilindro cinta claro
clave cobre codigo cola color columna combinar comenzar comparar completar
comprar comprender comun conducir conectar confirmar conjunto conocer
conseguir considerar construir consultar contacto contar contener continuar
contrato controlar convertir copia correa correcto correr cortar corto crear
crecer cristal cruzar cuadro cubierta cuenta cuerda cuerpo cuidado curva danio
dato deber decidir declarar dedicar defecto definir dejar delgado demanda
dentro depender derecho desarmar descargar describir desde desgaste destino
detalle detectar determinar diario dibujar diferencia dificil dimension
direccion disco disenar disponer distancia dividir doblar documento dominar
duracion duro efecto eficiencia eje ejemplo elegir elemento elevar eliminar
empezar empresa empujar encender encontrar energia enfriar enganche ensayo
entender entrada entregar enviar equipo error escala escribir escuchar espacio
especial espesor esperar espuma esquema estable estado estimar estirar estudio
etapa evaluar evitar exacto examinar excepto existir experiencia explicar
exponer extender exterior extremo fabrica facil factor falla familia fase
favor fecha fibra figura fijar filtro final fino fisico flecha flujo forma
formar fuego fuerza funcionar fundir futuro ganancia general generar girar
golpe grado grafico grande grasa gravedad grosor grupo guardar guia hacer
hallar herramienta hierro hilo hoja hondo horizontal hueco humedad idea
identificar igual imagen impedir importante imprimir impulso incluir indicar
industria inferior influir informar ingreso inicio instalar instante intentar
interior interno interruptor introducir invertir juego junta juntar lado
lamina largo lateral lavar lectura lento levantar liberar libro ligero limite
limpiar linea liquido lista llave llegar llenar llevar local lograr longitud
lote lugar madera magnitud malla mando manejar manera manguera mano mantener
maquina marca marco masa material medio medir mejorar menor mensual metal
metodo metro mezcla milimetro minimo modificar modo molde momento montaje
mostrar motor mover mucho muestra natural necesario negro nivel nombre norma
normal notar nuevo numero objeto observar obtener ocupar ocurrir oficina
ofrecer operar orden organizar origen oscuro palanca panel papel para parar
parte partir pasar paso pedir pegar pensar perder perfil periodo permitir peso
pieza pintura placa plano plastico plazo poner porcion posible posicion precio
preparar presion primero probar problema producir profundo prohibir promedio
proponer proteger prueba pulso punta punto quedar quemar quitar rango rapido
rayo razon reaccion realizar recibir recoger reducir reemplazar referencia
reflejo regla regular relacion rellenar remover rendir reparar repetir reponer
representar requerir resolver resorte respetar responder resultado retirar
reunir revisar rigido rodar rodillo romper rosca rotar rueda ruido sacar
salida salir salto sector seguir segundo seleccionar sencillo sentido separar
serie servir simple sistema sitio sobrar solido soltar solucion sonido soporte
subir suficiente sujetar sumar superficie suponer tabla tamano tapa tarea
tarjeta tecla tecnica temperatura tener terminar textura tiempo tipo tirar
tocar tomar tornillo total trabajo traer tramo tratar tubo turno ubicar unidad
unir usar util vacio valor valvula vapor variar velocidad vender ventaja
ventana verificar version vertical viaje vidrio viejo volumen volver zona
"""

palabras = sorted(set(CORPUS.split()))
rng.shuffle(palabras)

print(f"{len(palabras)} palabras en el corpus")
print(f"Longitud promedio: {np.mean([len(p) for p in palabras]):.1f} caracteres")
print(f"\nPrimeras diez: {', '.join(palabras[:10])}")

# %% [markdown]
# ### Tokenización: del texto a los números
#
# Un modelo no manipula letras, manipula números. **Tokenizar** es partir el texto
# en unidades y asignarle un número a cada una.
#
# Los modelos grandes usan *subpalabras* (trozos como `micro`, `contro`, `lador`).
# Nosotros vamos a usar la unidad más simple posible: **el carácter**. Es peor
# para la calidad y muchísimo mejor para entender.
#
# Agregamos un símbolo especial, `.`, que marca a la vez el arranque y el final
# de una palabra. Sin un símbolo de fin, el modelo no tendría cómo decidir cuándo
# parar de generar.

# %%
FIN = "."
vocabulario = [FIN] + sorted(set("".join(palabras)))
a_indice = {c: i for i, c in enumerate(vocabulario)}
a_caracter = {i: c for c, i in a_indice.items()}
V = len(vocabulario)

print(f"Vocabulario de {V} símbolos: {' '.join(vocabulario)}")
print(f"\n'sensor' se codifica como: {[a_indice[c] for c in 'sensor']}")

# %% [markdown]
# ---
# ## Bloque 2 — El modelo más simple que existe: contar (30 minutos)
#
# Antes de cualquier red neuronal, el modelo de lenguaje más elemental es un
# **bigrama**: mirar solo el carácter anterior y preguntarse, según el corpus,
# qué caracteres lo siguieron y con qué frecuencia.
#
# No hay entrenamiento, no hay gradiente. Hay una tabla de conteos.

# %%
conteos = np.zeros((V, V), dtype=np.float64)

for palabra in palabras:
    secuencia = [FIN] + list(palabra) + [FIN]
    for actual, siguiente in zip(secuencia, secuencia[1:]):
        conteos[a_indice[actual], a_indice[siguiente]] += 1

# Suavizado: se le suma 1 a todo para que ninguna transición tenga probabilidad
# exactamente cero. Sin esto, un par que nunca apareció haría explotar el
# logaritmo al calcular la pérdida.
probabilidades = (conteos + 1) / (conteos + 1).sum(axis=1, keepdims=True)

print(f"Después de '{FIN}' (arranque de palabra), los caracteres más probables:")
inicio = pd.Series(probabilidades[a_indice[FIN]], index=vocabulario).sort_values(ascending=False)
print(inicio.head(8).round(3).to_string())

print("\nDespués de 'q':")
tras_q = pd.Series(probabilidades[a_indice["q"]], index=vocabulario).sort_values(ascending=False)
print(tras_q.head(4).round(3).to_string())

# %% [markdown]
# Mirá el segundo bloque: después de una `q`, el modelo le da casi toda la
# probabilidad a la `u`. **Nadie le enseñó ortografía**; simplemente en el corpus
# la `q` siempre estuvo seguida de `u`. Eso es todo lo que hace un modelo de
# lenguaje, a cualquier escala.

# %%
fig, ax = plt.subplots(figsize=(8.5, 7))
imagen = ax.imshow(probabilidades, cmap="Blues", vmin=0, vmax=0.5)
ax.set_xticks(range(V), vocabulario, fontsize=8)
ax.set_yticks(range(V), vocabulario, fontsize=8)
ax.set_xlabel("carácter siguiente")
ax.set_ylabel("carácter actual")
ax.grid(False)
fig.colorbar(imagen, ax=ax, shrink=0.75, label="probabilidad")
titular(ax, "La tabla completa del modelo de bigramas",
        "Cada fila es una distribución de probabilidad: suma 1.")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Esa imagen es el modelo entero.** Todos sus parámetros están ahí: una matriz
# de 27×27 números, uno por cada par de caracteres posible. Un modelo comercial
# tiene la misma naturaleza; lo que tiene es más contexto y muchísimos más
# parámetros.

# %% [markdown]
# ### Generar: muestrear de la distribución
#
# Generar texto es repetir: mirar el símbolo actual, tomar su fila de
# probabilidades, **sortear** un símbolo con esas probabilidades, y repetir hasta
# que salga el símbolo de fin.

# %%
def generar_bigrama(cantidad=12, semilla=238):
    generador = np.random.default_rng(semilla)
    salidas = []
    for _ in range(cantidad):
        indice, letras = a_indice[FIN], []
        while True:
            indice = generador.choice(V, p=probabilidades[indice])
            if indice == a_indice[FIN] or len(letras) > 20:
                break
            letras.append(a_caracter[indice])
        salidas.append("".join(letras))
    return salidas


print("Palabras inventadas por el modelo de bigramas:\n")
print("   " + "   ".join(generar_bigrama()))

# %% [markdown]
# Son impronunciables, y eso es esperable: **el modelo solo mira un carácter hacia
# atrás**. Cuando está eligiendo la quinta letra ya se olvidó por completo de las
# primeras tres. Con esa memoria no hay forma de armar una palabra coherente.
#
# El arreglo obvio es mirar más contexto. Y ahí es donde hace falta la red.

# %% [markdown]
# ---
# ## Bloque 3 — Teoría · La métrica: perplejidad (15 minutos)
#
# Para comparar dos modelos de lenguaje hace falta un número. El que se usa es la
# **pérdida de entropía cruzada** —la misma de A-6, ahora con muchas clases en
# lugar de dos— y su versión más interpretable, la **perplejidad**:
#
# $$\text{perplejidad} = e^{L}$$
#
# La perplejidad se lee así: **entre cuántas opciones equivalentes está dudando el
# modelo en cada paso.** Una perplejidad de 1 significa certeza absoluta. Una
# perplejidad igual al tamaño del vocabulario significa que el modelo no aprendió
# nada: está eligiendo al azar entre todos los símbolos.
#
# Con nuestro vocabulario, un modelo que no sabe nada tendría perplejidad ≈ V.

# %%
def perdida_bigrama(lista_de_palabras):
    total, cantidad = 0.0, 0
    for palabra in lista_de_palabras:
        secuencia = [FIN] + list(palabra) + [FIN]
        for actual, siguiente in zip(secuencia, secuencia[1:]):
            total += -np.log(probabilidades[a_indice[actual], a_indice[siguiente]])
            cantidad += 1
    return total / cantidad


L_bigrama = perdida_bigrama(palabras)
print(f"Modelo que elige al azar: pérdida {np.log(V):.4f}  perplejidad {V:.1f}")
print(f"Modelo de bigramas:       pérdida {L_bigrama:.4f}  perplejidad {np.exp(L_bigrama):.1f}")
print(f"\nDe dudar entre {V} símbolos pasamos a dudar entre "
      f"{np.exp(L_bigrama):.0f}. Contar sirve.")

# %% [markdown]
# ---
# ## Bloque 4 — El modelo neuronal: mirar más atrás (40 minutos)
#
# Ahora una red que mira **tres caracteres** de contexto en lugar de uno. La
# arquitectura es la de A-6 con dos agregados:
#
# **1. Una tabla de embeddings.** En vez de representar cada carácter con su
# número (lo que le diría a la red que la `b` está "entre" la `a` y la `c`, que es
# falso), cada carácter se representa con un vector de números que **la red
# aprende sola**. Caracteres que se comportan parecido terminan con vectores
# parecidos.
#
# **2. Salida softmax.** En A-6 la salida era una sigmoide para dos clases. Acá
# hay que repartir probabilidad entre V opciones, y para eso está la función
# *softmax*: exponencia todo y normaliza para que sume 1.
#
# Todo lo demás —tanh, pérdida, gradiente, descenso— es idéntico.

# %%
CONTEXTO = 3      # cuántos caracteres mira hacia atrás
DIM_EMB = 6       # tamaño del vector de cada carácter
OCULTAS = 32      # neuronas de la capa oculta


def armar_ejemplos(lista_de_palabras):
    """Convierte palabras en pares (contexto de 3 caracteres -> siguiente)."""
    entradas, salidas = [], []
    for palabra in lista_de_palabras:
        ventana = [a_indice[FIN]] * CONTEXTO
        for caracter in list(palabra) + [FIN]:
            siguiente = a_indice[caracter]
            entradas.append(ventana)
            salidas.append(siguiente)
            ventana = ventana[1:] + [siguiente]
    return np.array(entradas), np.array(salidas)


# Partimos las PALABRAS, no los ejemplos: si la misma palabra aportara ejemplos a
# los dos conjuntos, el modelo estaría evaluándose sobre algo que ya vio.
corte = int(len(palabras) * 0.85)
palabras_ent, palabras_val = palabras[:corte], palabras[corte:]
X_ent, y_ent = armar_ejemplos(palabras_ent)
X_val, y_val = armar_ejemplos(palabras_val)

print(f"{len(palabras_ent)} palabras de entrenamiento -> {len(X_ent)} ejemplos")
print(f"{len(palabras_val)} palabras de validación    -> {len(X_val)} ejemplos")
print(f"\nPrimer ejemplo: contexto {[a_caracter[i] for i in X_ent[0]]} "
      f"-> siguiente '{a_caracter[y_ent[0]]}'")

# %%
class ModeloDeLenguaje:
    """Modelo de lenguaje de caracteres con contexto fijo. NumPy puro.

    embeddings -> concatenar -> capa oculta (tanh) -> softmax sobre el vocabulario
    """

    def __init__(self, V, contexto, dim_emb, ocultas, semilla=238):
        gen = np.random.default_rng(semilla)
        self.V, self.contexto = V, contexto
        self.C = gen.normal(0, 1.0, (V, dim_emb))
        entradas = contexto * dim_emb
        self.W1 = gen.normal(0, np.sqrt(2.0 / entradas), (entradas, ocultas))
        self.b1 = np.zeros(ocultas)
        self.W2 = gen.normal(0, np.sqrt(1.0 / ocultas), (ocultas, V))
        self.b2 = np.zeros(V)
        self.historial = []

    def adelante(self, X):
        self.emb = self.C[X].reshape(len(X), -1)     # (n, contexto*dim_emb)
        self.h = np.tanh(self.emb @ self.W1 + self.b1)
        logits = self.h @ self.W2 + self.b2
        # Softmax estable: se resta el máximo antes de exponenciar para que
        # exp() no desborde con logits grandes.
        logits = logits - logits.max(axis=1, keepdims=True)
        exponenciales = np.exp(logits)
        self.p = exponenciales / exponenciales.sum(axis=1, keepdims=True)
        return self.p

    def perdida(self, p, y):
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    def paso(self, X, y, tasa):
        n = len(X)
        p = self.adelante(X)
        L = self.perdida(p, y)

        dlogits = p.copy()
        dlogits[np.arange(n), y] -= 1.0      # derivada de softmax + entropía cruzada
        dlogits /= n

        dW2 = self.h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dz1 = dh * (1 - self.h ** 2)
        dW1 = self.emb.T @ dz1
        db1 = dz1.sum(axis=0)
        demb = (dz1 @ self.W1.T).reshape(n, self.contexto, -1)

        dC = np.zeros_like(self.C)
        # np.add.at acumula bien cuando un mismo carácter aparece varias veces.
        np.add.at(dC, X, demb)

        for parametro, gradiente in [(self.C, dC), (self.W1, dW1), (self.b1, db1),
                                     (self.W2, dW2), (self.b2, db2)]:
            parametro -= tasa * gradiente
        return L

    def cantidad_de_parametros(self):
        return sum(p.size for p in [self.C, self.W1, self.b1, self.W2, self.b2])

    def copiar_pesos(self):
        return [p.copy() for p in (self.C, self.W1, self.b1, self.W2, self.b2)]

    def restaurar_pesos(self, copia):
        self.C, self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in copia]


modelo = ModeloDeLenguaje(V, CONTEXTO, DIM_EMB, OCULTAS)
print(f"El modelo tiene {modelo.cantidad_de_parametros():,} parámetros.")

# %% [markdown]
# ### Parada temprana
#
# En A-4 vimos las dos curvas de error separándose y le pusimos nombre:
# sobreajuste. Acá vamos a hacer algo al respecto en lugar de solo mirarlo.
#
# La técnica se llama **parada temprana** (*early stopping*) y es la más simple
# que existe: cada tantas épocas se mide la pérdida en validación y se guarda una
# copia de los pesos si mejoró. Al terminar, se restauran los mejores pesos y se
# tira el resto del entrenamiento.
#
# Con 730 palabras y un modelo de unos 1600 parámetros, el sobreajuste llega
# rápido: sin esto, el modelo termina peor que la tabla de conteos.

# %%
import time

TASA, EPOCAS, CADA = 0.5, 2500, 100
inicio = time.time()
mejor_perdida, mejores_pesos, mejor_epoca = float("inf"), None, 0

for epoca in range(1, EPOCAS + 1):
    L = modelo.paso(X_ent, y_ent, TASA)
    if epoca % CADA == 0 or epoca == 1:
        L_val = modelo.perdida(modelo.adelante(X_val), y_val)
        modelo.historial.append({"época": epoca, "entrenamiento": L, "validación": L_val})
        if L_val < mejor_perdida:
            mejor_perdida, mejores_pesos, mejor_epoca = L_val, modelo.copiar_pesos(), epoca
        if epoca % 500 == 0 or epoca == 1:
            print(f"  época {epoca:5d}   entrenamiento {L:.4f}   validación {L_val:.4f}")

modelo.restaurar_pesos(mejores_pesos)
print(f"\nEntrenamiento completo en {time.time() - inicio:.1f} segundos, en CPU.")
print(f"Mejor validación: {mejor_perdida:.4f} en la época {mejor_epoca}.")
print(f"Se descartaron las últimas {EPOCAS - mejor_epoca} épocas: "
      f"a partir de ahí el modelo empeoraba en datos nuevos.")

# %%
L_final_ent = modelo.perdida(modelo.adelante(X_ent), y_ent)
L_final_val = modelo.perdida(modelo.adelante(X_val), y_val)
L_bigrama_val = perdida_bigrama(palabras_val)

resumen = pd.DataFrame({
    "Al azar": {"pérdida": np.log(V), "perplejidad": V},
    "Bigramas (contexto 1)": {"pérdida": L_bigrama_val,
                              "perplejidad": np.exp(L_bigrama_val)},
    "Red neuronal (contexto 3)": {"pérdida": L_final_val,
                                  "perplejidad": np.exp(L_final_val)},
}).round(3)
print("Evaluados sobre las palabras de VALIDACIÓN, que ningún modelo vio:\n")
resumen

# %%
historial = pd.DataFrame(modelo.historial)

fig, ax = plt.subplots()
ax.plot(historial["época"], historial["entrenamiento"], color=SERIE[0],
        label="entrenamiento")
ax.plot(historial["época"], historial["validación"], color=SERIE[1],
        label="validación")
ax.axhline(L_bigrama_val, color=ESTADO["grave"], linestyle="--", linewidth=1.4)
ax.text(EPOCAS * 0.45, L_bigrama_val + 0.04, "modelo de bigramas",
        color=ESTADO["grave"], fontsize=9)
ax.axvline(mejor_epoca, color=TINTA_SUAVE, linestyle=":", linewidth=1.4)
ax.text(mejor_epoca + 40, ax.get_ylim()[1] * 0.92, "parada temprana",
        color=TINTA_SUAVE, fontsize=9)
ax.set_xlabel("época")
ax.set_ylabel("pérdida")
ax.legend()
titular(ax, "¿La red le gana al modelo de conteos?",
        "Y en el camino: ¿en qué momento empieza a memorizar en lugar de generalizar?")
plt.show()

# %% [markdown]
# ---
# ## Bloque 5 — Generar, y la temperatura (25 minutos)
#
# Ahora generamos con la red. Y acá aparece un parámetro que seguramente viste
# nombrado en cualquier interfaz de IA sin que nadie explique qué hace: la
# **temperatura**.
#
# Antes de sortear el próximo símbolo, se dividen los logits por un número `T`:
#
# - `T < 1` **agranda** las diferencias: lo probable se vuelve casi seguro. Texto
#   conservador, repetitivo, aburrido.
# - `T = 1` usa la distribución tal cual la aprendió el modelo.
# - `T > 1` **achata** las diferencias: lo improbable gana chances. Texto
#   arriesgado, creativo, y con más disparates.
#
# La temperatura **no cambia el modelo**: cambia cómo se lo consulta.

# %%
def generar(modelo, cantidad=10, temperatura=1.0, semilla=7):
    gen = np.random.default_rng(semilla)
    salidas = []
    for _ in range(cantidad):
        ventana = [a_indice[FIN]] * CONTEXTO
        letras = []
        while len(letras) < 20:
            emb = modelo.C[np.array([ventana])].reshape(1, -1)
            h = np.tanh(emb @ modelo.W1 + modelo.b1)
            logits = (h @ modelo.W2 + modelo.b2).ravel() / temperatura
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            indice = gen.choice(modelo.V, p=p)
            if indice == a_indice[FIN]:
                break
            letras.append(a_caracter[indice])
            ventana = ventana[1:] + [indice]
        salidas.append("".join(letras))
    return salidas


for T in [0.3, 0.7, 1.0, 1.5, 2.5]:
    palabras_generadas = generar(modelo, cantidad=8, temperatura=T)
    print(f"T = {T:<4}  {'  '.join(palabras_generadas)}")

# %% [markdown]
# ### Leé esa salida con atención
#
# Con temperatura baja salen fragmentos que parecen castellano técnico y se
# repiten mucho. Con temperatura alta salen cosas impronunciables. En el medio
# aparecen palabras que **no existen pero podrían existir**: tienen sufijos
# plausibles, alternancia de vocales y consonantes, largo razonable.
#
# > **Eso es una alucinación, en su forma más pura y más fácil de ver.** El
# > modelo no está mintiendo ni fallando: está haciendo exactamente aquello para
# > lo que fue entrenado, que es producir secuencias probables. Que una secuencia
# > sea probable y que sea verdadera son dos propiedades distintas, y el modelo
# > solo tiene acceso a la primera.
# >
# > Cuando un asistente comercial te inventa una cita bibliográfica con autor,
# > año y editorial perfectamente verosímiles, está haciendo esto mismo. La
# > escala cambia; el mecanismo no.

# %% [markdown]
# ---
# ## Bloque 6 — La distancia hasta un modelo comercial (15 minutos)
#
# Con lo que ya sabés, se puede dimensionar la diferencia con números en lugar de
# con asombro.

# %%
comparacion = pd.DataFrame([
    {"modelo": "El de este cuaderno", "parámetros": modelo.cantidad_de_parametros(),
     "contexto": f"{CONTEXTO} caracteres", "unidad": "carácter",
     "entrenamiento": "segundos, una CPU sin GPU"},
    {"modelo": "GPT-2 chico (2019)", "parámetros": 124_000_000,
     "contexto": "1024 tokens", "unidad": "subpalabra",
     "entrenamiento": "días, varias GPU"},
    {"modelo": "Un modelo abierto chico de hoy", "parámetros": 500_000_000,
     "contexto": "32 000 tokens", "unidad": "subpalabra",
     "entrenamiento": "semanas, cientos de GPU"},
    {"modelo": "Un modelo comercial grande", "parámetros": 500_000_000_000,
     "contexto": "200 000 tokens", "unidad": "subpalabra",
     "entrenamiento": "meses, decenas de miles de GPU"},
]).set_index("modelo")
comparacion["parámetros"] = comparacion["parámetros"].map(lambda v: f"{v:,}")
comparacion

# %% [markdown]
# ### Las tres diferencias reales
#
# 1. **Escala.** Del orden de 10⁷ veces más parámetros y datos. Con esa escala
#    aparecen capacidades que a escala chica sencillamente no están.
# 2. **La arquitectura.** Los modelos modernos usan *transformers*, con un
#    mecanismo de **atención** que les permite mirar cualquier parte del contexto
#    en lugar de una ventana fija de tres caracteres. Es la diferencia más
#    importante después de la escala.
# 3. **El ajuste posterior.** Un modelo recién entrenado solo completa texto. Para
#    que conteste preguntas y siga instrucciones hace falta una etapa adicional de
#    ajuste con ejemplos de conversación.
#
# **Lo que NO cambia:** predecir el símbolo siguiente, embeddings, capas,
# activaciones, softmax, entropía cruzada, gradiente, descenso. Todo eso está en
# las cien líneas que corriste recién.

# %% [markdown]
# ---
# ## Bloque 7 — Ejercicios

# %% [markdown]
# ### Ejercicio A7.1 [B] — Perplejidad y contexto
#
# Entrená tres modelos idénticos salvo por el tamaño del contexto (1, 3 y 5
# caracteres) y guardá en `perplejidades` un diccionario
# `{contexto: perplejidad_en_validación}`.
#
# *Pista:* `armar_ejemplos` usa la constante global `CONTEXTO`. Vas a tener que
# reasignarla antes de armar los ejemplos de cada modelo.

# %%
# TU CÓDIGO ACÁ
perplejidades = {}

# %%
check("Probaste los tres contextos", set(perplejidades) == {1, 3, 5})
if set(perplejidades) == {1, 3, 5}:
    check("Con más contexto, mejor perplejidad",
          perplejidades[3] < perplejidades[1],
          "mirar más atrás tiene que ayudar; si no ayuda, revisá que estés rearmando los ejemplos")
    print()
    for c in sorted(perplejidades):
        print(f"   contexto {c} carácter(es) -> perplejidad {perplejidades[c]:.2f}")

# %% [markdown]
# ### Ejercicio A7.2 [I] — Tu propio corpus
#
# Entrená el modelo sobre un corpus **tuyo**. Opciones:
#
# - Los mensajes de log o de alerta que emite tu nodo.
# - Los nombres de las variables y funciones de tu firmware.
# - Cualquier lista de palabras de un dominio que conozcas.
#
# Necesitás **al menos 80 palabras**. Guardalas en `mi_corpus` (una lista de
# cadenas en minúscula, sin espacios adentro de cada elemento) y entrená.
#
# Después contestá: ¿qué estructura del dominio aprendió el modelo? Miralo en las
# palabras que genera, no en la pérdida.

# %%
# TU CÓDIGO ACÁ
mi_corpus = []
mis_palabras_generadas = []

# %%
check("Tu corpus tiene al menos 80 palabras", len(mi_corpus) >= 80)
check("Son cadenas sin espacios",
      bool(mi_corpus) and all(isinstance(p, str) and " " not in p for p in mi_corpus))
check("Generaste palabras con tu modelo", len(mis_palabras_generadas) >= 8)
if mis_palabras_generadas:
    print("\n   Tu modelo inventó:", "  ".join(mis_palabras_generadas[:12]))

# %% [markdown]
# **Tu respuesta:** *(doble clic para editar)*
#
# La estructura del dominio que el modelo aprendió es…

# %% [markdown]
# ### Ejercicio A7.3 [I] — La temperatura, medida
#
# "Más creativo" es una impresión. Convertila en un número.
#
# Para cada temperatura en `[0.2, 0.5, 1.0, 2.0, 4.0]`, generá 200 palabras y
# calculá **la proporción de palabras generadas que NO están en el corpus** (o
# sea, invenciones genuinas). Guardalo en `proporcion_nuevas`.
#
# Después calculá también la proporción de palabras generadas que son
# **pronunciables**, aproximado como: no tienen tres consonantes seguidas.
# Guardalo en `proporcion_pronunciables`.

# %%
# TU CÓDIGO ACÁ
proporcion_nuevas = {}
proporcion_pronunciables = {}

# %%
_temperaturas = {0.2, 0.5, 1.0, 2.0, 4.0}
check("Mediste las cinco temperaturas",
      set(proporcion_nuevas) == _temperaturas and set(proporcion_pronunciables) == _temperaturas)
if set(proporcion_nuevas) == _temperaturas:
    check("Con más temperatura, más invenciones",
          proporcion_nuevas[4.0] > proporcion_nuevas[0.2])
    check("Con más temperatura, menos pronunciables",
          proporcion_pronunciables[4.0] < proporcion_pronunciables[0.2])
    print(f"\n   {'T':>5}  {'nuevas':>8}  {'pronunciables':>14}")
    for t in sorted(_temperaturas):
        print(f"   {t:>5}  {proporcion_nuevas[t]:>7.1%}  {proporcion_pronunciables[t]:>13.1%}")

# %% [markdown]
# **La pregunta que importa:** ¿existe una temperatura que maximice las dos cosas
# a la vez? Si no existe, ¿qué te dice eso sobre la palabra "creatividad" aplicada
# a un modelo de lenguaje?

# %% [markdown]
# **Tu respuesta:** *(doble clic para editar)*
#

# %% [markdown]
# ### Ejercicio A7.4 [A] — Documentar el modelo, no elogiarlo
#
# Sin verificación automática.
#
# Escribí la ficha técnica del modelo que entrenaste en A7.2, con estos campos.
# Es la clase de documento que en la industria se llama *model card*, y que cada
# vez más se exige por normativa.
#
# 1. **Qué hace** (una oración).
# 2. **Con qué datos se entrenó**: cuántas palabras, de dónde salieron, quién las
#    escribió.
# 3. **Métrica**: perplejidad en validación, y contra qué baseline se compara.
# 4. **Para qué NO sirve**: al menos tres usos para los que este modelo es
#    inadecuado, con el motivo.
# 5. **Sesgos previsibles**: si tu corpus tiene sobrerrepresentado algún tipo de
#    palabra, el modelo lo va a amplificar. ¿Cuál?
#
# El punto 4 es el que más cuesta y el más importante. Un sistema sin límites
# declarados es un sistema del que nadie se hace cargo.

# %% [markdown]
# **Tu ficha técnica:** *(doble clic para editar)*
#
# **1. Qué hace:**
#
# **2. Datos de entrenamiento:**
#
# **3. Métrica y baseline:**
#
# **4. Para qué NO sirve:**
#
# **5. Sesgos previsibles:**

# %% [markdown]
# ---
# ## Cierre del cuaderno A-7
#
# **Lo que quedó instalado en tu cabeza:**
#
# - Un modelo de lenguaje asigna probabilidad al símbolo siguiente. Todo lo demás
#   es repetir esa operación.
# - El modelo de bigramas cabe en una imagen: sus parámetros **son** la tabla.
# - Más contexto mejora la perplejidad; la perplejidad dice entre cuántas
#   opciones duda el modelo.
# - La temperatura no cambia el modelo, cambia cómo se lo consulta, y hay un
#   intercambio real entre invención y coherencia.
# - Una alucinación no es una falla del mecanismo: **es el mecanismo**. Probable
#   y verdadero son cosas distintas.
#
# **Checklist de entrega**
#
# - [ ] Las tres perplejidades por tamaño de contexto (A7.1).
# - [ ] Tu propio corpus entrenado, con las palabras generadas (A7.2).
# - [ ] La tabla de temperatura contra invención y pronunciabilidad (A7.3).
# - [ ] La ficha técnica completa, con los tres usos inadecuados (A7.4).
#
# **Sigue en:** `A8_LLM_local_opcional.ipynb` — opcional, requiere descargar un
# modelo real de unos 500 MB.
