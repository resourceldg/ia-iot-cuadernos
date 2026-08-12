#!/usr/bin/env python3
"""
Generador de los tres datasets del anexo — ISFT N.º 238, Trayecto F.

Simula la telemetria de los tres proyectos de la cohorte 2026:

    enchufe_consumo.csv   Enchufe inteligente (consumo electrico domiciliario)
    riego_humedad.csv     Riego automatico (humedad de suelo y temperatura)
    silobolsa_gas.csv     Monitoreo de silobolsa (CO2, temperatura, humedad)

Los tres archivos salen "sucios" a proposito: llevan inyectados los cuatro
tipos de dato invalido que define el modulo F-2 del cuadernillo (fuera de
rango fisico, faltante, repetido sospechoso y fuera de tiempo). Limpiarlos
es parte del trabajo del estudiante, no del generador.

La semilla esta fija: todos los estudiantes trabajan sobre los mismos
numeros y los resultados se pueden comparar entre equipos.

Uso:
    python3 datos/generar_datasets.py
"""

import json
import pathlib

import numpy as np
import pandas as pd

SEMILLA = 238  # numero del instituto, para que sea facil de recordar
AQUI = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(SEMILLA)

# Registro de lo que se rompe a proposito. Se guarda aparte, en docente/,
# para que el profesor pueda corregir sin volver a correr el generador.
bitacora = {}


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def marcas_de_tiempo(inicio, dias, minutos_por_muestra):
    total = int(dias * 24 * 60 / minutos_por_muestra)
    return pd.date_range(inicio, periods=total, freq=f"{minutos_por_muestra}min")


def hora_decimal(ts):
    """Hora del dia como numero (13:30 -> 13.5), en un array de numpy."""
    return np.asarray(ts.hour) + np.asarray(ts.minute) / 60.0


def inyectar_fallas(df, columna, n_rango, n_faltante, n_congelado,
                    n_tiempo, rango_absurdo, nombre_dataset):
    """Rompe el dataset de las cuatro maneras del modulo F-2.

    Devuelve el df modificado y anota en la bitacora que se rompio.
    """
    registro = {}
    n = len(df)
    # Se eligen posiciones sin repetir para que las fallas no se pisen entre si.
    libres = rng.permutation(np.arange(20, n - 20))
    cursor = 0

    def tomar(cantidad):
        nonlocal cursor
        elegidas = sorted(int(i) for i in libres[cursor:cursor + cantidad])
        cursor += cantidad
        return elegidas

    # 1) Fuera de rango fisico: valores que el sensor no podria medir nunca.
    idx = tomar(n_rango)
    df.loc[idx, columna] = rng.uniform(*rango_absurdo, size=len(idx))
    registro["fuera_de_rango"] = idx

    # 2) Faltantes: el sensor no respondio y quedo el hueco.
    idx = tomar(n_faltante)
    df.loc[idx, columna] = np.nan
    registro["faltante"] = idx

    # 3) Repetido sospechoso: el valor se congela varias muestras seguidas.
    congelados = []
    for _ in range(n_congelado):
        inicio = int(tomar(1)[0])
        largo = int(rng.integers(12, 30))
        fin = min(inicio + largo, n)
        df.loc[inicio:fin - 1, columna] = df.loc[inicio, columna]
        congelados.append({"desde": inicio, "hasta": fin - 1, "largo": fin - inicio})
    registro["repetido_sospechoso"] = congelados

    # 4) Fuera de tiempo: el reloj de la placa se desconfiguro y volvio atras.
    idx = tomar(n_tiempo)
    for i in idx:
        salto = pd.Timedelta(days=int(rng.integers(200, 900)))
        df.loc[i, "timestamp"] = df.loc[i, "timestamp"] - salto
    registro["fuera_de_tiempo"] = idx

    bitacora[nombre_dataset] = registro
    return df


# ---------------------------------------------------------------------------
# 1. Enchufe inteligente
# ---------------------------------------------------------------------------

def enchufe():
    """Heladera + uso domiciliario, muestreado cada 15 minutos por 30 dias.

    El compresor de la heladera cicla; encima se suman picos de uso a la
    manana y a la noche. En dos ventanas el compresor queda 'pegado' y no
    corta: ese es el patron anomalo que despues hay que detectar.
    """
    ts = marcas_de_tiempo("2026-08-17 00:00", dias=30, minutos_por_muestra=15)
    n = len(ts)
    hora = hora_decimal(ts)

    # Ciclo del compresor: ~15 min encendido cada 45 min, con jitter.
    fase = (np.arange(n) + rng.integers(0, 4)) % 4
    compresor = np.where(fase == 0, 85.0, 3.0)
    compresor += rng.normal(0, 2.0, n)

    # Uso domiciliario: pico de manana (7-9) y pico de noche (19-23).
    pico_manana = 180 * np.exp(-((hora - 8.0) ** 2) / 1.2)
    pico_noche = 260 * np.exp(-((hora - 21.0) ** 2) / 3.0)
    uso = (pico_manana + pico_noche) * rng.uniform(0.4, 1.3, n)
    # No todos los dias se cocina: algunos dias el pico casi no aparece.
    dia_indice = np.asarray((ts - ts[0]).days)
    factor_dia = rng.uniform(0.3, 1.2, dia_indice.max() + 1)[dia_indice]
    uso = uso * factor_dia

    potencia = compresor + uso
    potencia = np.clip(potencia, 0.5, None)

    # Falla real (no es un dato invalido, es un fenomeno): el compresor
    # queda pegado durante dos periodos y el consumo base se dispara.
    anomalia = np.zeros(n, dtype=int)
    for inicio_dia, largo_dias in [(11, 2), (24, 3)]:
        m = (dia_indice >= inicio_dia) & (dia_indice < inicio_dia + largo_dias)
        potencia[m] = 95 + uso[m] + rng.normal(0, 6, m.sum())
        anomalia[m] = 1

    df = pd.DataFrame({
        "timestamp": ts,
        "potencia_W": np.round(potencia, 1),
        "corriente_A": np.round(potencia / 220.0, 3),
        "rele_cerrado": 1,
        "compresor_pegado": anomalia,  # etiqueta de referencia, ver README
    })
    return inyectar_fallas(
        df, "potencia_W",
        n_rango=14, n_faltante=22, n_congelado=3, n_tiempo=4,
        rango_absurdo=(-40.0, -5.0),  # potencia negativa: fisicamente imposible
        nombre_dataset="enchufe_consumo",
    )


# ---------------------------------------------------------------------------
# 2. Riego automatico
# ---------------------------------------------------------------------------

def riego():
    """Humedad de suelo cada 30 minutos por 60 dias.

    La humedad cae de forma exponencial y cae mas rapido cuando hace calor.
    Cuando baja de 30 % el controlador riega y la humedad sube de golpe.
    """
    ts = marcas_de_tiempo("2026-08-17 00:00", dias=60, minutos_por_muestra=30)
    n = len(ts)
    hora = hora_decimal(ts)
    dia = np.arange(n) / 48.0

    # Temperatura: ciclo diario + calentamiento lento de agosto a octubre.
    temperatura = (
        16.0
        + 7.0 * np.sin(2 * np.pi * (hora - 9.0) / 24.0)
        + 0.09 * dia
        + rng.normal(0, 0.8, n)
    )

    humedad = np.zeros(n)
    riego_activo = np.zeros(n, dtype=int)
    humedad[0] = 62.0
    for i in range(1, n):
        # La evaporacion crece con la temperatura.
        perdida = 0.22 + 0.035 * max(temperatura[i] - 15.0, 0.0)
        h = humedad[i - 1] - perdida + rng.normal(0, 0.15)
        if h < 30.0:
            # El controlador riega y la humedad salta.
            h = rng.uniform(68.0, 78.0)
            riego_activo[i] = 1
        humedad[i] = min(h, 95.0)

    df = pd.DataFrame({
        "timestamp": ts,
        "humedad_suelo_pct": np.round(humedad, 2),
        "temperatura_C": np.round(temperatura, 2),
        "bomba_activa": riego_activo,
    })
    return inyectar_fallas(
        df, "humedad_suelo_pct",
        n_rango=10, n_faltante=18, n_congelado=4, n_tiempo=3,
        rango_absurdo=(115.0, 190.0),  # humedad > 100 %: imposible
        nombre_dataset="riego_humedad",
    )


# ---------------------------------------------------------------------------
# 3. Monitoreo de silobolsa
# ---------------------------------------------------------------------------

def silobolsa():
    """CO2, temperatura y humedad dentro de la bolsa, cada 30 min por 90 dias.

    El CO2 tiene una linea base estable, pero cuando arranca un foco de
    fermentacion sube de forma sostenida durante varios dias. Ese es el
    fenomeno que despues hay que anticipar: la etiqueta 'riesgo_24h' marca
    si dentro de las proximas 24 horas el CO2 va a superar 1000 ppm.
    """
    ts = marcas_de_tiempo("2026-08-17 00:00", dias=90, minutos_por_muestra=30)
    n = len(ts)
    hora = hora_decimal(ts)
    dia = np.arange(n) / 48.0

    temperatura = (
        18.0
        + 3.5 * np.sin(2 * np.pi * (hora - 10.0) / 24.0)
        + 0.05 * dia
        + rng.normal(0, 0.6, n)
    )
    humedad_rel = 58.0 + 6.0 * np.sin(2 * np.pi * (hora - 4.0) / 24.0) + rng.normal(0, 1.5, n)

    # Linea base de CO2 con deriva lenta.
    co2 = 430.0 + 25.0 * np.sin(2 * np.pi * dia / 30.0) + rng.normal(0, 12, n)

    def episodio(inicio_dia, dias_subida, amplitud, calor):
        """Un foco: sube con forma exponencial y despues ventila en 2 dias."""
        i0 = int(inicio_dia * 48)
        i1 = int((inicio_dia + dias_subida) * 48)
        t = np.linspace(0, 1, i1 - i0)
        co2[i0:i1] += amplitud * (t ** 2.4)
        temperatura[i0:i1] += calor * (t ** 2.0)  # la fermentacion tambien calienta
        i2 = min(i1 + 96, n)
        caida = np.linspace(1.0, 0.0, i2 - i1)
        co2[i1:i2] += amplitud * caida
        temperatura[i1:i2] += calor * caida

    # Tres focos que efectivamente se descontrolan y cruzan el umbral.
    for inicio_dia, dias_subida in [(19, 7), (48, 9), (74, 6)]:
        episodio(inicio_dia, dias_subida, amplitud=950.0, calor=4.5)

    # Y cuatro arranques falsos: el CO2 sube de forma parecida pero se queda a
    # mitad de camino y retrocede solo. Son los que hacen que el problema NO se
    # resuelva con un umbral sobre el nivel actual: cuando el sensor marca
    # 800 ppm, todavia no se sabe si esto es un foco o un susto. Sin estos
    # casos el ejercicio de clasificacion seria trivial.
    for inicio_dia, dias_subida, amplitud in [(9, 5, 380.0), (35, 6, 430.0),
                                              (60, 5, 400.0), (85, 4, 350.0)]:
        episodio(inicio_dia, dias_subida, amplitud=amplitud, calor=2.0)

    co2 = np.clip(co2, 350.0, None)

    # Precalentamiento del sensor de gas: tras cada reinicio del nodo, las
    # primeras lecturas son basura. Esto NO es una falla inyectada al azar,
    # es un comportamiento real documentado en el datasheet del MQ-135.
    reinicios = sorted(rng.choice(np.arange(50, n - 50), size=6, replace=False))
    precalentando = np.zeros(n, dtype=int)
    for r in reinicios:
        largo = int(rng.integers(6, 12))  # 3 a 6 horas
        co2[r:r + largo] = rng.uniform(1800, 3200, len(co2[r:r + largo]))
        precalentando[r:r + largo] = 1

    # Etiqueta supervisada: el CO2 supera 1000 ppm dentro de las proximas 24 h.
    umbral, ventana = 1000.0, 48
    futuro = pd.Series(co2).rolling(ventana, min_periods=1).max().shift(-ventana + 1)
    riesgo = (futuro.bfill() > umbral).astype(int).to_numpy()
    # Las muestras de precalentamiento no deben ensuciar la etiqueta.
    limpio = np.where(precalentando == 1, np.nan, co2)
    futuro_limpio = pd.Series(limpio).rolling(ventana, min_periods=1).max().shift(-ventana + 1)
    riesgo = (futuro_limpio.bfill().ffill() > umbral).astype(int).to_numpy()

    df = pd.DataFrame({
        "timestamp": ts,
        "co2_ppm": np.round(co2, 1),
        "temperatura_C": np.round(temperatura, 2),
        "humedad_rel_pct": np.round(humedad_rel, 2),
        "riesgo_24h": riesgo,
    })
    bitacora["silobolsa_precalentamiento"] = {
        "reinicios": [int(r) for r in reinicios],
        "nota": "muestras con CO2 absurdo por precalentamiento del sensor MQ",
    }
    return inyectar_fallas(
        df, "co2_ppm",
        n_rango=12, n_faltante=26, n_congelado=3, n_tiempo=5,
        rango_absurdo=(-300.0, -10.0),  # CO2 negativo: imposible
        nombre_dataset="silobolsa_gas",
    )


# ---------------------------------------------------------------------------

def main():
    tablas = {
        "enchufe_consumo.csv": enchufe(),
        "riego_humedad.csv": riego(),
        "silobolsa_gas.csv": silobolsa(),
    }
    for nombre, df in tablas.items():
        destino = AQUI / nombre
        df.to_csv(destino, index=False)
        faltantes = int(df.isna().sum().sum())
        print(f"{nombre:24s} {len(df):5d} filas  {len(df.columns)} columnas  "
              f"{faltantes} celdas vacias")

    clave = AQUI.parent / "docente" / "fallas_inyectadas.json"
    clave.parent.mkdir(exist_ok=True)
    clave.write_text(json.dumps(bitacora, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nClave de correccion escrita en: docente/{clave.name}")


if __name__ == "__main__":
    main()
