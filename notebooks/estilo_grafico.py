"""
Estilo y paleta de gráficos del anexo — ISFT N.º 238, Trayecto F.

Se importa desde los cuadernos con:

    from estilo_grafico import aplicar_estilo, SERIE, ESTADO, SECUENCIAL
    aplicar_estilo()

La paleta no es una elección estética: está validada para que las series se
distingan entre sí también para una persona con daltonismo (deuteranopia,
protanopia o tritanopia), que es alrededor del 8 % de los varones. El orden de
los colores importa y no se cambia: cada par consecutivo fue verificado.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paleta categórica: para series que son IDENTIDADES distintas (proyecto A,
# proyecto B, sensor 1, sensor 2). Se usan en orden, del 0 en adelante.
# ---------------------------------------------------------------------------
SERIE = [
    "#2a78d6",  # 0 azul
    "#eb6834",  # 1 naranja
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 amarillo
    "#e87ba4",  # 4 magenta
    "#008300",  # 5 verde
    "#4a3aa7",  # 6 violeta
    "#e34948",  # 7 rojo
]

# En un gráfico de dispersión (donde todas las series se comparan contra todas,
# no solo contra la vecina) el límite seguro son TRES colores. Con más, agrupá
# el resto en "otros" o hacé varios gráficos chicos.
SERIE_DISPERSION = SERIE[:3]

# ---------------------------------------------------------------------------
# Paleta de estado: reservada para estados del sistema. Nunca se usa como
# "color de serie 4". Siempre va acompañada de una etiqueta de texto, porque
# el color solo no puede cargar el significado.
# ---------------------------------------------------------------------------
ESTADO = {
    "bien": "#0ca30c",
    "atencion": "#fab219",
    "grave": "#ec835a",
    "critico": "#d03b3b",
}

# ---------------------------------------------------------------------------
# Rampa secuencial: para MAGNITUD continua (un mapa de calor, una intensidad).
# Un solo tono, de claro a oscuro. Nunca un arcoíris.
# ---------------------------------------------------------------------------
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# Tinta del gráfico: los ejes y la grilla tienen que ser RECESIVOS. Si la
# grilla compite con los datos, el gráfico está mal.
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
TINTA_APAGADA = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
FONDO = "#fcfcfb"


def aplicar_estilo():
    """Configura matplotlib con el estilo del anexo."""
    mpl.rcParams.update({
        "figure.figsize": (9, 4.2),
        "figure.dpi": 110,
        "figure.facecolor": FONDO,
        "axes.facecolor": FONDO,
        "axes.prop_cycle": mpl.cycler(color=SERIE),
        "axes.edgecolor": EJE,
        "axes.labelcolor": TINTA_SUAVE,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,     # la grilla va DETRÁS de los datos, siempre
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": TINTA_APAGADA,
        "ytick.color": TINTA_APAGADA,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "savefig.facecolor": FONDO,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def titular(ax, pregunta, respuesta=None):
    """Pone como título la PREGUNTA que el gráfico contesta.

    Un título que dice 'CO2 vs tiempo' describe los ejes, que ya están
    rotulados. Un título que dice '¿el CO2 viene subiendo?' le dice al lector
    qué tiene que buscar. Si además sabés la respuesta, va como subtítulo.
    """
    if respuesta:
        # El subtítulo ocupa su propio renglón: el título sube para dejarle lugar.
        ax.set_title(pregunta, pad=28)
        ax.text(0.0, 1.015, respuesta, transform=ax.transAxes,
                fontsize=9.5, color=TINTA_SUAVE, va="bottom")
    else:
        ax.set_title(pregunta)
    return ax


def guardar(fig, nombre):
    """Guarda la figura en figuras/ para pegarla en el informe F-5."""
    import pathlib
    destino = pathlib.Path("figuras")
    destino.mkdir(exist_ok=True)
    ruta = destino / f"{nombre}.png"
    fig.savefig(ruta)
    print(f"Figura guardada en {ruta}")
    return ruta
