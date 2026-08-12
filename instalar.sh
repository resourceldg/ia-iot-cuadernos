#!/usr/bin/env bash
# Instalador del Anexo Práctico de IA — ISFT N.º 238, Trayecto F, 2026
#
#     bash instalar.sh
#
# Crea un entorno virtual en entorno/, instala las dependencias, regenera los
# datos si hiciera falta y deja andando JupyterLab.

set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AQUI"

echo "=============================================="
echo " Anexo Práctico de IA — ISFT N.º 238"
echo "=============================================="
echo

# --- 1. Python ---------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: no encuentro python3."
    echo "Instalalo con:  sudo apt install python3 python3-venv"
    exit 1
fi

VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[1/4] Python $VERSION encontrado."

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "ERROR: hace falta Python 3.9 o superior."
    exit 1
fi

# --- 2. Entorno virtual ------------------------------------------------------
if [ -d entorno ]; then
    echo "[2/4] El entorno ya existe, lo reuso."
else
    echo "[2/4] Creando el entorno virtual en entorno/ ..."
    if ! python3 -m venv entorno; then
        echo
        echo "ERROR: falló la creación del entorno virtual."
        echo "En Debian/Ubuntu suele faltar el paquete venv:"
        echo "    sudo apt install python3-venv"
        exit 1
    fi
fi

PYTHON="$AQUI/entorno/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$AQUI/entorno/Scripts/python.exe"   # Git Bash en Windows
fi

# --- 3. Dependencias ---------------------------------------------------------
echo "[3/4] Instalando dependencias (puede tardar unos minutos) ..."
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt

echo "      Verificando ..."
"$PYTHON" - <<'PY'
faltan = []
for nombre in ["numpy", "pandas", "matplotlib", "sklearn", "jupyterlab"]:
    try:
        modulo = __import__(nombre)
        version = getattr(modulo, "__version__", "ok")
        print(f"        {nombre:12s} {version}")
    except ImportError:
        faltan.append(nombre)
if faltan:
    raise SystemExit("ERROR: no se instalaron: " + ", ".join(faltan))
PY

# --- 4. Datos ----------------------------------------------------------------
if [ -f datos/silobolsa_gas.csv ]; then
    echo "[4/4] Los datos ya están generados."
else
    echo "[4/4] Generando los conjuntos de datos ..."
    "$PYTHON" datos/generar_datasets.py
fi

echo
echo "=============================================="
echo " Listo."
echo "=============================================="
echo
echo " Para trabajar, cada vez:"
echo
echo "     source entorno/bin/activate"
echo "     jupyter lab notebooks/"
echo
echo " Empezá por A0_Entorno_y_herramientas.ipynb"
echo
echo " El cuaderno A-8 es opcional y necesita dos paquetes más:"
echo "     pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "     pip install transformers"
echo

read -r -p "¿Arranco JupyterLab ahora? [s/N] " respuesta
case "$respuesta" in
    [sS]*) exec "$AQUI/entorno/bin/jupyter" lab notebooks/ ;;
    *)     echo "Ok. Arrancalo cuando quieras con los comandos de arriba." ;;
esac
