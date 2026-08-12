#!/usr/bin/env python3
"""
Constructor de notebooks del Anexo Práctico de IA — ISFT N.º 238.

Toma los archivos fuente en formato "percent" (_fuentes/A*.py) y genera
los .ipynb en notebooks/. No requiere jupytext ni ninguna dependencia
externa: escribe el JSON del notebook directamente.

Formato de las fuentes:

    # %% [markdown]
    # Texto en markdown, cada linea prefijada con "# "
    # %%
    codigo_python()

Uso:
    python3 _fuentes/build_notebooks.py
"""

import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
FUENTES = RAIZ / "_fuentes"
DESTINO = RAIZ / "notebooks"

KERNEL = {
    "kernelspec": {
        "display_name": "Python 3 (anexo-ia)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def parsear(texto):
    """Divide una fuente percent-format en una lista de (tipo, lineas)."""
    celdas = []
    tipo, buffer = None, []

    def cerrar():
        if tipo is None:
            return
        # Recorta lineas vacias al inicio y al final de la celda.
        cuerpo = buffer[:]
        while cuerpo and not cuerpo[0].strip():
            cuerpo.pop(0)
        while cuerpo and not cuerpo[-1].strip():
            cuerpo.pop()
        if cuerpo:
            celdas.append((tipo, cuerpo))

    for linea in texto.splitlines():
        marca = linea.rstrip()
        if marca.startswith("# %%"):
            cerrar()
            tipo = "markdown" if "[markdown]" in marca else "code"
            buffer = []
            continue
        if tipo == "markdown":
            # En markdown se descarta el prefijo "# " de cada linea.
            if linea.startswith("# "):
                buffer.append(linea[2:])
            elif linea.strip() == "#":
                buffer.append("")
            else:
                buffer.append(linea)
        elif tipo == "code":
            buffer.append(linea)
    cerrar()
    return celdas


def a_notebook(celdas):
    salida = []
    for tipo, lineas in celdas:
        # nbformat guarda cada linea con su "\n", salvo la ultima.
        fuente = [l + "\n" for l in lineas[:-1]] + [lineas[-1]]
        celda = {"cell_type": tipo, "metadata": {}, "source": fuente}
        if tipo == "code":
            celda["execution_count"] = None
            celda["outputs"] = []
        salida.append(celda)
    return {
        "cells": salida,
        "metadata": KERNEL,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    DESTINO.mkdir(exist_ok=True)
    fuentes = sorted(f for f in FUENTES.glob("A*.py"))
    if not fuentes:
        print("No se encontraron fuentes A*.py en", FUENTES)
        return 1
    for fuente in fuentes:
        celdas = parsear(fuente.read_text(encoding="utf-8"))
        destino = DESTINO / (fuente.stem + ".ipynb")
        destino.write_text(
            json.dumps(a_notebook(celdas), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        codigo = sum(1 for t, _ in celdas if t == "code")
        print(f"{destino.name:52s} {len(celdas):3d} celdas ({codigo} de codigo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
