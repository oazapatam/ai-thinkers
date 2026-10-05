"""
Controles de Streamlit que comparten varias pestanas de la interfaz.
"""
from __future__ import annotations

import streamlit as st

import qwen_vl as q


def _usar_recomendada(clave: str) -> None:
    st.session_state[clave] = q.INSTRUCCION_BUSQUEDA


def instruccion_busqueda(clave: str) -> str:
    """Campo de la instruccion del texto que busca (la consulta), con la
    recomendada por defecto. Vacio = la generica del modelo. Debajo dice cual
    se esta usando, porque un campo vacio no lo muestra, y ofrece volver a la
    recomendada. Devuelve la instruccion que hay que enviar."""
    st.session_state.setdefault(clave, q.INSTRUCCION_BUSQUEDA)
    escrita = st.text_input(
        "Instrucción del texto (la consulta)", key=clave,
        placeholder=f"Vacía: se usa la genérica, «{q.INSTRUCCION_POR_DEFECTO}»",
        help="En búsqueda, la consulta lleva una instrucción que dice qué buscar. Con la "
             "genérica en los dos lados, «una factura de venta…» sacó coseno 0.20 contra la "
             "imagen de una factura; con esta, 0.71. Enter para aplicar un cambio.").strip()
    col_nota, col_boton = st.columns([3, 2], vertical_alignment="center")
    if escrita == q.INSTRUCCION_BUSQUEDA:
        col_nota.caption("Se usa la recomendada para buscar imágenes.")
    elif not escrita:
        col_nota.caption("Se usa la genérica del modelo.")
    else:
        col_nota.caption("Se usa la instrucción escrita.")
    if escrita != q.INSTRUCCION_BUSQUEDA:
        col_boton.button("Usar la recomendada", key=f"{clave}_recomendada",
                         on_click=_usar_recomendada, args=(clave,))
    return escrita or q.INSTRUCCION_POR_DEFECTO
