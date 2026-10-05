"""
Paleta y ajustes comunes de los graficos de la interfaz, en tema claro y oscuro.

Los colores son la paleta de referencia de visualizacion. Sus posiciones 1, 2
y 3 (azul, naranja, aqua) son las unicas que pasan el validador en TODOS los
pares, en claro y en oscuro, asi que son las unicas que pueden convivir en un
scatter: A, B y C en la pestana de comparar, y color x forma para los grupos en
la de agrupar.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

PALETA = {
    "light": {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "tinta": "#0b0b0b",
              "tinta2": "#52514e", "tenue": "#898781", "rejilla": "#e1e0d9", "eje": "#c3c2b7",
              "fondo": "#ffffff"},
    "dark": {"A": "#3987e5", "B": "#d95926", "C": "#199e70", "tinta": "#ffffff",
             "tinta2": "#c3c2b7", "tenue": "#898781", "rejilla": "#3a3a37", "eje": "#4a4a46",
             "fondo": "#0e1117"},
}
TRANSPARENTE = "rgba(0,0,0,0)"      # el fondo lo pone el tema de Streamlit
CONFIG_PLOTLY = {"displaylogo": False}
DIMENSION = "dimensión"             # eje x de los graficos y primera columna de la tabla


def paleta() -> dict:
    """La del tema de esta sesion. Se llama en cada corrida del script: un
    modulo importado se evalua una sola vez por proceso."""
    return PALETA.get(getattr(st.context.theme, "type", None) or "light", PALETA["light"])


def layout_2d(fig: go.Figure, P: dict, alto: int, x: str, y: str) -> go.Figure:
    fig.update_layout(height=alto, margin=dict(l=70, r=16, t=36, b=56),
                      paper_bgcolor=TRANSPARENTE, plot_bgcolor=TRANSPARENTE,
                      font=dict(color=P["tinta2"], size=12),
                      legend=dict(orientation="h", x=0, y=1.12))
    comun = dict(gridcolor=P["rejilla"], zerolinecolor=P["eje"], linecolor=P["eje"],
                 tickfont=dict(color=P["tenue"]), automargin=True, title_standoff=12)
    fig.update_xaxes(title=x, **comun)
    fig.update_yaxes(title=y, **comun)
    return fig
