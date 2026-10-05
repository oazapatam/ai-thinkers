"""
Pestana "Agrupar por texto" de la interfaz: se escriben uno o varios textos y
cada archivo de una carpeta se asocia al texto que mas se le parece, si su
coseno pasa el punto de corte; los que no llegan quedan sin asignar. Los
graficos son los de "Agrupar una carpeta" (pestana_carpeta.py): el reparto con
la linea de corte y la proyeccion 2D/3D, con cada texto ubicado entre sus
archivos. Tambien crea una subcarpeta por texto.

Con las fotos de ../text_examples/mezclados (10 arboles, 11 vacas y 11 gatos)
y la instruccion de busqueda, cada texto por separado se queda exactamente con
sus fotos si el corte esta entre 0.28 y 0.32 ("un arbol" es el mas justo); con
los tres textos a la vez, cada foto va al suyo.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import controles
import pestana_carpeta as pc
import qwen_vl as q
from estilo import CONFIG_PLOTLY, layout_2d, paleta
from nucleo import Documento, estadisticas_grupos

CLAVE = "pt_"                      # prefijo de los widgets de esta pestana
PISTA = "qwen3vl_texto"            # subcarpeta de salida
CARPETA_INICIAL = q.RAIZ.parent / "text_examples" / "mezclados"   # junto al proyecto
CORTE_INICIAL = 0.29
MAX_TEXTOS = 9                     # tres colores x tres formas
SIN_ASIGNAR = "sin asignar"
NOTA = ("**Por qué importa:** con un texto se puede buscar en una base de conocimiento por lo "
        "que muestran sus imágenes, no solo por lo que dicen sus palabras. Una descripción como "
        "«una factura con tabla de productos» o «un diagrama de arquitectura» encuentra las "
        "páginas e imágenes de la documentación que le corresponden, aunque nadie las haya "
        "etiquetado.")


@st.cache_data(show_spinner=False, max_entries=256)
def _vector_texto(texto: str, id_modelo: str, instruccion: str) -> np.ndarray:
    return q.embedding(q.Entrada("texto", texto, None), "texto", id_modelo, instruccion).vector


def _carpeta_de(texto: str) -> str:
    """Nombre de carpeta a partir del texto: sin tildes, signos ni espacios."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:40] or "texto"


def asignar(X: np.ndarray, T: np.ndarray, umbral: float) -> tuple[np.ndarray, np.ndarray]:
    """Cada archivo va al texto mas parecido si su coseno llega al corte; si no,
    al grupo len(T), 'sin asignar'. Devuelve (grupo de cada archivo, cosenos)."""
    S = X @ T.T
    return np.where(S.max(axis=1) >= umbral, S.argmax(axis=1), len(T)), S


def figura_reparto(S: np.ndarray, et: np.ndarray, archivos: list[str], textos: list[str],
                   umbral: float, cod: dict, P: dict) -> go.Figure:
    """Una barra por archivo con su coseno contra el texto mas parecido, de
    mayor a menor y del color de su texto. La linea punteada es el corte: lo que
    queda debajo va sin asignar."""
    mejor = S.max(axis=1)
    orden = np.argsort(-mejor)
    detalle = [" · ".join(f"«{t}» {S[i, j]:.3f}" for j, t in enumerate(textos)) for i in range(len(S))]
    fig = go.Figure()
    for g in [*cod, len(textos)]:
        filas = [i for i in orden if et[i] == g]
        if not filas:
            continue
        nombre = f"«{textos[g]}»" if g < len(textos) else SIN_ASIGNAR
        fig.add_bar(x=[archivos[i] for i in filas], y=mejor[filas], name=f"{nombre} ({len(filas)})",
                    marker=dict(color=cod[g][0] if g in cod else P["tenue"], cornerradius=3),
                    customdata=[detalle[i] for i in filas],
                    hovertemplate="<b>%{x}</b><br>%{customdata}<extra></extra>")
    fig.add_hline(y=umbral, line=dict(color=P["tinta"], width=1.5, dash="dash"),
                  annotation_text=f"corte: coseno {umbral:.3f}", annotation_position="top right",
                  annotation_font=dict(color=P["tinta"], size=12),
                  annotation_bgcolor=P["fondo"])          # legible sobre las barras
    fig = layout_2d(fig, P, 400, "", "coseno con el texto más parecido")
    fig.update_layout(bargap=0.2, legend=dict(y=1.02, yanchor="bottom"))
    fig.update_xaxes(categoryorder="array", categoryarray=[archivos[i] for i in orden],
                     tickangle=-90, tickfont=dict(size=9, color=P["tenue"]),
                     showticklabels=len(archivos) <= pc.MAX_ETIQUETAS_HOJAS)
    return fig


# --------------------------------------------------------------------------
# pasos de la pestana
# --------------------------------------------------------------------------
def _textos() -> tuple[list[str], str]:
    col_textos, col_instr = st.columns([1, 1], gap="large")
    escrito = col_textos.text_area(
        "Textos, uno por línea", value="una vaca", height=120, key=f"{CLAVE}textos",
        help=f"Cada línea es un grupo (hasta {MAX_TEXTOS}). Ctrl+Enter para aplicar.")
    col_textos.caption("Con un texto se separan los archivos que se le parecen del resto; con "
                       "varios, cada archivo va al texto que más se le parece.")
    with col_instr:
        instruccion = controles.instruccion_busqueda(f"{CLAVE}instruccion")
    textos = list(dict.fromkeys(t.strip() for t in escrito.splitlines() if t.strip()))
    if len(textos) > MAX_TEXTOS:
        st.caption(f"Se usan los primeros {MAX_TEXTOS} textos.")
    return textos[:MAX_TEXTOS], instruccion


def _metricas(et: np.ndarray, n_textos: int, clases: list[str] | None) -> None:
    asociados = et < n_textos
    m = st.columns(4)
    m[0].metric("Archivos", len(et), border=True)
    m[1].metric("Asociados a un texto", int(asociados.sum()), border=True)
    m[2].metric("Sin asignar", int((~asociados).sum()), border=True,
                help="Su coseno con todos los textos queda por debajo del punto de corte.")
    if clases and asociados.any():
        puros = sum(Counter(c for c, e in zip(clases, et) if e == g).most_common(1)[0][1]
                    for g in set(et[asociados].tolist()))
        m[3].metric("Pureza", f"{puros / asociados.sum():.0%}", border=True,
                    help="De los archivos asociados, cuántos son del tipo real más común de su "
                         "grupo. Solo si la carpeta trae etiquetas.csv.")
    else:
        m[3].metric("Textos", n_textos, border=True)


def _tabla(S: np.ndarray, et: np.ndarray, archivos: list[str], textos: list[str],
           clases: list[str] | None) -> None:
    filas = [{"archivo": a, "texto asignado": f"«{textos[e]}»" if e < len(textos) else "—",
              "coseno": float(S[i].max())} |
             {f"«{t}»": float(S[i, j]) for j, t in enumerate(textos)} |
             ({"tipo real": clases[i]} if clases else {})
             for i, (a, e) in enumerate(zip(archivos, et))]
    tabla = pd.DataFrame(filas).sort_values(["texto asignado", "coseno"], ascending=[True, False])
    formato = st.column_config.NumberColumn(format="%.3f")
    with st.expander("Archivo por archivo"):
        st.dataframe(tabla, hide_index=True,
                     column_config={c: formato for c in tabla.columns if c.startswith(("«", "coseno"))})


def mostrar(id_modelo: str, instruccion: str, k: int) -> None:
    _agrupar(id_modelo, instruccion, k)
    st.info(NOTA, icon=":material/lightbulb:")      # siempre al final, aunque falten datos


def _agrupar(id_modelo: str, instruccion: str, k: int) -> None:
    P = paleta()
    st.caption("Escribe uno o varios textos. Cada archivo de la carpeta se asocia al texto que "
               "más se le parece, si su coseno pasa el punto de corte; así se arman los grupos.")
    elegida = pc.elegir_carpeta(CLAVE, CARPETA_INICIAL)
    if elegida is None:
        return
    carpeta, que, archivos = elegida
    textos, instr_texto = _textos()
    if not textos:
        st.info("Escribe al menos un texto.")
        return
    try:
        datos = pc.cargar(carpeta, que, instruccion, len(archivos), CLAVE)
        if datos is None:
            return
        with st.spinner("Calculando los embeddings de los textos…"):
            T = np.array([q.truncar(_vector_texto(t, id_modelo, instr_texto), k) for t in textos])
    except (q.ErrorServidor, FileNotFoundError) as err:
        st.error(str(err))
        return
    pc.avisos(datos, que)

    X = q.truncar(datos["matriz"], k)
    umbral = st.slider(
        "Punto de corte: coseno mínimo con el texto", 0.0, 0.9, CORTE_INICIAL, 0.005,
        format="%.3f", key=f"{CLAVE}corte",
        help="Un archivo se asocia a un texto solo si su coseno con él llega a este valor. Entre "
             "texto e imagen los cosenos son bajos (0.3–0.5 para lo que coincide): mira el "
             "gráfico de reparto para ubicar el corte entre los que se parecen y los que no.")
    et, S = asignar(X, T, umbral)
    colores = [P["A"], P["B"], P["C"]]
    cod = {g: (colores[g % 3], pc.FORMAS[g // 3]) for g in range(len(textos))}
    nombres = {g: _carpeta_de(t) for g, t in enumerate(textos)} | {len(textos): "sin-asignar"}
    docs = [Documento(a, Path(r), t, None) for a, r, t in
            zip(datos["archivos"], datos["rutas"], datos["textos"])]
    with warnings.catch_warnings():      # un grupo de un solo archivo no tiene pares
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = estadisticas_grupos(X, docs, et)

    _metricas(et, len(textos), datos["clases"])
    st.markdown("**Reparto: cada archivo con su texto más parecido**")
    st.plotly_chart(figura_reparto(S, et, datos["archivos"], textos, umbral, cod, P),
                    theme=None, config=CONFIG_PLOTLY, key=f"{CLAVE}reparto")

    st.markdown("**Los grupos en 2D / 3D**")
    dims, metodo = pc.elegir_vista(CLAVE)
    Y, Yt, varianza = pc.proyectar(X, metodo, dims, T)
    clases = datos["clases"]
    hover = [f"<b>{a}</b><br>" + (f"«{textos[e]}»" if e < len(textos) else SIN_ASIGNAR) +
             f"<br>coseno con su texto: {S[i].max():.3f}" +
             (f"<br>tipo real: {clases[i]}" if clases else "")
             for i, (a, e) in enumerate(zip(datos["archivos"], et))]
    marcas = pc.Marcas(Yt, [f"«{t}»" for t in textos], [cod[g][0] for g in range(len(textos))])
    st.plotly_chart(pc.figura_grupos(Y, et, cod, {g: f"«{t}»" for g, t in enumerate(textos)},
                                     hover, P, marcas=marcas, resto=SIN_ASIGNAR),
                    theme=None, config=CONFIG_PLOTLY, key=f"{CLAVE}grupos")
    pc.nota_proyeccion(dims, varianza, k)
    st.caption("La ✕ de cada texto está proyectada en la misma base que los archivos: cae hacia "
               "sus archivos, pero más cerca del centro, porque texto e imagen no ocupan la misma "
               "región del espacio.")

    _tabla(S, et, datos["archivos"], textos, clases)
    grupos = pc.Grupos(X, et, docs, stats, nombres)
    pc.carpetas(grupos, {"carpeta": str(carpeta.resolve()), "textos": textos,
                         "punto_de_corte": umbral, "instruccion_texto": instr_texto,
                         "modalidad": que, "instruccion": instruccion, "dimensiones": k,
                         "modelo": id_modelo},
                clave=CLAVE, pista=PISTA)
