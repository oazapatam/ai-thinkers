"""
Pestana "Agrupar una carpeta" de la interfaz: embeddings de todos los archivos
de una carpeta, grupos por punto de corte de coseno, dendrograma, proyeccion
2D/3D de los grupos y una subcarpeta por grupo con sus documentos.

Los grupos salen del mismo arbol aglomerativo sobre distancia coseno que usa
clasificar.py (nucleo.agrupar): el arbol se corta donde el coseno cae por
debajo del punto de corte. Con enlace completo la garantia es exacta: TODO par
de documentos dentro de un grupo tiene coseno >= el punto de corte.
"""
from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

import qwen_vl as q
from clasificar import organizar
from estilo import CONFIG_PLOTLY, TRANSPARENTE, layout_2d, paleta
from nucleo import ENLACES, Documento, agrupar, estadisticas_grupos, nombrar_grupos

PISTA = "qwen3vl"            # subcarpeta de salida, como tfidf/texto/visual/multimodal
CORTE_INICIAL = 0.60         # meseta de 7 grupos del corpus del demo con texto + imagen
FORMAS = ["circle", "diamond", "square"]
MAX_ETIQUETAS_HOJAS = 80     # con mas documentos el dendrograma no muestra los nombres
MAS_PARECIDO = "más parecido afuera"

AYUDA_ENLACE = ("Cómo se mide el coseno entre dos grupos al unirlos. **completo**: el par menos "
                "parecido; garantiza que todo par dentro de un grupo supera el corte. "
                "**promedio**: el promedio de todos los pares. **simple**: el par más parecido; "
                "encadena documentos y tiende a formar un grupo grande.")


@dataclass
class Grupos:
    """Resultado de cortar el arbol: a que grupo va cada documento y sus datos."""
    X: np.ndarray                  # vectores (n x k), norma 1
    et: np.ndarray                 # grupo de cada documento, 0 = el mas grande
    docs: list[Documento]
    stats: dict                    # nucleo.estadisticas_grupos
    nombres: dict[int, str]

    @property
    def tam(self) -> np.ndarray:
        return np.bincount(self.et)


# --------------------------------------------------------------------------
# calculos
# --------------------------------------------------------------------------
def nombrar(docs: list[Documento], et: np.ndarray) -> dict[int, str]:
    """Nombre de cada grupo con sus terminos mas caracteristicos (el mismo de
    clasificar.py); los grupos sin texto, como los de imagenes sueltas, se
    llaman 'sin-texto'."""
    try:
        nombres = nombrar_grupos(docs, et)
    except ValueError:                       # no quedo vocabulario para nombrar
        nombres = {}
    salida = {}
    for g in sorted(set(et.tolist())):
        con_texto = any(docs[i].texto.strip() for i in np.nonzero(et == g)[0])
        salida[g] = nombres.get(g) or ("sin-nombre" if con_texto else "sin-texto")
    return salida


@st.cache_data(show_spinner=False, max_entries=16)
def proyectar(X: np.ndarray, metodo: str, dims: int, extra: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Coordenadas 2D/3D de cada documento, las de 'extra' (p. ej. textos) en la
    misma proyeccion y, con PCA, la varianza que explican los ejes. PCA se ajusta
    solo con X y despues ubica 'extra'; t-SNE no sabe ubicar puntos nuevos, asi
    que los incluye en el ajuste. No depende del punto de corte."""
    extra = np.zeros((0, X.shape[1]), dtype=X.dtype) if extra is None else extra
    if metodo == "t-SNE" and len(X) >= 6:
        from sklearn.manifold import TSNE
        todo = np.vstack([X, extra])
        Y = TSNE(n_components=dims, metric="cosine", init="pca", random_state=0,
                 perplexity=min(30.0, (len(todo) - 1) / 3)).fit_transform(todo)
        return Y[:len(X)], Y[len(X):], None
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(dims, len(X))).fit(X)

    def ubicar(Z: np.ndarray) -> np.ndarray:
        if not len(Z):
            return np.zeros((0, dims))
        return np.pad(pca.transform(Z), ((0, 0), (0, dims - pca.n_components_)))
    return ubicar(X), ubicar(extra), float(pca.explained_variance_ratio_.sum())


def codificar(et: np.ndarray, orden_hojas: list[int], P: dict) -> dict[int, tuple[str, str]]:
    """Color y forma de cada grupo, asignados en el orden en que aparecen en el
    dendrograma para que dos grupos vecinos nunca compartan color. Tres colores
    x tres formas alcanzan para 9 grupos; los demas y los sueltos van en gris."""
    tam = np.bincount(et)
    vistos: list[int] = []
    for hoja in orden_hojas:
        g = int(et[hoja])
        if tam[g] > 1 and g not in vistos:
            vistos.append(g)
    colores = [P["A"], P["B"], P["C"]]
    return {g: (colores[i % 3], FORMAS[i // 3]) for i, g in enumerate(vistos[:9])}


# --------------------------------------------------------------------------
# graficos
# --------------------------------------------------------------------------
def figura_dendrograma(Z: np.ndarray, et: np.ndarray, umbral: float, archivos: list[str],
                       cod: dict, d: dict, P: dict) -> go.Figure:
    """Arbol de uniones con el eje en coseno. Las ramas que quedan por debajo
    del corte llevan el color de su grupo; las que unen grupos, gris."""
    fig = go.Figure()
    por_color: dict[str, tuple[list, list]] = {}
    for xs, ys, color in zip(d["icoord"], d["dcoord"], d["color_list"]):
        xx, yy = por_color.setdefault(color, ([], []))
        xx += list(xs) + [None]
        yy += list(ys) + [None]
    for color, (xx, yy) in por_color.items():
        fig.add_scatter(x=xx, y=yy, mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color=color, width=1.5 if color == P["tenue"] else 2.2))

    hojas = d["leaves"]
    x_hojas = 5 + 10 * np.arange(len(hojas))
    grupos_hoja = [int(et[h]) for h in hojas]
    fig.add_scatter(x=x_hojas, y=np.zeros(len(hojas)), mode="markers", showlegend=False,
                    marker=dict(size=7, color=[cod.get(g, (P["tenue"],))[0] for g in grupos_hoja]),
                    text=[archivos[h] for h in hojas], customdata=[f"G{g}" for g in grupos_hoja],
                    hovertemplate="%{text} · %{customdata}<extra></extra>")

    alto = max(float(Z[:, 2].max()) * 1.05, 0.05)     # todos iguales: el arbol no tiene altura
    for g in cod:                                      # numero de cada grupo bajo sus hojas
        xs = x_hojas[[i for i, gh in enumerate(grupos_hoja) if gh == g]]
        fig.add_annotation(x=float(xs.mean()), y=-0.05 * alto, text=f"<b>G{g}</b>",
                           showarrow=False, font=dict(color=P["tinta"], size=12))
    fig.add_hline(y=1 - umbral, line=dict(color=P["tinta"], width=1.5, dash="dash"),
                  annotation_text=f"corte: coseno {umbral:.3f}", annotation_position="top left",
                  annotation_font=dict(color=P["tinta"], size=12))

    cosenos = np.round(np.arange(1.0, 1 - alto - 0.1, -0.1), 1)
    fig = layout_2d(fig, P, 420, "", "coseno")
    fig.update_yaxes(range=[-0.1 * alto, alto], tickvals=1 - cosenos,
                     ticktext=[f"{c:.1f}" for c in cosenos], zeroline=False)
    etiquetas = len(hojas) <= MAX_ETIQUETAS_HOJAS
    fig.update_xaxes(showgrid=False, zeroline=False, tickangle=-90,
                     tickvals=x_hojas if etiquetas else [],
                     ticktext=[archivos[h] for h in hojas] if etiquetas else [],
                     tickfont=dict(size=9, color=P["tenue"]))
    return fig


@dataclass
class Marcas:
    """Puntos que no son documentos (los textos de 'Agrupar por texto'): se
    dibujan con una X del color de su grupo y su nombre al lado."""
    Y: np.ndarray
    nombres: list[str]
    colores: list[str]


def figura_grupos(Y: np.ndarray, et: np.ndarray, cod: dict, leyenda: dict[int, str],
                  hover: list[str], P: dict, centros: dict[int, str] | None = None,
                  marcas: Marcas | None = None,
                  resto: str = "sueltos y otros grupos") -> go.Figure:
    """Un punto por documento en la proyeccion, con el color y la forma de su
    grupo. 'centros' escribe una etiqueta en el centro de cada grupo; 'resto'
    nombra a los documentos que van en gris."""
    tres_d = Y.shape[1] == 3
    traza = go.Scatter3d if tres_d else go.Scatter
    tam = np.bincount(et)

    def coords(filas) -> dict:
        return dict(zip("xyz", Y[filas].T))          # x, y y, en 3D, z

    fig = go.Figure()
    for g, (color, forma) in cod.items():
        idx = np.nonzero(et == g)[0]
        fig.add_trace(traza(**coords(idx), mode="markers", name=f"{leyenda[g]} ({tam[g]})",
                            marker=dict(size=6 if tres_d else 11, color=color, symbol=forma,
                                        line=dict(color=P["fondo"], width=1 if tres_d else 1.5)),
                            text=[hover[i] for i in idx], hovertemplate="%{text}<extra></extra>"))
    grises = np.nonzero(~np.isin(et, list(cod)))[0]
    if len(grises):
        fig.add_trace(traza(**coords(grises), mode="markers", name=f"{resto} ({len(grises)})",
                            marker=dict(size=4 if tres_d else 8, color=P["tenue"], opacity=0.8),
                            text=[hover[i] for i in grises], hovertemplate="%{text}<extra></extra>"))
    if centros:
        medios = np.array([Y[et == g].mean(axis=0) for g in centros])
        fig.add_trace(traza(**dict(zip("xyz", medios.T)), mode="text",
                            text=[f"<b>{t}</b>" for t in centros.values()],
                            textposition="top center", textfont=dict(color=P["tinta"], size=13),
                            hoverinfo="skip", showlegend=False))
    if marcas is not None and len(marcas.Y):
        fig.add_trace(traza(**dict(zip("xyz", marcas.Y.T)), mode="markers+text", name="textos",
                            marker=dict(size=8 if tres_d else 16, symbol="x", color=marcas.colores,
                                        line=dict(color=P["tinta"], width=1)),
                            text=[f"<b>{n}</b>" for n in marcas.nombres], textposition="top center",
                            textfont=dict(color=P["tinta"], size=13),
                            hovertemplate="texto %{text}<extra></extra>"))
    return _ejes_grupos(fig, tres_d, P)


def _ejes_grupos(fig: go.Figure, tres_d: bool, P: dict) -> go.Figure:
    if tres_d:
        eje = {"showbackground": False, "gridcolor": P["rejilla"], "zerolinecolor": P["eje"],
               "tickfont": {"color": P["tenue"], "size": 9}, "title": {"text": ""}}
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor=TRANSPARENTE,
                          font=dict(color=P["tinta2"]),
                          legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
                          scene=dict(xaxis=eje, yaxis=eje, zaxis=eje, aspectmode="data"))
        return fig
    fig = layout_2d(fig, P, 620, "eje 1", "eje 2")
    fig.update_layout(legend=dict(y=1.02, yanchor="bottom"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# --------------------------------------------------------------------------
# pasos de la pestana
# --------------------------------------------------------------------------
def elegir_carpeta(clave: str = "", inicial: Path = q.DIR_ENTRADA
                    ) -> tuple[Path, str, list[Path]] | None:
    """Carpeta y modalidad. 'clave' antepone un prefijo a los widgets, para poder
    usarla en mas de una pestana."""
    col_carpeta, col_que = st.columns([3, 2], vertical_alignment="bottom")
    ruta = col_carpeta.text_input("Carpeta con los documentos", value=str(inicial),
                                  key=f"{clave}ruta_carpeta",
                                  help="Se toman los archivos de esa carpeta, sin entrar a "
                                       f"subcarpetas. Por defecto, {os.path.relpath(inicial, q.RAIZ)}.")
    que = col_que.segmented_control(
        "Qué se envía al modelo", list(q.MODALIDADES), default=q.TEXTO_E_IMAGEN, required=True,
        format_func=q.MODALIDADES.get, key=f"{clave}que_carpeta",
        help="Un archivo que no tiene esa modalidad va con lo que tenga: todo cae en el "
             "mismo espacio.")
    carpeta = Path(ruta).expanduser()
    if not carpeta.is_dir():
        st.warning(f"No existe la carpeta `{carpeta}`.")
        return None
    archivos = q.archivos_soportados(carpeta)
    if len(archivos) < 2:
        st.info(f"Se necesitan al menos dos archivos soportados ({', '.join(q.EXTENSIONES)}).")
        return None
    tipos = Counter("PDF" if p.suffix.lower() in q.EXT_PDF else
                    "imagen" if p.suffix.lower() in q.EXT_IMAGEN else "texto" for p in archivos)
    st.caption(f"{len(archivos)} archivos: " + " · ".join(f"{n} {t}" for t, n in tipos.items()))
    return carpeta, que, archivos


def cargar(carpeta: Path, que: str, instruccion: str, n_archivos: int,
            clave: str = "") -> dict | None:
    """Embeddings de la carpeta: de la cache si estan, o con un boton si no."""
    datos = q.embeddings_carpeta(carpeta, que, instruccion, solo_cache=True)
    if datos is not None:
        return datos
    if not st.button(f"Calcular los embeddings de {n_archivos} archivos", type="primary",
                     key=f"{clave}calcular"):
        st.caption("Se calculan una vez por carpeta, modalidad e instrucción y quedan en caché: "
                   "después, mover el punto de corte es instantáneo.")
        return None
    barra = st.progress(0.0, "Calculando embeddings…")
    datos = q.embeddings_carpeta(carpeta, que, instruccion,
                                 progreso=lambda h, t: barra.progress(h / t, f"{h} de {t} archivos"))
    barra.empty()
    return datos


def avisos(datos: dict, que: str) -> None:
    omitidos = datos["omitidos"]
    if omitidos:
        st.warning(f"Se omitieron {len(omitidos)} archivos: " + "; ".join(omitidos[:5])
                   + (" …" if len(omitidos) > 5 else ""))
    otras = Counter(u for u in datos["usadas"] if u != que)
    if otras:
        st.caption(f"{sum(otras.values())} archivos no tienen {q.MODALIDADES[que].lower()}; se "
                   "enviaron con lo que tienen: " +
                   ", ".join(f"{n} como {q.MODALIDADES[u].lower()}" for u, n in otras.items()) + ".")


def _metricas(g: Grupos, clases: list[str] | None, umbral: float, enlace: str) -> None:
    m = st.columns(4)
    m[0].metric("Grupos", len(g.tam), border=True)
    m[1].metric("Documentos", len(g.X), border=True)
    m[2].metric("Sueltos", int((g.tam == 1).sum()), border=True,
                help="Documentos que no llegan al punto de corte con ningún otro.")
    if clases:
        from sklearn.metrics import adjusted_rand_score
        m[3].metric("ARI con los tipos reales", f"{adjusted_rand_score(clases, g.et):.3f}",
                    border=True, help="Solo para el corpus del demo, que trae etiquetas. "
                                      "1 = reproduce exactamente los tipos reales.")
    else:
        m[3].metric("Grupo más grande", int(g.tam.max()), border=True)
    if enlace == "completo":
        st.caption(f"Garantía: todo par de documentos dentro de un grupo tiene coseno ≥ "
                   f"{umbral:.3f} ({g.X.shape[1]} dimensiones).")


def _dendrograma(g: Grupos, archivos: list[str], umbral: float, enlace: str, P: dict) -> dict:
    """Dibuja el arbol con el corte y devuelve el color y la forma de cada grupo."""
    Z = linkage(pdist(g.X, "cosine"), method=ENLACES[enlace])
    cod = codificar(g.et, dendrogram(Z, no_plot=True)["leaves"], P)
    grupo_nodo = {i: int(e) for i, e in enumerate(g.et)}   # None: la rama mezcla grupos
    for j, (a, b, *_) in enumerate(Z):
        ga, gb = grupo_nodo[int(a)], grupo_nodo[int(b)]
        grupo_nodo[len(g.X) + j] = ga if ga == gb else None
    d = dendrogram(Z, no_plot=True, link_color_func=lambda n: (
        cod[grupo_nodo[n]][0] if grupo_nodo[n] in cod else P["tenue"]))
    st.markdown("**Dónde corta el punto de corte**")
    st.caption("Cada unión del árbol ocurre a la altura del coseno entre los dos grupos que junta. "
               "Lo que cuelga por debajo de la línea punteada queda en un mismo grupo.")
    st.plotly_chart(figura_dendrograma(Z, g.et, umbral, archivos, cod, d, P),
                    theme=None, config=CONFIG_PLOTLY)
    return cod


def elegir_vista(clave: str = "") -> tuple[int, str]:
    """2D o 3D, PCA o t-SNE."""
    col_dim, col_met = st.columns([1, 1])
    vista = col_dim.segmented_control("Vista", ["3D", "2D"], default="3D", required=True,
                                      key=f"{clave}vista_grupos")
    metodo = col_met.segmented_control(
        "Proyección", ["PCA", "t-SNE"], default="PCA", required=True, key=f"{clave}proyeccion",
        help="PCA es lineal y conserva la forma global. t-SNE separa mejor los grupos a la vista "
             "pero deforma las distancias.")
    return int(vista[0]), metodo


def nota_proyeccion(dims: int, varianza: float | None, k: int) -> None:
    if varianza is not None:
        st.caption(f"PCA: estos {dims} ejes explican el {varianza:.0%} de la varianza. Es una "
                   f"proyección: los grupos se calcularon en las {k} dimensiones, no en el gráfico.")
    else:
        st.caption("t-SNE conserva vecindarios, no distancias: la separación entre grupos es "
                   f"orientativa. Los grupos se calcularon en las {k} dimensiones.")


def _proyeccion(g: Grupos, datos: dict, cod: dict, P: dict) -> None:
    st.markdown("**Los grupos en 2D / 3D**")
    dims, metodo = elegir_vista()
    Y, _, varianza = proyectar(g.X, metodo, dims)
    clases = datos["clases"]
    hover = [f"<b>{a}</b><br>G{e} · {g.nombres[e]}<br>coseno con el centro del grupo: "
             f"{float(g.X[i] @ g.stats[e]['centroide']):.3f}" +
             (f"<br>tipo real: {clases[i]}" if clases else "")
             for i, (a, e) in enumerate(zip(datos["archivos"], g.et))]
    st.plotly_chart(figura_grupos(Y, g.et, cod, {e: f"G{e} · {g.nombres[e]}" for e in cod}, hover,
                                  P, centros={e: f"G{e}" for e in cod}),
                    theme=None, config=CONFIG_PLOTLY)
    nota_proyeccion(dims, varianza, g.X.shape[1])
    varios = int((g.tam > 1).sum())
    if len(cod) < varios:
        st.caption(f"Hay {varios} grupos con más de un documento; los que pasan de 9 van en gris, "
                   "con su número en la tabla y al pasar el mouse.")


def _tablas(g: Grupos, clases: list[str] | None) -> None:
    resumen = pd.DataFrame([{
        "grupo": f"G{e}", "nombre": g.nombres[e], "documentos": info["n"],
        "coseno mínimo": info["par_min"], "coseno medio": info["par_medio"],
        MAS_PARECIDO: info["externo_max"], "margen": info["margen"],
    } for e, info in g.stats.items()])
    formato = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(resumen, hide_index=True, column_config=dict.fromkeys(
        ("coseno mínimo", "coseno medio", MAS_PARECIDO), formato) | {
        "margen": st.column_config.NumberColumn(
            format="%+.3f", help="coseno mínimo dentro − más parecido afuera; negativo = el "
                                 "grupo se solapa con otro")})
    with st.expander("Documento por documento"):
        filas = [{"archivo": di["archivo"], "grupo": f"G{e}", "nombre": g.nombres[e],
                  "coseno con el centro": di["con_centroide"], "vecino en el grupo": di["vecino"],
                  MAS_PARECIDO: di["externo"]} |
                 ({"tipo real": clases[di["indice"]]} if clases else {})
                 for e, info in g.stats.items() for di in info["docs"]]
        st.dataframe(pd.DataFrame(filas), hide_index=True,
                     column_config={"coseno con el centro": formato})


def _generar(g: Grupos, ruta_salida: str, parametros: dict, pista: str) -> None:
    """Crea <salida>/<pista>/ con una subcarpeta por grupo (clasificar.organizar).
    Solo reemplaza una carpeta <pista> que sea una salida anterior de esta
    herramienta: si existe y no tiene asignaciones.csv, no la toca."""
    raiz = Path(ruta_salida).expanduser()
    destino = raiz / pista
    if destino.exists() and not (destino / "asignaciones.csv").exists():
        st.error(f"`{destino}` ya existe y no es una salida de esta herramienta: no la toco. "
                 "Elige otra carpeta de salida.")
        return
    organizar(g.docs, g.et, pista, g.stats, g.nombres, raiz)
    (destino / "corte.json").write_text(json.dumps(parametros, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    st.success(f"{len(g.stats)} carpetas creadas en `{destino}`")
    st.code("\n".join(f"grupo_{e:02d}_{g.nombres[e]}/  ({info['n']} documentos)"
                      for e, info in g.stats.items()), language=None)


def carpetas(g: Grupos, parametros: dict, clave: str = "", pista: str = PISTA) -> None:
    """Carpeta de salida y boton para crear una subcarpeta por grupo."""
    st.markdown("**Crear las carpetas**")
    col_ruta, col_boton = st.columns([4, 1], vertical_alignment="bottom")
    ruta_salida = col_ruta.text_input("Carpeta de salida", value=str(q.RAIZ / "data" / "salida"),
                                      key=f"{clave}salida_carpeta")
    st.caption(f"Se crea `{Path(ruta_salida).expanduser() / pista}` con una subcarpeta por grupo "
               "(copia de sus documentos, sus vectores y la matriz de cosenos), más "
               "asignaciones.csv, resumen_grupos.csv y corte.json con los parámetros usados.")
    if col_boton.button("Generar carpetas", type="primary", width="stretch", key=f"{clave}generar"):
        _generar(g, ruta_salida, parametros, pista)


def mostrar(id_modelo: str, instruccion: str, k: int) -> None:
    P = paleta()
    st.caption("Agrupa todos los documentos de una carpeta (PDF, imagen o texto) por el coseno "
               "entre sus embeddings, dibuja los grupos y crea una subcarpeta por grupo.")
    elegida = elegir_carpeta()
    if elegida is None:
        return
    carpeta, que, archivos = elegida
    try:
        datos = cargar(carpeta, que, instruccion, len(archivos))
    except (q.ErrorServidor, FileNotFoundError) as err:
        st.error(str(err))
        return
    if datos is None:
        return
    avisos(datos, que)

    col_corte, col_enlace = st.columns([3, 2], vertical_alignment="bottom")
    umbral = col_corte.slider(
        "Punto de corte: coseno mínimo para quedar en el mismo grupo", 0.05, 0.99,
        CORTE_INICIAL, 0.005, format="%.3f", key="corte",
        help="Más alto = grupos más estrictos y más numerosos. Busca un rango donde el número "
             "de grupos no cambie: ahí la partición es estable.")
    enlace = col_enlace.segmented_control("Criterio de unión", list(ENLACES), default="completo",
                                          required=True, key="enlace", help=AYUDA_ENLACE)

    X = q.truncar(datos["matriz"], k)
    et, _, _ = agrupar(X, umbral=umbral, enlace=enlace)
    docs = [Documento(a, Path(r), t, None) for a, r, t in
            zip(datos["archivos"], datos["rutas"], datos["textos"])]
    with warnings.catch_warnings():      # un grupo de un solo documento no tiene pares
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = estadisticas_grupos(X, docs, et)
    g = Grupos(X, et, docs, stats, nombrar(docs, et))

    _metricas(g, datos["clases"], umbral, enlace)
    cod = _dendrograma(g, datos["archivos"], umbral, enlace, P)
    _proyeccion(g, datos, cod, P)
    _tablas(g, datos["clases"])
    carpetas(g, {"carpeta": str(carpeta.resolve()), "punto_de_corte": umbral, "enlace": enlace,
                  "modalidad": que, "instruccion": instruccion, "dimensiones": k,
                  "modelo": id_modelo, "grupos": len(g.tam)})
