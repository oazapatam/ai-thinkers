"""
Interfaz grafica sobre el modelo desplegado en vLLM (Qwen3-VL-Embedding-2B), con
seis pestanas:

  Intro                    que es un embedding multimodal: diagrama con tres
                           entradas reales y sus vectores (pestana_intro.py).
  Comparar archivos        dos archivos (y un tercero opcional): los valores de
                           los vectores, los angulos entre ellos y una
                           proyeccion 3D.
  Agrupar una carpeta      todos los archivos de una carpeta agrupados por
                           punto de corte de coseno, con dendrograma, grafico
                           2D/3D y una subcarpeta por grupo (pestana_carpeta.py).
  Comparar texto e imagen  un texto escrito contra una imagen; con varias
                           descripciones, cual se parece mas a la imagen.
  Agrupar por texto        cada archivo de una carpeta va al texto que mas se le
                           parece, si pasa el corte (pestana_texto.py).
  Conclusiones             lo que nos llevamos del demo (pestana_conclusiones.py).

La instruccion del embedding y las dimensiones (barra lateral) valen para todas.

    cd src
    ../.venv/bin/streamlit run interfaz.py        (o desde la raiz: make desplegar)

Proyeccion 3D: los vectores se ortonormalizan en orden (Gram-Schmidt)
    e1 = A,   e2 = parte de B perpendicular a A,   e3 = parte de C perpendicular a ambos
asi cada uno queda con sus coordenadas exactas y los angulos se ven sin
distorsion (un PCA del corpus mostraria 9.9 grados donde hay 38.7). Con solo A
y B los dos quedan en el plano e1-e2: dos vectores siempre generan un plano.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import controles
import pestana_carpeta
import pestana_conclusiones
import pestana_intro
import pestana_texto
import qwen_vl as q
from estilo import CONFIG_PLOTLY, DIMENSION, TRANSPARENTE, layout_2d, paleta

st.set_page_config(page_title="Embeddings multimodales", page_icon="📐", layout="wide")

LADOS = "ABC"          # C es opcional
P = paleta()           # A, B y C son las posiciones 1, 2 y 3 de la paleta categorica


# --------------------------------------------------------------------------
# llamadas al servidor, cacheadas por contenido del archivo
# --------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def modelo(url: str) -> q.Modelo:
    return q.modelo_servido(url)


@st.cache_data(show_spinner=False, max_entries=32)
def leer(nombre: str, datos: bytes) -> q.Entrada:
    return q.leer_archivo(nombre, datos)


@st.cache_data(show_spinner=False, max_entries=96)
def embedding(nombre: str, datos: bytes, que: str, id_modelo: str,
              instruccion: str) -> q.Embedding:
    return q.embedding(leer(nombre, datos), que, id_modelo, instruccion)


@st.cache_data(show_spinner=False, max_entries=256)
def embedding_texto(texto: str, id_modelo: str, instruccion: str) -> q.Embedding:
    """Embedding de un texto escrito en la interfaz (no de un archivo)."""
    return q.embedding(q.Entrada("texto", texto, None), "texto", id_modelo, instruccion)


def angulo(cos: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def describir(que: str) -> str:
    return q.MODALIDADES[que].lower()


def pares(lados) -> list[tuple[str, str]]:
    return list(combinations(lados, 2))


# --------------------------------------------------------------------------
# proyeccion 3D
# --------------------------------------------------------------------------
def _sumar_eje(ejes: list[np.ndarray], v: np.ndarray) -> None:
    """Paso de Gram-Schmidt: agrega la parte de v perpendicular a los ejes que
    ya hay, si aporta una direccion nueva y todavia faltan ejes."""
    if len(ejes) < 3:
        u = v - sum(((v @ e) * e for e in ejes), np.zeros_like(v))
        n = np.linalg.norm(u)
        if n > 1e-6:
            ejes.append(u / n)


def base_ortonormal(vectores: list[np.ndarray]) -> np.ndarray:
    """Tres ejes ortonormales (3 x dims). Gram-Schmidt sobre los vectores en
    orden: e1 = A, e2 = parte de B perpendicular a A, e3 = parte de C
    perpendicular a los dos. Si falta un eje (no hay C, o un vector no aporta
    direccion nueva) se completa con un eje canonico: los vectores no tienen
    nada en el, asi que no cambia ninguna coordenada."""
    ejes: list[np.ndarray] = []
    for v in vectores:
        _sumar_eje(ejes, v)
    for i in np.argsort(np.abs(vectores[0]))[:16]:        # ejes canonicos que A casi no usa
        _sumar_eje(ejes, np.eye(1, len(vectores[0]), i)[0])
    return np.array(ejes)


def _arco(u: np.ndarray, v: np.ndarray, radio: float, pasos: int = 40) -> np.ndarray:
    """Arco de circulo maximo entre dos vectores unitarios, a 'radio' del origen."""
    th = float(np.arccos(np.clip(u @ v, -1.0, 1.0)))
    if th < 1e-6 or th > np.pi - 1e-6:        # iguales u opuestos: no hay un unico arco
        return radio * u[None, :]
    t = np.linspace(0, 1, pasos)[:, None]
    return radio * (np.sin((1 - t) * th) * u + np.sin(t * th) * v) / np.sin(th)


def _esfera() -> go.Scatter3d:
    """Esfera unidad en alambre: meridianos y paralelos cada 30 grados."""
    xs, ys, zs = [], [], []
    t = np.linspace(0, 2 * np.pi, 73)
    for lon in np.radians(np.arange(0, 180, 30)):
        xs += list(np.cos(t) * np.cos(lon)) + [None]
        ys += list(np.cos(t) * np.sin(lon)) + [None]
        zs += list(np.sin(t)) + [None]
    for lat in np.radians([-60, -30, 0, 30, 60]):
        xs += list(np.cos(lat) * np.cos(t)) + [None]
        ys += list(np.cos(lat) * np.sin(t)) + [None]
        zs += [np.sin(lat)] * len(t) + [None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=P["rejilla"], width=1),
                        hoverinfo="skip", showlegend=False)


def _ejes() -> list:
    """Tres lineas por el origen con sus nombres (los ejes de plotly se ocultan)."""
    ejes = np.eye(3)
    xs, ys, zs = [], [], []
    for d in ejes:
        xs += [-d[0], d[0], None]
        ys += [-d[1], d[1], None]
        zs += [-d[2], d[2], None]
    return [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=P["eje"], width=2),
                         hoverinfo="skip", showlegend=False),
            go.Scatter3d(x=ejes[:, 0] * 1.1, y=ejes[:, 1] * 1.1, z=ejes[:, 2] * 1.1 - 0.05,
                         mode="text", text=["e₁", "e₂", "e₃"], hoverinfo="skip", showlegend=False,
                         textfont=dict(color=P["tinta2"], size=14))]


def _disco() -> go.Mesh3d:
    """Plano e1-e2: el que generan A y B."""
    t = np.linspace(0, 2 * np.pi, 65)
    return go.Mesh3d(x=np.r_[0, np.cos(t)], y=np.r_[0, np.sin(t)], z=np.zeros(66),
                     i=np.zeros(64, int), j=np.arange(1, 65), k=np.arange(2, 66),
                     color=P["tinta2"], opacity=0.07, hoverinfo="skip", showscale=False)


def _texto(punto, texto: str, color: str, tamano: int = 14) -> go.Scatter3d:
    return go.Scatter3d(x=[punto[0]], y=[punto[1]], z=[punto[2]], mode="text", text=[texto],
                        textfont=dict(color=color, size=tamano), showlegend=False, hoverinfo="skip")


def _flecha(punta, color: str, nombre: str, hover: str) -> list:
    punta = np.asarray(punta, dtype=float)
    cuerpo = punta * 0.9                  # la linea termina donde empieza el cono
    return [
        go.Scatter3d(x=[0, cuerpo[0]], y=[0, cuerpo[1]], z=[0, cuerpo[2]], mode="lines",
                     line=dict(color=color, width=7), name=nombre, legendgroup=nombre,
                     hovertemplate=hover + "<extra></extra>"),
        go.Cone(x=[punta[0]], y=[punta[1]], z=[punta[2]], u=[punta[0]], v=[punta[1]], w=[punta[2]],
                anchor="tip", sizemode="absolute", sizeref=0.13, showscale=False,
                colorscale=[[0, color], [1, color]], legendgroup=nombre, hoverinfo="skip"),
        _texto([punta[0] * 1.05, punta[1] * 1.05, punta[2] + 0.12],
               f"<b>{nombre[0]}</b>", P["tinta"], 16),
    ]


def figura_3d(vec: dict[str, np.ndarray], nombres: dict[str, str]) -> go.Figure:
    lados = list(vec)
    base = base_ortonormal([vec[lado] for lado in lados])
    X3 = {lado: base @ vec[lado] for lado in lados}     # coordenadas exactas, norma 1

    trazas = [_esfera(), *_ejes(), _disco()]

    guia = dict(color=P["tenue"], width=3, dash="dash")
    if len(lados) == 2:
        # sombra de B sobre A: su longitud es cos(theta)
        pb, cos = X3["B"], float(X3["A"] @ X3["B"])
        trazas.append(go.Scatter3d(x=[pb[0], cos], y=[pb[1], 0], z=[0, 0], mode="lines",
                                   line=guia, showlegend=False, hoverinfo="skip"))
        trazas.append(_texto([cos, -0.17, 0], f"cos θ = {cos:.3f}", P["tinta2"], 12))
    else:
        # caida de C al plano de A y B: su largo es cuanto se sale C de ese plano
        pc = X3["C"]
        trazas.append(go.Scatter3d(x=[pc[0], pc[0]], y=[pc[1], pc[1]], z=[pc[2], 0], mode="lines",
                                   line=guia, showlegend=False, hoverinfo="skip"))

    # un arco por par; con tres, en radios distintos para que las etiquetas no se monten
    radios = [(0.3, 0.55)] if len(lados) == 2 else [(0.22, 0.32), (0.42, 0.52), (0.62, 0.72)]
    for (u, v), (radio, r_texto) in zip(pares(lados), radios):
        arco = _arco(X3[u], X3[v], radio)
        trazas.append(go.Scatter3d(x=arco[:, 0], y=arco[:, 1], z=arco[:, 2], mode="lines",
                                   line=dict(color=P["tinta"], width=4),
                                   showlegend=False, hoverinfo="skip"))
        grados = angulo(float(X3[u] @ X3[v]))
        texto = f"θ = {grados:.1f}°" if len(lados) == 2 else f"{u}{v} {grados:.1f}°"
        medio = arco[len(arco) // 2] / radio
        trazas.append(_texto(medio * r_texto, f"<b>{texto}</b>", P["tinta"],
                             15 if len(lados) == 2 else 13))

    for lado in lados:
        x, y, z = X3[lado]
        trazas += _flecha(X3[lado], P[lado], f"{lado} · {nombres[lado]}",
                          f"<b>{lado}</b> · {nombres[lado]}<br>({x:.3f}, {y:.3f}, {z:.3f})")

    oculto = {"visible": False, "range": [-1.2, 1.2]}
    fig = go.Figure(trazas)
    fig.update_layout(
        height=630, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor=TRANSPARENTE, font=dict(color=P["tinta"]),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(color=P["tinta2"])),
        scene=dict(aspectmode="cube", xaxis=oculto, yaxis=oculto, zaxis=oculto,
                   camera=dict(eye=dict(x=0.2, y=-0.88, z=0.88), center=dict(x=0.05, y=0.05, z=-0.08))))
    return fig


# --------------------------------------------------------------------------
# graficos de los valores
# --------------------------------------------------------------------------
def figura_componentes(vec: dict[str, np.ndarray], nombres: dict[str, str]) -> go.Figure:
    k = len(next(iter(vec.values())))
    modo = "lines+markers" if k <= 256 else "lines"
    fig = go.Figure()
    for lado, v in vec.items():
        fig.add_scatter(x=np.arange(k), y=v, mode=modo, name=f"{lado} · {nombres[lado]}",
                        line=dict(color=P[lado], width=1.5), marker=dict(size=5),
                        hovertemplate=f"{lado} = %{{y:+.5f}}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    return layout_2d(fig, P, 330, DIMENSION, "valor del componente")


def matriz_modalidades(u: str, v: str, E: dict, entradas: dict, elegidas: dict, k: int):
    """Angulo entre u y v para cada combinacion de lo que se envia de cada uno,
    con la seleccion actual resaltada."""
    mu, mv = entradas[u].modalidades, entradas[v].modalidades
    celdas = []
    for a in mu:
        fila = []
        for b in mv:
            c = float(q.truncar(E[u][a].vector, k) @ q.truncar(E[v][b].vector, k))
            fila.append(f"{angulo(c):.1f}°  ·  cos {c:.3f}")
        celdas.append(fila)
    matriz = pd.DataFrame(celdas, index=[f"{u}: {q.MODALIDADES[x]}" for x in mu],
                          columns=[f"{v}: {q.MODALIDADES[x]}" for x in mv])
    elegida = (mu.index(elegidas[u]), mv.index(elegidas[v]))
    return matriz.style.apply(
        lambda df: pd.DataFrame(
            [["font-weight: 700; background-color: rgba(128,128,128,0.18)"
              if (i, j) == elegida else "" for j in range(df.shape[1])]
             for i in range(df.shape[0])], index=df.index, columns=df.columns), axis=None)


# --------------------------------------------------------------------------
# barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Modelo desplegado")
    servidor = None
    try:
        servidor = modelo(q.URL_EMBEDDING)
        st.markdown(f"✅ **{servidor.raiz}**  \n`{q.URL_EMBEDDING}` · contexto de "
                    f"{servidor.max_tokens:,} tokens")
    except q.ErrorServidor as err:
        st.markdown(f"❌ **embedding** no responde  \n`{q.URL_EMBEDDING}`")
        st.caption(str(err))

    st.divider()
    instruccion = st.text_area(
        "Instrucción del embedding", value=q.INSTRUCCION_POR_DEFECTO, height=100,
        placeholder=f"Vacía: se usa la genérica, «{q.INSTRUCCION_POR_DEFECTO}»",
        help="Va como mensaje de sistema y cambia el vector: el mismo archivo con otra "
             "instrucción apunta en otra dirección. Se aplica igual a todos los archivos, en "
             f"las dos pestañas. Vacía = la del modelo («{q.INSTRUCCION_POR_DEFECTO}»). "
             "Ctrl+Enter para aplicarla.")
    instruccion = instruccion.strip() or q.INSTRUCCION_POR_DEFECTO
    k = st.select_slider(
        "Dimensiones del vector", options=q.DIMENSIONES, value=q.DIMENSIONES[-1],
        help="El modelo es Matryoshka: los primeros k componentes, renormalizados, "
             "son un embedding válido de k dimensiones.")

# --------------------------------------------------------------------------
# pestana "Comparar archivos"
# --------------------------------------------------------------------------
AYUDA_ANGULO = ("θ = arccos(coseno). 0° = misma dirección, 90° = sin relación, 180° = opuestos. "
                "Los vectores tienen norma 1, así que el coseno es su producto punto y "
                "‖u − v‖ = √(2 − 2·cos θ).")


def _un_archivo(lado: str):
    """Cargador de un lado; devuelve (nombre, datos, entrada, que) o None."""
    opcional = (" <span style='font-weight:400; opacity:0.7'>(opcional)</span>"
                if lado == "C" else "")
    st.markdown(_cabecera(lado, f"Archivo {lado}") + opcional, unsafe_allow_html=True)
    subido = st.file_uploader(f"Archivo {lado}", type=q.EXTENSIONES, key=f"archivo_{lado}",
                              label_visibility="collapsed")
    if subido is None:
        return None
    datos = subido.getvalue()
    try:
        entrada = leer(subido.name, datos)
    except Exception as err:          # PDF corrupto, imagen ilegible, formato raro
        st.error(f"No se pudo leer **{subido.name}**: {err}")
        return None
    if not entrada.modalidades:
        st.warning(f"**{subido.name}** no tiene texto ni imagen que enviar al modelo.")
        return None

    huella = hashlib.md5(datos, usedforsecurity=False).hexdigest()[:10]
    que = st.segmented_control(
        "Qué se envía al modelo", entrada.modalidades, default=entrada.modalidades[-1],
        required=True, format_func=q.MODALIDADES.get, key=f"que_{lado}_{huella}")

    detalles = [f"{len(entrada.texto):,} caracteres de texto" if entrada.texto
                else "sin capa de texto"]
    if entrada.imagen is not None:
        w, h = entrada.imagen.size
        detalles.append(f"{'página 1' if subido.name.lower().endswith('.pdf') else 'imagen'} "
                        f"de {w}×{h} px")
    st.caption(" · ".join(detalles))

    c_img, c_txt = st.columns([1, 2]) if entrada.imagen is not None else (None, st.container())
    if c_img is not None:
        c_img.image(entrada.imagen, width="stretch")
    if entrada.texto:
        with c_txt.expander("Texto extraído", expanded=entrada.imagen is None):
            st.code(entrada.texto[:4000] + (" …" if len(entrada.texto) > 4000 else ""),
                    language=None, wrap_lines=True, height=260)
    return subido.name, datos, entrada, que


def _metricas(vec: dict[str, np.ndarray], lados: list[str]) -> None:
    st.divider()
    if len(lados) == 2:
        cos = float(np.clip(vec["A"] @ vec["B"], -1.0, 1.0))
        m = st.columns(3)
        m[0].metric("Coseno  A·B", f"{cos:.4f}", border=True,
                    help="Los dos vectores tienen norma 1, así que el coseno es su producto punto.")
        m[1].metric("Ángulo θ", f"{angulo(cos):.2f}°", border=True, help=AYUDA_ANGULO)
        m[2].metric("Distancia  ‖A − B‖", f"{np.sqrt(max(0.0, 2 - 2 * cos)):.4f}", border=True,
                    help="Con norma 1: ‖A − B‖ = √(2 − 2·cos θ). Ordena igual que el ángulo.")
        return
    for col, (u, v) in zip(st.columns(3), pares(lados)):
        c = float(np.clip(vec[u] @ vec[v], -1.0, 1.0))
        col.metric(f"Ángulo  {u} ↔ {v}", f"{angulo(c):.2f}°", border=True, help=AYUDA_ANGULO,
                   delta=f"cos {c:.4f} · distancia {np.sqrt(max(0.0, 2 - 2 * c)):.4f}",
                   delta_color="off", delta_arrow="off")


def _explicacion_3d(vec: dict[str, np.ndarray], lados: list[str]) -> str:
    if len(lados) == 2:
        return ("Dos vectores siempre caben en un plano: e₁ apunta a A y e₂ es la parte de B "
                "perpendicular a A, así que el ángulo que ves es el ángulo real en las "
                f"{k} dimensiones, sin la distorsión de un PCA. La línea punteada es la sombra "
                "de B sobre A: mide exactamente cos θ. e₃ es la normal al plano: A y B no "
                "tienen nada en ella. Sube un tercer archivo en C para ver una vista "
                "realmente 3D.")
    altura = float(base_ortonormal([vec[x] for x in lados])[2] @ vec["C"])
    return ("e₁ apunta a A, e₂ es la parte de B perpendicular a A y e₃ la parte de C "
            "perpendicular a los dos: los tres vectores quedan con sus coordenadas exactas, "
            f"así que los tres ángulos que ves son los reales en las {k} dimensiones. La "
            f"línea punteada baja de C al plano de A y B: mide {abs(altura):.3f}, lo que C "
            "se sale de ese plano.")


def _tab_valores(vec: dict[str, np.ndarray], nombres: dict[str, str], lados: list[str],
                 clave: str = "") -> None:
    """Componentes y tabla completa. 'clave' antepone un prefijo a los widgets,
    para poder usarla en mas de una pestana."""
    normas = [f"‖{x}‖ = {np.linalg.norm(vec[x]):.4f}" for x in lados]
    productos = [f"**{u}·{v} = Σ {u}ᵢ·{v}ᵢ = {float(vec[u] @ vec[v]):.4f}**" for u, v in pares(lados)]
    st.markdown(" · ".join(normas + productos) + " — con norma 1, el producto punto es el coseno")
    st.plotly_chart(figura_componentes(vec, nombres), theme=None, config=CONFIG_PLOTLY,
                    key=f"{clave}componentes")

    tabla = pd.DataFrame({DIMENSION: np.arange(k)} | {x: vec[x] for x in lados} |
                         {f"{x}ᵢ·{y}ᵢ": vec[x] * vec[y] for x, y in pares(lados)})
    formato = st.column_config.NumberColumn(format="%+.6f")
    st.dataframe(tabla, hide_index=True, height=380,
                 column_config={c: formato for c in tabla.columns if c != DIMENSION})
    st.download_button("Descargar los vectores (CSV)", tabla.to_csv(index=False),
                       file_name=f"vectores_{k}d.csv", mime="text/csv", icon=":material/download:",
                       key=f"{clave}descarga")


def comparar() -> None:
    st.caption("Sube dos archivos (PDF, imagen o texto) y, si quieres, un tercero. Cada uno se "
               "convierte en un vector con **Qwen3-VL-Embedding-2B** y se comparan por el ángulo "
               "entre ellos.")
    seleccion = {}
    for lado, col in zip(LADOS, st.columns(len(LADOS), gap="medium")):
        with col:
            elegido = _un_archivo(lado)
        if elegido is not None:
            seleccion[lado] = elegido
    if not {"A", "B"} <= set(seleccion):
        st.info("Sube un archivo en A y otro en B para calcular sus embeddings y el ángulo entre "
                "ellos. C es opcional: con un tercer archivo la proyección 3D deja de ser plana.")
        return

    lados = list(seleccion)                               # ["A", "B"] o ["A", "B", "C"]
    nombres = {lado: s[0] for lado, s in seleccion.items()}
    entradas = {lado: s[2] for lado, s in seleccion.items()}
    elegidas = {lado: s[3] for lado, s in seleccion.items()}
    try:
        with st.spinner("Calculando embeddings…"):
            E = {lado: {m: embedding(n, d, m, servidor.id, instruccion) for m in e.modalidades}
                 for lado, (n, d, e, _) in seleccion.items()}
    except q.ErrorServidor as err:
        st.error(f"El modelo de embeddings devolvió un error: {err}")
        return
    vec = {lado: q.truncar(E[lado][elegidas[lado]].vector, k) for lado in lados}

    _metricas(vec, lados)
    nota = [f"{k} dimensiones"] + [
        f"{lado}: {describir(elegidas[lado])} ({E[lado][elegidas[lado]].tokens:,} tokens)"
        for lado in lados]
    nota.append("instrucción por defecto" if instruccion == q.INSTRUCCION_POR_DEFECTO
                else f"instrucción: «{instruccion}»")
    st.caption(" · ".join(nota))

    tab_3d, tab_valores, tab_modalidad = st.tabs(
        ["Proyección 3D", "Valores de los vectores", "Ángulo por modalidad"])
    with tab_3d:
        st.plotly_chart(figura_3d(vec, nombres), theme=None, config=CONFIG_PLOTLY)
        st.caption(_explicacion_3d(vec, lados))
    with tab_valores:
        _tab_valores(vec, nombres, lados)
    with tab_modalidad:
        for u, v in pares(lados):
            st.markdown(f"**Ángulo entre {u} y {v} según lo que se envía de cada uno**")
            st.dataframe(matriz_modalidades(u, v, E, entradas, elegidas, k))
        st.caption("Filas: lo que se envía del primero. Columnas: lo que se envía del segundo. "
                   "En negrita, la selección actual. Todas las combinaciones son comparables "
                   "porque el modelo pone texto e imágenes en el mismo espacio.")


# --------------------------------------------------------------------------
# pestana "Comparar texto e imagen"
# --------------------------------------------------------------------------
# El texto lleva la instruccion de busqueda (q.INSTRUCCION_BUSQUEDA) y la imagen
# la de la barra lateral. Con fotos tambien funciona: "un gato negro sentado"
# 0.55 contra la foto de un gato negro, "un perro" 0.22.
EXT_IMAGEN_O_PDF = sorted(e.lstrip(".") for e in q.EXT_IMAGEN | q.EXT_PDF)
MAX_TEXTOS = 20


def _cabecera(lado: str, titulo: str) -> str:
    return (f"<span style='color:{P[lado]}; font-size:1.3em'>■</span> "
            f"<b style='font-size:1.15em'>{titulo}</b>")


def _corto(texto: str, n: int = 60) -> str:
    return texto if len(texto) <= n else texto[:n - 1] + "…"


def figura_ranking(textos: list[str], cosenos: np.ndarray, elegido: int) -> go.Figure:
    """Una barra por texto con su coseno contra la imagen, el mas parecido
    arriba. El que se ve en detalle va resaltado; los demas, en gris."""
    orden = np.argsort(cosenos)                       # plotly apila de abajo hacia arriba
    fig = go.Figure(go.Bar(
        x=cosenos[orden], y=[f"{i + 1}. {_corto(textos[i])}" for i in orden], orientation="h",
        marker=dict(color=[P["A"] if i == elegido else P["tenue"] for i in orden], cornerradius=4),
        text=[f"{cosenos[i]:.3f} · {angulo(cosenos[i]):.1f}°" for i in orden],
        textposition="outside", textfont=dict(color=P["tinta2"]), cliponaxis=False,
        customdata=[textos[i] for i in orden],
        hovertemplate="%{customdata}<br>coseno %{x:.3f}<extra></extra>"))
    fig = layout_2d(fig, P, 90 + 42 * len(textos), "coseno con la imagen", "")
    fig.update_layout(bargap=0.35, showlegend=False, margin=dict(r=120))
    fig.update_xaxes(range=[min(0.0, float(cosenos.min())), max(float(cosenos.max()) * 1.2, 0.1)])
    return fig


def _entradas_texto_imagen():
    """Texto(s) a la izquierda e imagen a la derecha. Devuelve (textos,
    instruccion del texto, nombre y bytes de la imagen) o None si falta algo."""
    col_texto, col_imagen = st.columns(2, gap="large")
    with col_texto:
        st.markdown(_cabecera("A", "A · Texto"), unsafe_allow_html=True)
        escrito = st.text_area("Texto", value="un gato negro sentado", height=150, key="ti_textos",
                               label_visibility="collapsed",
                               help="Una descripción por línea para comparar varias con la misma "
                                    "imagen. Ctrl+Enter para aplicar.")
        st.caption("Puedes escribir varias descripciones, una por línea, y ver cuál se parece "
                   "más a la imagen.")
        instr_texto = controles.instruccion_busqueda("ti_instruccion")
    with col_imagen:
        st.markdown(_cabecera("B", "B · Imagen"), unsafe_allow_html=True)
        subida = st.file_uploader("Imagen", type=EXT_IMAGEN_O_PDF, key="ti_imagen",
                                  label_visibility="collapsed")
        st.caption(f"La imagen usa la instrucción de la barra lateral: «{instruccion}». "
                   "De un PDF se usa solo la imagen de la página 1.")
        if subida is not None:
            try:
                entrada = leer(subida.name, subida.getvalue())
            except Exception as err:      # imagen ilegible, PDF corrupto
                st.error(f"No se pudo leer **{subida.name}**: {err}")
                return None
            w, h = entrada.imagen.size
            st.image(entrada.imagen, width=260, caption=f"{subida.name} · {w}×{h} px")

    textos = list(dict.fromkeys(t.strip() for t in escrito.splitlines() if t.strip()))
    if not textos or subida is None:
        st.info("Escribe al menos un texto y sube una imagen para compararlos.")
        return None
    if len(textos) > MAX_TEXTOS:
        st.caption(f"Se usan las primeras {MAX_TEXTOS} descripciones.")
    return textos[:MAX_TEXTOS], instr_texto, subida


def _detalle_texto_imagen(texto: str, vt: np.ndarray, vi: np.ndarray, nombre_imagen: str,
                          tokens: tuple[int, int], instr_texto: str) -> None:
    cos = float(np.clip(vt @ vi, -1.0, 1.0))
    m = st.columns(3)
    m[0].metric("Coseno  texto·imagen", f"{cos:.4f}", border=True,
                help="Los dos vectores tienen norma 1, así que el coseno es su producto punto.")
    m[1].metric("Ángulo θ", f"{angulo(cos):.2f}°", border=True, help=AYUDA_ANGULO)
    m[2].metric("Distancia  ‖A − B‖", f"{np.sqrt(max(0.0, 2 - 2 * cos)):.4f}", border=True,
                help="Con norma 1: ‖A − B‖ = √(2 − 2·cos θ). Ordena igual que el ángulo.")
    st.caption(f"{k} dimensiones · texto: {tokens[0]:,} tokens, instrucción «{instr_texto}» · "
               f"imagen: {tokens[1]:,} tokens, instrucción «{instruccion}»")

    vec = {"A": vt, "B": vi}
    nombres = {"A": f"texto «{_corto(texto, 40)}»", "B": f"imagen {nombre_imagen}"}
    tab_3d, tab_valores = st.tabs(["Proyección 3D", "Valores de los vectores"])
    with tab_3d:
        st.plotly_chart(figura_3d(vec, nombres), theme=None, config=CONFIG_PLOTLY, key="ti_3d")
        st.caption("e₁ apunta al texto y e₂ es la parte de la imagen perpendicular a él: el ángulo "
                   f"que ves es el real en las {k} dimensiones. La línea punteada es la sombra de "
                   "la imagen sobre el texto: mide exactamente cos θ.")
    with tab_valores:
        _tab_valores(vec, nombres, ["A", "B"], clave="ti_")


def comparar_texto_imagen() -> None:
    st.caption("Escribe un texto y sube una imagen. Los dos van al mismo modelo y caen en el mismo "
               "espacio, así que se comparan por el ángulo entre sus vectores.")
    entradas = _entradas_texto_imagen()
    if entradas is None:
        return
    textos, instr_texto, subida = entradas
    try:
        with st.spinner("Calculando embeddings…"):
            imagen = embedding(subida.name, subida.getvalue(), "imagen", servidor.id, instruccion)
            de_textos = [embedding_texto(t, servidor.id, instr_texto) for t in textos]
    except q.ErrorServidor as err:
        st.error(f"El modelo de embeddings devolvió un error: {err}")
        return
    vi = q.truncar(imagen.vector, k)
    vts = [q.truncar(e.vector, k) for e in de_textos]
    cosenos = np.array([float(v @ vi) for v in vts])

    elegido = 0
    st.divider()
    if len(textos) > 1:
        st.markdown("**¿Qué descripción se parece más a la imagen?**")
        lugar = st.container()
        # otra imagen, otros textos u otra instruccion: el detalle vuelve al mas parecido
        huella = hashlib.md5("\n".join([*textos, instr_texto]).encode() + subida.getvalue(),
                             usedforsecurity=False).hexdigest()[:10]
        elegido = st.selectbox(
            "Texto que se ve en detalle", list(np.argsort(-cosenos)), key=f"ti_detalle_{huella}",
            format_func=lambda i: f"{i + 1}. {_corto(textos[i], 80)}  ·  coseno {cosenos[i]:.3f}")
        lugar.plotly_chart(figura_ranking(textos, cosenos, elegido), theme=None,
                           config=CONFIG_PLOTLY, key="ti_ranking")
        st.caption("Entre texto e imagen el valor absoluto del coseno depende mucho de la "
                   "instrucción y del largo del texto: compara las descripciones entre sí.")
    _detalle_texto_imagen(textos[elegido], vts[elegido], vi, subida.name,
                          (de_textos[elegido].tokens, imagen.tokens), instr_texto)


# --------------------------------------------------------------------------
# pagina
# --------------------------------------------------------------------------
st.title("Embeddings multimodales")
if servidor is None:
    st.error(f"El servidor de embeddings no responde en `{q.URL_EMBEDDING}`. Revisa que el "
             "contenedor `qwen3-vl-embedding` esté arriba (`docker ps`) o define la variable "
             "de entorno `URL_EMBEDDING`.")
    st.stop()

# la intro; luego lo que se hace con archivos, lo que se hace con un texto y al final
# las conclusiones
(pagina_intro, pagina_comparar, pagina_carpeta, pagina_texto_imagen, pagina_por_texto,
 pagina_conclusiones) = st.tabs(
    ["Intro", "Comparar archivos", "Agrupar una carpeta", "Comparar texto e imagen",
     "Agrupar por texto", "Conclusiones"])
with pagina_intro:
    pestana_intro.mostrar(servidor.id, instruccion, k)
with pagina_comparar:
    comparar()
with pagina_carpeta:
    pestana_carpeta.mostrar(servidor.id, instruccion, k)
with pagina_texto_imagen:
    comparar_texto_imagen()
with pagina_por_texto:
    pestana_texto.mostrar(servidor.id, instruccion, k)
with pagina_conclusiones:
    pestana_conclusiones.mostrar(k)
