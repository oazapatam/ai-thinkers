"""
Pestana "Intro": que es un embedding multimodal. Un diagrama del mecanismo con
tres entradas de ejemplo (un texto, una foto y la pagina de un PDF con su
texto) que pasan por el modelo desplegado y salen como un vector cada una. Los
valores y los cosenos del diagrama son los reales: se calculan con el modelo.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import qwen_vl as q
from estilo import paleta

TEXTO = "una vaca en el campo"
FOTO = q.RAIZ.parent / "text_examples" / "vacas" / "vaca_05.jpg"
FOTO_RESPALDO = q.DIR_ENTRADA / "doc_001.pdf"     # si no esta text_examples: otra pagina
PDF = q.DIR_ENTRADA / "doc_007.pdf"
CELDAS = 24                                       # primeros valores dibujados de cada vector
FILAS = (100, 215, 330)                           # altura de cada entrada en el diagrama

# Norma que entregan los modelos locales del demo para las mismas entradas, sin
# normalizar (medido con nucleo.MODELO_VISUAL y nucleo.MODELO_TEXTO en CPU;
# sentence-transformers solo normaliza con normalize_embeddings=True).
NORMAS_SIN_NORMALIZAR = [("CLIP ViT-B/32", "la foto de la vaca", 10.908),
                         ("CLIP ViT-B/32", f"«{TEXTO}»", 8.770),
                         ("MiniLM multilingüe", f"«{TEXTO}»", 4.614),
                         ("MiniLM multilingüe", "«una factura de venta con una tabla…»", 3.602)]


def _miles(n: int) -> str:
    """2048 -> '2 048': separador de miles con espacio fino, como se escribe en español."""
    return f"{n:,}".replace(",", " ")


def _miniatura(imagen: Image.Image, lado: int) -> str:
    im = imagen.convert("RGB")
    im.thumbnail((lado, lado))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def ejemplos(id_modelo: str, instruccion: str, marcas: tuple) -> list[dict]:
    """Las tres entradas con su miniatura, su vector completo y sus tokens.
    'marcas' (fechas de los archivos) solo invalida la cache si cambian."""
    foto = FOTO if FOTO.exists() else FOTO_RESPALDO
    foto_entrada = q.leer_archivo(foto.name, foto.read_bytes())
    pdf = q.leer_archivo(PDF.name, PDF.read_bytes())
    salida = []
    for nombre, entrada, que in (("texto", q.Entrada("texto", TEXTO, None), "texto"),
                                 ("foto", foto_entrada, "imagen"),
                                 ("PDF", pdf, q.TEXTO_E_IMAGEN)):
        e = q.embedding(entrada, que, id_modelo, instruccion)
        salida.append({"nombre": nombre, "vector": e.vector, "tokens": e.tokens,
                       "caracteres": e.caracteres,
                       "tamano": entrada.imagen.size if entrada.imagen is not None else None,
                       "miniatura": _miniatura(entrada.imagen, 240) if entrada.imagen else None})
    return salida


def _celdas(v: np.ndarray, y: float, vmax: float, P: dict) -> str:
    """Los primeros valores del vector como celdas: azul negativo, naranja
    positivo, mas intenso cuanto mayor el valor absoluto."""
    partes = []
    for i, x in enumerate(v[:CELDAS]):
        color = P["A"] if x < 0 else P["B"]
        opacidad = 0.12 + 0.88 * min(abs(float(x)) / vmax, 1.0)
        partes.append(f'<rect x="{660 + i * 11.5:.1f}" y="{y - 14}" width="10" height="28" rx="2" '
                      f'fill="{color}" fill-opacity="{opacidad:.2f}">'
                      f'<title>dimensión {i}: {x:+.4f}</title></rect>')
    return "".join(partes)


def _entradas(ej: list[dict]) -> str:
    """Columna izquierda: el texto, la foto y la pagina del PDF."""
    y1, y2, y3 = FILAS
    _, foto, pdf = ej
    w, h = foto["tamano"]
    return f"""
  <text x="20" y="{y1 - 42}" font-size="11" fill-opacity=".65" fill="currentColor">texto</text>
  <rect x="20" y="{y1 - 32}" width="200" height="64" rx="8" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <text x="120" y="{y1 + 5}" font-size="14" text-anchor="middle" fill="currentColor">«{html.escape(TEXTO)}»</text>

  <text x="20" y="{y2 - 48}" font-size="11" fill-opacity=".65" fill="currentColor">foto</text>
  <image href="{foto['miniatura']}" x="20" y="{y2 - 40}" width="120" height="80" preserveAspectRatio="xMidYMid slice"/>
  <rect x="20" y="{y2 - 40}" width="120" height="80" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <text x="150" y="{y2 - 6}" font-size="11" fill-opacity=".65" fill="currentColor">{w}×{h} px</text>
  <text x="150" y="{y2 + 10}" font-size="11" fill-opacity=".65" fill="currentColor">{_miles(foto['tokens'])} tokens</text>

  <text x="20" y="{y3 - 53}" font-size="11" fill-opacity=".65" fill="currentColor">PDF: página 1 + su texto</text>
  <image href="{pdf['miniatura']}" x="20" y="{y3 - 45}" width="64" height="90" preserveAspectRatio="xMidYMid slice"/>
  <rect x="20" y="{y3 - 45}" width="64" height="90" fill="none" stroke="currentColor" stroke-opacity=".35"/>
  <text x="96" y="{y3 - 6}" font-size="11" fill-opacity=".65" fill="currentColor">{_miles(pdf['caracteres'])} caracteres</text>
  <text x="96" y="{y3 + 10}" font-size="11" fill-opacity=".65" fill="currentColor">{_miles(pdf['tokens'])} tokens</text>
"""


def _modelo(k: int) -> str:
    """Caja central: las cuatro etapas por las que pasa cada entrada."""
    etapas = [("1 · Mensaje", "instrucción (system) + contenido (user)"),
              ("2 · Tokens", "palabras, y 1 por cada 32×32 px de imagen"),
              ("3 · Transformer", "2 mil millones de parámetros"),
              ("4 · Vector", f"último token → {_miles(k)} números ÷ su norma")]
    partes = ['<rect x="300" y="48" width="300" height="352" rx="12" fill="none" '
              'stroke="currentColor" stroke-width="1.5"/>',
              '<text x="450" y="74" font-size="15" font-weight="600" text-anchor="middle" '
              'fill="currentColor">Qwen3-VL-Embedding-2B</text>',
              '<text x="450" y="91" font-size="11" text-anchor="middle" fill-opacity=".65" '
              'fill="currentColor">desplegado en vLLM · el mismo para las tres entradas</text>']
    for n, (titulo, detalle) in enumerate(etapas):
        top = 108 + n * 72
        partes.append(f'<rect x="322" y="{top}" width="256" height="56" rx="8" fill="none" '
                      f'stroke="currentColor" stroke-opacity=".35"/>'
                      f'<text x="450" y="{top + 23}" font-size="13" font-weight="600" '
                      f'text-anchor="middle" fill="currentColor">{titulo}</text>'
                      f'<text x="450" y="{top + 41}" font-size="11" text-anchor="middle" '
                      f'fill-opacity=".7" fill="currentColor">{html.escape(detalle)}</text>')
        if n < len(etapas) - 1:
            partes.append(f'<line x1="450" y1="{top + 57}" x2="450" y2="{top + 70}" '
                          f'stroke="currentColor" marker-end="url(#flecha-intro)"/>')
    return "".join(partes)


def diagrama(ej: list[dict], k: int, P: dict) -> str:
    vectores = [q.truncar(e["vector"], k) for e in ej]
    vmax = max(float(np.abs(v[:CELDAS]).max()) for v in vectores)
    cos = {(i, j): float(vectores[i] @ vectores[j]) for i, j in ((0, 1), (0, 2), (1, 2))}
    flechas, salidas = [], []
    rotulos = ("texto", "imagen", "imagen + texto")
    titulos = ("vector del texto", "vector de la foto", "vector del PDF")
    for y, rotulo, titulo, v in zip(FILAS, rotulos, titulos, vectores):
        flechas.append(f'<line x1="228" y1="{y}" x2="296" y2="{y}" stroke="currentColor" '
                       f'marker-end="url(#flecha-intro)"/>'
                       f'<text x="262" y="{y - 7}" font-size="11" text-anchor="middle" '
                       f'fill-opacity=".7" fill="currentColor">{rotulo}</text>'
                       f'<line x1="602" y1="{y}" x2="652" y2="{y}" stroke="currentColor" '
                       f'marker-end="url(#flecha-intro)"/>')
        primeros = " ".join(f"{x:+.3f}" for x in v[:4]).replace("-", "−")
        salidas.append(f'<text x="660" y="{y - 22}" font-size="11" fill-opacity=".7" '
                       f'fill="currentColor">{titulo} · {_miles(k)} números · norma '
                       f'{np.linalg.norm(v):.3f}</text>'
                       + _celdas(v, y, vmax, P) +
                       f'<text x="942" y="{y + 5}" font-size="14" fill="currentColor">…</text>'
                       f'<text x="660" y="{y + 32}" font-size="11" fill="currentColor" '
                       f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
                       f'[{primeros} …]</text>')
    pie = (f'<text x="500" y="436" font-size="12.5" text-anchor="middle" fill="currentColor">'
           f'Mismo espacio, se comparan por el coseno:  texto ↔ foto '
           f'<tspan font-weight="600">{cos[0, 1]:.2f}</tspan>  ·  texto ↔ PDF '
           f'<tspan font-weight="600">{cos[0, 2]:.2f}</tspan>  ·  foto ↔ PDF '
           f'<tspan font-weight="600">{cos[1, 2]:.2f}</tspan></text>')
    claim = (f"Un texto, una foto y la página de un PDF entran al mismo modelo y cada uno sale "
             f"como un vector de {_miles(k)} números con norma 1, en el mismo espacio.")
    return f"""
<figure style="margin: 0 0 1.2rem 0;">
<svg viewBox="0 0 1000 450" role="img" aria-label="{claim}"
     style="width: 100%; max-width: 1100px; height: auto; color: inherit; font-family: inherit;">
  <defs>
    <marker id="flecha-intro" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>
  </defs>
  <text x="120" y="28" font-size="12" font-weight="600" text-anchor="middle" fill-opacity=".6"
        fill="currentColor" letter-spacing=".06em">ENTRADA</text>
  <text x="450" y="28" font-size="12" font-weight="600" text-anchor="middle" fill-opacity=".6"
        fill="currentColor" letter-spacing=".06em">MODELO</text>
  <text x="800" y="28" font-size="12" font-weight="600" text-anchor="middle" fill-opacity=".6"
        fill="currentColor" letter-spacing=".06em">SALIDA: UN VECTOR POR ENTRADA</text>
  {_entradas(ej)}
  {"".join(flechas)}
  {_modelo(k)}
  {"".join(salidas)}
  {pie}
</svg>
<figcaption style="font-size: 0.9rem; opacity: 0.75; margin-top: 0.4rem;">
  {claim} Las celdas son los primeros {CELDAS} valores reales de cada vector (azul negativo,
  naranja positivo; más intenso, mayor valor). Como los tres quedan en el mismo espacio, se
  comparan por el coseno del ángulo entre ellos: el texto queda más cerca de la foto de la vaca
  que de la factura.
</figcaption>
</figure>
"""


def figura_norma(P: dict) -> str:
    """A y 2A apuntan igual; B forma 50 grados con los dos. El producto punto se
    duplica con el largo, el coseno no cambia."""
    o, s = (30, 205), 90                               # origen y pixeles por unidad

    def punta(grados: float, largo: float) -> tuple[float, float]:
        r = np.radians(grados)
        return o[0] + s * largo * np.cos(r), o[1] - s * largo * np.sin(r)

    def flecha(grados, largo, color, id_marca, opacidad=1.0):
        x, y = punta(grados, largo * 0.96)
        return (f'<line x1="{o[0]}" y1="{o[1]}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" '
                f'stroke-width="3" stroke-opacity="{opacidad}" marker-end="url(#{id_marca})"/>')

    bx, by = punta(8, 1)
    ax, ay = punta(58, 1)
    a2x, a2y = punta(58, 2)
    arco = " ".join(f"{o[0] + 34 * np.cos(np.radians(g)):.1f},{o[1] - 34 * np.sin(np.radians(g)):.1f}"
                    for g in np.linspace(8, 58, 20))
    cos = float(np.cos(np.radians(50)))
    marcas = "".join(
        f'<marker id="{i}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}" fill-opacity="{a}"/></marker>'
        for i, c, a in (("norma-a", P["A"], 1), ("norma-a2", P["A"], 0.45), ("norma-b", P["B"], 1)))
    return f"""
<figure style="margin: 0;">
<svg viewBox="0 0 330 225" role="img" style="width: 100%; max-width: 440px; height: auto; color: inherit;"
     aria-label="A y 2A apuntan igual y forman 50 grados con B: el producto punto se duplica, el coseno no cambia.">
  <defs>{marcas}</defs>
  {flecha(58, 2, P["A"], "norma-a2", 0.45)}
  {flecha(58, 1, P["A"], "norma-a")}
  {flecha(8, 1, P["B"], "norma-b")}
  <polyline points="{arco}" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="{o[0] + 50}" y="{o[1] - 24}" font-size="13" fill="currentColor">θ = 50°</text>
  <text x="{ax + 8:.0f}" y="{ay + 5:.0f}" font-size="14" font-weight="600" fill="currentColor">A · norma 1</text>
  <text x="{a2x + 8:.0f}" y="{a2y + 5:.0f}" font-size="14" font-weight="600" fill="currentColor">2A · norma 2</text>
  <text x="{bx + 8:.0f}" y="{by + 5:.0f}" font-size="14" font-weight="600" fill="currentColor">B · norma 1</text>
  <text x="205" y="100" font-size="15" fill="currentColor">A · B = <tspan font-weight="600">{cos:.2f}</tspan></text>
  <text x="205" y="124" font-size="15" fill="currentColor">2A · B = <tspan font-weight="600">{2 * cos:.2f}</tspan></text>
  <text x="205" y="156" font-size="15" fill="currentColor">cos θ en los dos:</text>
  <text x="205" y="178" font-size="15" fill="currentColor"><tspan font-weight="600">{cos:.2f}</tspan></text>
</svg>
<figcaption style="font-size: 0.9rem; opacity: 0.75;">Mismo ángulo, el doble de largo: el producto
punto se duplica, el coseno no cambia.</figcaption>
</figure>
"""


def _en_una_linea(html_: str) -> str:
    """st.html sanitiza y quita el <svg>; st.markdown con HTML lo deja, pero una
    linea en blanco o una sangria cortarian el bloque: va todo en una linea."""
    return " ".join(linea.strip() for linea in html_.splitlines())


def _norma(ej: list[dict], k: int) -> None:
    norma_qwen = float(np.linalg.norm(q.truncar(ej[1]["vector"], k)))
    col_texto, col_figura = st.columns([3, 2], gap="large")
    with col_texto:
        st.markdown("#### Qué es la norma")
        st.markdown(
            "La **norma** es el largo del vector: ‖v‖ = √(v₁² + v₂² + … + vₙ²). Qwen3-VL entrega "
            "todos sus vectores con **norma 1**: sus puntas quedan sobre una esfera de radio 1 y "
            "lo único que cambia entre uno y otro es la dirección. Con norma 1:\n"
            "- el **producto punto es el coseno**: A·B = ‖A‖ ‖B‖ cos θ = cos θ;\n"
            "- la **distancia** sale del ángulo: ‖A − B‖ = √(2 − 2 cos θ).")
        st.markdown("#### Hay modelos que no normalizan")
        filas = [("Qwen3-VL-Embedding-2B (vLLM)", "la foto de la vaca", norma_qwen),
                 *NORMAS_SIN_NORMALIZAR]
        st.markdown("No todos entregan norma 1. Estos son los modelos de este proyecto con las "
                    "mismas entradas:\n\n| Modelo | Entrada | Norma que entrega |\n|---|---|---|\n" +
                    "\n".join(f"| {m} | {e} | {n:.3f} |" for m, e, n in filas))
        st.markdown(
            "Si la norma no es 1, el producto punto mezcla el ángulo con el largo: un vector más "
            "largo da un producto punto mayor aunque no se parezca más (la figura). Hay dos "
            "formas de evitarlo: **normalizar** cada vector, dividiéndolo por su norma, antes de "
            "guardarlo o compararlo (es lo que hace `nucleo.py` con CLIP y MiniLM), o **usar el "
            "coseno**: cos θ = A·B / (‖A‖ ‖B‖). En una base de datos vectorial eso es elegir la "
            "métrica coseno, y no producto punto, cuando los vectores no vienen normalizados.")
    with col_figura:
        st.markdown(_en_una_linea(figura_norma(paleta())), unsafe_allow_html=True)


def mostrar(id_modelo: str, instruccion: str, k: int) -> None:
    st.markdown("### Qué es un embedding multimodal")
    st.markdown("Un **embedding** es una lista de números, un vector, que resume el contenido de "
                "una entrada. Es **multimodal** cuando el mismo modelo recibe texto, imágenes o "
                "los dos, y los deja en el mismo espacio: así un texto se puede comparar con una "
                "imagen.")
    marcas = tuple(p.stat().st_mtime_ns for p in (FOTO, FOTO_RESPALDO, PDF) if p.exists())
    try:
        with st.spinner("Calculando los tres ejemplos con el modelo…"):
            ej = ejemplos(id_modelo, instruccion, marcas)
    except (q.ErrorServidor, OSError) as err:          # OSError incluye archivo no encontrado
        st.warning(f"No se pudieron calcular los ejemplos: {err}")
        return
    st.markdown(_en_una_linea(diagrama(ej, k, paleta())), unsafe_allow_html=True)
    _norma(ej, k)
    st.markdown("**Qué hay en cada pestaña**")
    st.markdown(
        "- **Comparar archivos**: el ángulo entre dos o tres archivos, con su proyección 3D "
        "exacta y los valores de los vectores.\n"
        "- **Agrupar una carpeta**: grupos por un punto de corte de coseno, sin decir cuántos "
        "hay, y una subcarpeta por grupo.\n"
        "- **Comparar texto e imagen**: qué descripción se parece más a una imagen.\n"
        "- **Agrupar por texto**: reparte los archivos de una carpeta entre los textos que "
        "escribas.\n"
        "- **Conclusiones**: lo que nos llevamos del demo.")
