"""
Nucleo de la demo: carga de PDFs, calculo de embeddings en tres modalidades y
agrupamiento no supervisado.

Pistas (tracks) de representacion:
  tfidf       -> lexico puro (bolsa de palabras, sin modelo neuronal)
  texto       -> embedding denso de texto (transformer multilingue)
  visual      -> embedding de la IMAGEN de la pagina (CLIP). A 224 px el modelo
                 NO puede leer las letras: lo unico que sobrevive es el LAYOUT
                 (columnas, grillas, cajas, bloques, espacio en blanco).
  multimodal  -> fusion normalizada de 'texto' y 'visual'
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DIR_ENTRADA = RAIZ / "data" / "entrada"
DIR_PAGINAS = RAIZ / "data" / "paginas"
DIR_CACHE = RAIZ / "data" / "cache"

MODELO_TEXTO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_VISUAL = "openai/clip-vit-base-patch32"
ALPHA = 0.5          # peso del texto en la fusion multimodal
LADO_VISUAL = 224    # resolucion de entrada del encoder visual

PISTAS = ["tfidf", "texto", "visual", "multimodal"]
ETIQUETA_PISTA = {
    "tfidf": "Solo texto (TF-IDF lexico)",
    "texto": "Solo texto (embedding denso)",
    "visual": "Solo imagen / layout (CLIP)",
    "multimodal": "Multimodal (texto + layout)",
}


# --------------------------------------------------------------------------
def elegir_dispositivo(minimo_gb: float = 2.0) -> str:
    """Usa la GPU solo si hay memoria libre suficiente; si no, CPU.
    Los modelos son pequenos y en CPU tardan segundos para decenas de paginas."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    libre, _ = torch.cuda.mem_get_info()
    if libre / 1024**3 < minimo_gb:
        print(f"    (GPU ocupada: {libre/1024**3:.2f} GB libres -> se usa CPU)")
        return "cpu"
    return "cuda"


def _liberar():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class Documento:
    archivo: str
    ruta: Path
    texto: str
    png: Path


def _normalizar(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def cargar_documentos(carpeta: Path | str = DIR_ENTRADA, dpi: int = 110,
                      forzar: bool = False, max_paginas_texto: int = 5) -> list[Documento]:
    """Extrae el texto (hasta max_paginas_texto paginas) y rasteriza la PRIMERA
    pagina de cada PDF de la carpeta indicada.

    La primera pagina es la que lleva la senal de layout: membrete, tipo de
    encabezado, si hay tabla, formulario o prosa.
    """
    import pymupdf as fitz

    carpeta = Path(carpeta)
    destino_png = DIR_PAGINAS / carpeta.resolve().name
    destino_png.mkdir(parents=True, exist_ok=True)
    docs: list[Documento] = []
    sin_texto = []
    for pdf in sorted(carpeta.glob("*.pdf")):
        png = destino_png / (pdf.stem + ".png")
        with fitz.open(pdf) as d:
            texto = "\n".join(pag.get_text("text") for pag in d[:max_paginas_texto])
            if forzar or not png.exists() or png.stat().st_mtime < pdf.stat().st_mtime:
                d[0].get_pixmap(dpi=dpi).save(png)
        if len(texto.strip()) < 40:
            sin_texto.append(pdf.name)
        docs.append(Documento(pdf.name, pdf, texto, png))
    if not docs:
        raise SystemExit(f"No se encontraron PDFs en {carpeta}")
    if sin_texto:
        print(f"  aviso: {len(sin_texto)} PDF(s) sin capa de texto (escaneados). "
              f"Para ellos solo aporta la modalidad visual: {', '.join(sin_texto[:3])}"
              + (" ..." if len(sin_texto) > 3 else ""))
    return docs


# --------------------------------------------------------------------------
# modalidad 1: texto
# --------------------------------------------------------------------------
def _limpiar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def embeddings_tfidf(docs: list[Documento]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", sublinear_tf=True,
        min_df=2, max_df=0.85, ngram_range=(1, 2), max_features=20000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",   # ignora numeros
    )
    X = vec.fit_transform([_limpiar(d.texto) for d in docs]).toarray().astype(np.float32)
    return _normalizar(X)


def embeddings_texto(docs: list[Documento], modelo: str = MODELO_TEXTO,
                     palabras_por_trozo: int = 150) -> np.ndarray:
    """Embedding denso. Los documentos superan la ventana del modelo, asi que se
    parten en trozos y se promedian (mean pooling sobre trozos)."""
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(modelo, device=elegir_dispositivo())
    trozos, indices = [], []
    for i, d in enumerate(docs):
        palabras = _limpiar(d.texto).split()
        if not palabras:
            palabras = ["documento"]
        for j in range(0, len(palabras), palabras_por_trozo):
            trozos.append(" ".join(palabras[j:j + palabras_por_trozo]))
            indices.append(i)
    E = sbert.encode(trozos, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    indices = np.asarray(indices)
    out = np.zeros((len(docs), E.shape[1]), dtype=np.float32)
    for i in range(len(docs)):
        out[i] = E[indices == i].mean(axis=0)
    del sbert
    _liberar()
    return _normalizar(out)


# --------------------------------------------------------------------------
# modalidad 2: imagen (layout)
# --------------------------------------------------------------------------
def preparar_imagen(ruta: Path, lado: int = LADO_VISUAL):
    """Rellena la pagina a cuadrado (fondo blanco) y reescala. El relleno evita
    el recorte central del preprocesador, que destruiria el layout."""
    from PIL import Image

    im = Image.open(ruta).convert("RGB")
    w, h = im.size
    s = max(w, h)
    lienzo = Image.new("RGB", (s, s), (255, 255, 255))
    lienzo.paste(im, ((s - w) // 2, (s - h) // 2))
    return lienzo.resize((lado, lado), Image.LANCZOS)


def embeddings_visual(docs: list[Documento], modelo: str = MODELO_VISUAL,
                      lote: int = 16) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoProcessor

    disp = elegir_dispositivo()
    proc = AutoProcessor.from_pretrained(modelo)
    red = AutoModel.from_pretrained(modelo).to(disp).eval()

    salida = []
    with torch.no_grad():
        for i in range(0, len(docs), lote):
            imgs = [preparar_imagen(d.png) for d in docs[i:i + lote]]
            ent = proc(images=imgs, return_tensors="pt",
                       do_resize=False, do_center_crop=False).to(disp)
            v = red.get_image_features(**ent)
            # transformers >=5 devuelve el objeto de salida con la proyeccion
            # ya aplicada sobre pooler_output; versiones previas, un tensor.
            if not isinstance(v, torch.Tensor):
                v = v.pooler_output
            salida.append(v.float().cpu().numpy())
    del red
    _liberar()
    return _normalizar(np.vstack(salida))


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
def fusionar(E_texto: np.ndarray, E_visual: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Concatena bloques ya normalizados con pesos sqrt(alpha) / sqrt(1-alpha).
    Asi el coseno del vector fusionado es exactamente
        alpha * cos_texto + (1 - alpha) * cos_visual.
    """
    return np.hstack([np.sqrt(alpha) * _normalizar(E_texto),
                      np.sqrt(1 - alpha) * _normalizar(E_visual)]).astype(np.float32)


def calcular_todas(docs: list[Documento], alpha: float = ALPHA,
                   modelo_texto: str = MODELO_TEXTO,
                   modelo_visual: str = MODELO_VISUAL) -> dict[str, np.ndarray]:
    print("  - TF-IDF ...", flush=True)
    E = {"tfidf": embeddings_tfidf(docs)}
    print(f"  - texto denso ({modelo_texto}) ...", flush=True)
    E["texto"] = embeddings_texto(docs, modelo_texto)
    print(f"  - visual ({modelo_visual}) ...", flush=True)
    E["visual"] = embeddings_visual(docs, modelo_visual)
    print(f"  - fusion multimodal (alpha={alpha}) ...", flush=True)
    E["multimodal"] = fusionar(E["texto"], E["visual"], alpha)
    return E


# --------------------------------------------------------------------------
# agrupamiento
# --------------------------------------------------------------------------
ENLACES = {"promedio": "average", "completo": "complete", "simple": "single"}


def agrupar(X: np.ndarray, k: int | None = None, umbral: float | None = None,
            enlace: str = "promedio",
            k_min: int = 2, k_max: int = 14) -> tuple[np.ndarray, int, dict[int, float]]:
    """Agrupamiento jerarquico aglomerativo sobre distancia coseno.

    Todos los vectores estan normalizados a norma 1, asi que
        producto punto = coseno = 1 - distancia.

    Tres modos:
      umbral=u  -> corta el arbol donde la distancia vale 1-u. Con enlace
                   'completo' la garantia es exacta: TODO par de documentos
                   dentro de un mismo grupo tiene producto punto >= u.
      k=n       -> fuerza n grupos.
      k=None    -> escoge k por silueta entre k_min y k_max.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    metodo = ENLACES[enlace]

    if umbral is not None:
        et = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - umbral,
                                     metric="cosine", linkage=metodo).fit_predict(X)
        et = _renumerar(et)
        return et, len(set(et.tolist())), {}

    def ajustar(kk: int) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=kk, metric="cosine", linkage=metodo).fit_predict(X)

    siluetas: dict[int, float] = {}
    k_max = min(k_max, len(X) - 1)
    for kk in range(k_min, k_max + 1):
        siluetas[kk] = float(silhouette_score(X, ajustar(kk), metric="cosine"))
    if k is None:
        k = max(siluetas, key=siluetas.get)
    return _renumerar(ajustar(k)), k, siluetas


def _renumerar(et: np.ndarray) -> np.ndarray:
    """Reetiqueta los grupos de mayor a menor tamano."""
    from collections import Counter
    orden = [g for g, _ in Counter(et.tolist()).most_common()]
    mapa = {g: i for i, g in enumerate(orden)}
    return np.array([mapa[int(g)] for g in et])


def barrido_umbral(X: np.ndarray, umbrales, enlace: str = "completo") -> list[dict]:
    """Cuantos grupos salen para cada umbral de producto punto."""
    filas = []
    for u in umbrales:
        et, n, _ = agrupar(X, umbral=float(u), enlace=enlace)
        tam = np.bincount(et)
        filas.append({"umbral": float(u), "grupos": int(n),
                      "mayor": int(tam.max()), "sueltos": int((tam == 1).sum())})
    return filas


def estadisticas_grupos(X: np.ndarray, docs, etiquetas: np.ndarray) -> dict[int, dict]:
    """Para cada grupo: cohesion interna y separacion frente al resto.
    Para cada documento: producto punto con el centroide, con sus companeros
    de grupo, y con el documento mas parecido de fuera."""
    S = X @ X.T
    fuera_de_rango = np.eye(len(X), dtype=bool)
    out = {}
    for g in sorted(set(etiquetas.tolist())):
        idx = np.where(etiquetas == g)[0]
        ajenos = np.where(etiquetas != g)[0]
        centroide = X[idx].mean(axis=0)
        centroide = centroide / max(np.linalg.norm(centroide), 1e-12)

        docs_info = []
        for i in idx:
            companeros = idx[idx != i]
            fila = {
                "archivo": docs[i].archivo,
                "indice": int(i),
                "vector": X[i],
                "norma": float(np.linalg.norm(X[i])),
                "con_centroide": float(X[i] @ centroide),
            }
            if len(companeros):
                sims = S[i, companeros]
                j = companeros[int(np.argmax(sims))]
                fila |= {"intra_media": float(sims.mean()),
                         "intra_min": float(sims.min()),
                         "vecino": docs[j].archivo, "vecino_sim": float(sims.max())}
            else:
                fila |= {"intra_media": float("nan"), "intra_min": float("nan"),
                         "vecino": "-", "vecino_sim": float("nan")}
            if len(ajenos):
                j = ajenos[int(np.argmax(S[i, ajenos]))]
                fila |= {"externo": docs[j].archivo, "externo_sim": float(S[i, ajenos].max())}
            else:
                fila |= {"externo": "-", "externo_sim": float("nan")}
            docs_info.append(fila)

        sub = S[np.ix_(idx, idx)]
        pares = sub[~np.eye(len(idx), dtype=bool)] if len(idx) > 1 else np.array([np.nan])
        peor_externo = float(S[np.ix_(idx, ajenos)].max()) if len(ajenos) else float("nan")
        out[int(g)] = {
            "indices": idx,
            "centroide": centroide,
            "docs": docs_info,
            "n": len(idx),
            "par_min": float(np.nanmin(pares)),
            "par_medio": float(np.nanmean(pares)),
            "par_max": float(np.nanmax(pares)),
            "externo_max": peor_externo,
            "margen": float(np.nanmin(pares) - peor_externo),
        }
    _ = fuera_de_rango
    return out


def nombrar_grupos(docs: list[Documento], etiquetas: np.ndarray,
                   n_terminos: int = 3) -> dict[int, str]:
    """Nombra cada grupo con los terminos mas caracteristicos de sus documentos."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(strip_accents="unicode", min_df=1, max_df=0.6,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{4,}\b",
                          sublinear_tf=True)
    X = vec.fit_transform([_limpiar(d.texto) for d in docs])
    vocab = np.array(vec.get_feature_names_out())
    nombres = {}
    for g in sorted(set(etiquetas.tolist())):
        m = X[etiquetas == g].mean(axis=0)
        m = np.asarray(m).ravel()
        top = vocab[np.argsort(-m)[:n_terminos]]
        nombres[int(g)] = "-".join(top)
    return nombres
