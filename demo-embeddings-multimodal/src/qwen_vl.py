"""
Cliente del modelo de embeddings desplegado en vLLM y lectura de archivos para
la interfaz grafica (interfaz.py).

  embedding  -> Qwen/Qwen3-VL-Embedding-2B   (URL_EMBEDDING, 2048 dims, norma 1)

Todo se envia en formato 'messages', el unico que aplica la plantilla de chat
oficial del modelo. Asi texto, imagen y texto+imagen quedan en el MISMO espacio
y se pueden comparar entre si. Enviar la misma frase por 'input' da otro vector
(coseno ~0.4 con el de 'messages'), asi que nunca se mezclan los dos formatos.

El modelo sigue instrucciones: van como mensaje 'system' y cambian el vector.
La plantilla usa "Represent the user's input." cuando no se envia ninguna;
mandarla explicita da exactamente el mismo vector.

El modelo es Matryoshka: los primeros k componentes, renormalizados, son un
embedding valido de k dimensiones. Se piden las 2048 una vez y se trunca en
local (da exactamente lo mismo que el parametro 'dimensions' del servidor).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from nucleo import DIR_CACHE, DIR_ENTRADA, RAIZ

URL_EMBEDDING = os.environ.get("URL_EMBEDDING", "http://localhost:8001")
INSTRUCCION_POR_DEFECTO = "Represent the user's input."   # la de la plantilla del modelo
# En busqueda, la consulta (un texto que busca imagenes) lleva una instruccion
# que dice que buscar. Con la generica en los dos lados, "una factura de
# venta..." saco coseno 0.20 contra la imagen de una factura; con esta, 0.71.
# Tambien sirve para fotos: arboles, vacas y gatos se separan con ella.
INSTRUCCION_BUSQUEDA = "Find the document image that matches the description."

DPI_PAGINA = 110            # mismos valores que nucleo.cargar_documentos
MAX_PAGINAS_TEXTO = 5
LADO_MAX_IMAGEN = 1280      # una pagina carta a 110 dpi (935 x 1210) son ~1100 tokens
MAX_CARACTERES_EMBEDDING = 20000   # cabe en max_model_len=8192 junto con la imagen

EXT_PDF = {".pdf"}
EXT_IMAGEN = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
EXT_TEXTO = {".txt", ".md", ".csv", ".json", ".html", ".xml"}
EXTENSIONES = sorted(e.lstrip(".") for e in EXT_PDF | EXT_IMAGEN | EXT_TEXTO)

TEXTO_E_IMAGEN = "texto+imagen"
MODALIDADES = {"texto": "Texto", "imagen": "Imagen", TEXTO_E_IMAGEN: "Texto + imagen"}
DIMENSIONES = [64, 128, 256, 512, 1024, 2048]

ARCHIVO_ETIQUETAS = "etiquetas.csv"   # tipos reales (archivo, clase_real); no es un documento


class ErrorServidor(RuntimeError):
    """Fallo al hablar con vLLM: conexion rechazada o respuesta 4xx/5xx."""

    @property
    def contexto_excedido(self) -> bool:
        return "maximum context length" in str(self)


# --------------------------------------------------------------------------
# lectura de archivos
# --------------------------------------------------------------------------
@dataclass
class Entrada:
    nombre: str
    texto: str | None           # None si no hay capa de texto (imagen, PDF escaneado)
    imagen: Image.Image | None  # None si el archivo es solo texto

    @property
    def modalidades(self) -> list[str]:
        """Lo que se puede enviar al modelo, de la mas pobre a la mas rica."""
        m = []
        if self.texto:
            m.append("texto")
        if self.imagen is not None:
            m.append("imagen")
        if len(m) == 2:
            m.append(TEXTO_E_IMAGEN)
        return m


def leer_archivo(nombre: str, datos: bytes) -> Entrada:
    """PDF -> texto de las primeras paginas + imagen de la PRIMERA pagina (la que
    lleva el layout). Imagen -> solo imagen. Texto plano -> solo texto."""
    ext = Path(nombre).suffix.lower()
    texto, imagen = "", None
    if ext in EXT_PDF:
        import pymupdf as fitz
        with fitz.open(stream=datos, filetype="pdf") as d:
            texto = "\n".join(pag.get_text("text") for pag in d[:MAX_PAGINAS_TEXTO])
            pix = d[0].get_pixmap(dpi=DPI_PAGINA)
            imagen = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    elif ext in EXT_IMAGEN:
        imagen = Image.open(io.BytesIO(datos)).convert("RGB")
    elif ext in EXT_TEXTO:
        texto = datos.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Formato no soportado: '{ext or nombre}'. "
                         f"Se aceptan: {', '.join(EXTENSIONES)}")

    if imagen is not None:
        imagen.thumbnail((LADO_MAX_IMAGEN, LADO_MAX_IMAGEN))   # solo reduce
    lineas = (re.sub(r"\s+", " ", linea).strip() for linea in texto.splitlines())
    texto = "\n".join(linea for linea in lineas if linea)
    return Entrada(nombre, texto or None, imagen)


def _url_imagen(imagen: Image.Image) -> str:
    buf = io.BytesIO()
    imagen.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _contenido(e: Entrada, que: str, max_caracteres: int) -> list[dict]:
    """Partes del mensaje 'user': imagen primero y luego texto, como en el
    formato oficial de Qwen3-VL-Embedding."""
    partes = []
    if "imagen" in que:
        partes.append({"type": "image_url", "image_url": {"url": _url_imagen(e.imagen)}})
    if "texto" in que:
        partes.append({"type": "text", "text": e.texto[:max_caracteres]})
    return partes


# --------------------------------------------------------------------------
# servidor
# --------------------------------------------------------------------------
def _pedir(url: str, cuerpo: dict | None = None, timeout: float = 180) -> dict:
    datos = None if cuerpo is None else json.dumps(cuerpo).encode()
    req = urllib.request.Request(url, datos, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as err:
        cuerpo_err = err.read().decode(errors="replace")
        try:
            mensaje = json.loads(cuerpo_err)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            mensaje = cuerpo_err[:500]
        raise ErrorServidor(f"{url} respondio {err.code}: {mensaje}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
        raise ErrorServidor(f"No hay conexion con {url}: {getattr(err, 'reason', err)}") from None


@dataclass
class Modelo:
    id: str             # nombre con el que se sirve (va en cada peticion)
    raiz: str           # repositorio de Hugging Face
    max_tokens: int


def modelo_servido(url: str) -> Modelo:
    m = _pedir(url + "/v1/models", timeout=5)["data"][0]
    return Modelo(m["id"], m.get("root") or m["id"], int(m.get("max_model_len") or 0))


@dataclass
class Embedding:
    vector: np.ndarray   # 2048 componentes, norma 1
    tokens: int          # tokens que ocupo la entrada en el modelo
    caracteres: int      # caracteres de texto enviados (0 si no hubo texto)


def embedding(e: Entrada, que: str, modelo: str | None = None,
              instruccion: str = INSTRUCCION_POR_DEFECTO) -> Embedding:
    """Embedding de lo que indique 'que' ('texto', 'imagen' o 'texto+imagen'),
    guiado por 'instruccion' (vacia = la del modelo). Si el texto no cabe en el
    contexto del modelo se recorta a la mitad y se reintenta."""
    modelo = modelo or modelo_servido(URL_EMBEDDING).id
    instruccion = instruccion.strip() or INSTRUCCION_POR_DEFECTO
    max_car = MAX_CARACTERES_EMBEDDING
    while True:
        cuerpo = {"model": modelo, "encoding_format": "float",
                  "messages": [{"role": "system", "content": instruccion},
                               {"role": "user", "content": _contenido(e, que, max_car)}]}
        try:
            r = _pedir(URL_EMBEDDING + "/v1/embeddings", cuerpo)
        except ErrorServidor as err:
            if err.contexto_excedido and "texto" in que and max_car > 500:
                max_car //= 2
                continue
            raise
        v = np.asarray(r["data"][0]["embedding"], dtype=np.float32)
        car = min(len(e.texto), max_car) if "texto" in que else 0
        return Embedding(v / np.linalg.norm(v), int(r["usage"]["prompt_tokens"]), car)


def truncar(v: np.ndarray, k: int) -> np.ndarray:
    """Matryoshka: primeros k componentes, renormalizados a norma 1."""
    t = v[..., :k]
    return t / np.clip(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12, None)


# --------------------------------------------------------------------------
# carpetas completas, para la pestana de agrupar
# --------------------------------------------------------------------------
MAX_TEXTO_GUARDADO = 3000     # lo que se guarda de cada texto, para nombrar grupos


def _hash(texto: str) -> str:
    return hashlib.sha1(texto.encode(), usedforsecurity=False).hexdigest()


def archivos_soportados(carpeta: Path | str) -> list[Path]:
    """Archivos de la carpeta (sin entrar a subcarpetas) que se pueden enviar al
    modelo. Un etiquetas.csv no cuenta: son los tipos reales, no un documento."""
    extensiones = EXT_PDF | EXT_IMAGEN | EXT_TEXTO
    return sorted(p for p in Path(carpeta).iterdir()
                  if p.is_file() and p.suffix.lower() in extensiones and p.name != ARCHIVO_ETIQUETAS)


def _firma(archivos: list[Path], modelo: str, que: str, instruccion: str) -> str:
    contenido = "".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns};" for p in archivos)
    return (f"{modelo}|{que}|{instruccion}|{DPI_PAGINA}|{MAX_PAGINAS_TEXTO}|{LADO_MAX_IMAGEN}|"
            f"{MAX_CARACTERES_EMBEDDING}|{_hash(contenido)}")


def _clases_reales(archivos: list[str], carpeta: Path) -> list[str] | None:
    """Tipos reales, si hay un etiquetas.csv (archivo, clase_real) en la carpeta
    o en la de arriba que cubra todos sus archivos: data/etiquetas.csv para el
    corpus del demo, text_examples/etiquetas.csv para las fotos mezcladas."""
    for ruta in (carpeta / ARCHIVO_ETIQUETAS, carpeta.parent / ARCHIVO_ETIQUETAS):
        if ruta.exists():
            with open(ruta, newline="", encoding="utf-8") as f:
                clases = {fila["archivo"]: fila["clase_real"] for fila in csv.DictReader(f)}
            if all(a in clases for a in archivos):
                return [clases[a] for a in archivos]
    return None


_LISTAS = ("archivos", "rutas", "usadas", "textos", "clases", "omitidos")


def _leer_todos(archivos: list[Path]) -> tuple[list[tuple[Path, Entrada]], list[str]]:
    """Lee cada archivo; los que fallan o no traen nada se devuelven aparte."""
    entradas, omitidos = [], []
    for p in archivos:
        try:
            e = leer_archivo(p.name, p.read_bytes())
        except Exception as err:      # PDF corrupto, imagen ilegible
            omitidos.append(f"{p.name}: {err}")
            continue
        if e.modalidades:
            entradas.append((p, e))
        else:
            omitidos.append(f"{p.name}: no tiene texto ni imagen")
    return entradas, omitidos


def _desde_cache(cache: Path, firma: str) -> dict | None:
    if not cache.exists():
        return None
    z = np.load(cache)
    if str(z["firma"]) != firma:
        return None
    resultado = {c: z[c].tolist() for c in _LISTAS} | {"matriz": z["matriz"]}
    resultado["clases"] = resultado["clases"] or None
    return resultado


def _vectores(entradas: list[tuple[Path, Entrada]], usadas: list[str], modelo: str,
              instruccion: str, progreso, hilos: int) -> np.ndarray:
    vectores: list[np.ndarray | None] = [None] * len(entradas)
    with ThreadPoolExecutor(hilos) as ex:
        futuros = {ex.submit(embedding, e, u, modelo, instruccion): i
                   for i, ((_, e), u) in enumerate(zip(entradas, usadas))}
        for hechos, fut in enumerate(as_completed(futuros), start=1):
            vectores[futuros[fut]] = fut.result().vector
            if progreso:
                progreso(hechos, len(futuros))
    return np.vstack(vectores).astype(np.float32)


def embeddings_carpeta(carpeta: Path | str, que: str = TEXTO_E_IMAGEN,
                       instruccion: str = INSTRUCCION_POR_DEFECTO, progreso=None,
                       hilos: int = 8, solo_cache: bool = False) -> dict | None:
    """Embedding de cada archivo soportado de la carpeta, enviando lo que diga
    'que'. Un archivo que no tiene esa modalidad va con la mas rica que tenga
    (una imagen suelta no tiene texto): todo cae en el mismo espacio. Los que
    no se pueden leer se omiten y se informan.

    Queda en data/cache/, un archivo por carpeta, modalidad e instruccion, y se
    reusa mientras no cambien los archivos ni el modelo. Con solo_cache=True no
    calcula nada: devuelve None si no esta en cache. 'progreso(hechos, total)'
    se llama desde el hilo principal.

    vLLM en bf16 no es del todo determinista: el vector de un documento depende
    un poco de con que otras peticiones se agrupe y de lo que haya en su cache
    de prefijos (coseno ~0.9998 entre llamadas con el mismo archivo).

    Devuelve {'archivos', 'rutas', 'usadas', 'textos', 'clases', 'omitidos',
    'matriz'}; 'clases' es None fuera del corpus del demo."""
    carpeta = Path(carpeta).expanduser().resolve()
    archivos = archivos_soportados(carpeta)
    if not archivos:
        raise FileNotFoundError(f"No hay archivos soportados en {carpeta}")
    instruccion = instruccion.strip() or INSTRUCCION_POR_DEFECTO
    modelo = modelo_servido(URL_EMBEDDING)
    firma = _firma(archivos, modelo.raiz, que, instruccion)
    cache = DIR_CACHE / f"qwen3vl_{_hash(f'{carpeta}|{que}|{instruccion}')[:12]}.npz"

    guardado = _desde_cache(cache, firma)
    if guardado is not None:            # las etiquetas se leen siempre: pueden cambiar aparte
        guardado["clases"] = _clases_reales(guardado["archivos"], carpeta)
        return guardado
    if solo_cache:
        return None

    entradas, omitidos = _leer_todos(archivos)
    if not entradas:
        raise FileNotFoundError(f"Ningun archivo de {carpeta} se pudo leer")
    usadas = [que if que in e.modalidades else e.modalidades[-1] for _, e in entradas]
    nombres = [p.name for p, _ in entradas]
    resultado = {
        "archivos": nombres, "rutas": [str(p) for p, _ in entradas], "usadas": usadas,
        "textos": [(e.texto or "")[:MAX_TEXTO_GUARDADO] for _, e in entradas],
        "clases": _clases_reales(nombres, carpeta), "omitidos": omitidos,
        "matriz": _vectores(entradas, usadas, modelo.id, instruccion, progreso, hilos),
    }
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, firma=firma, matriz=resultado["matriz"],
             **{c: np.array(resultado[c] or []) for c in _LISTAS})
    return resultado
