"""
Compara las cuatro pistas de representacion contra las etiquetas reales y
genera las figuras de la demo en figuras/.

  python src/evaluar.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from clasificar import obtener_embeddings
from nucleo import ETIQUETA_PISTA, PISTAS, RAIZ, agrupar, cargar_documentos

DIR_FIG = RAIZ / "figuras"
CSV_ETIQUETAS = RAIZ / "data" / "etiquetas.csv"

# --- paleta (ver dataviz/references/palette.md; validada con validate_palette.js) ---
SUP = "#fcfcfb"            # superficie
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#b9b8b2"
# 4 colores que pasan el control para TODOS los pares (scatter)
DESTACADOS = {
    "factura":               "#2a78d6",
    "factura_narrativa":     "#eb6834",
    "historia_clinica":      "#1baf7a",
    "formulario_afiliacion": "#4a3aa7",
}
MARCAS = {"factura": "o", "factura_narrativa": "^",
          "historia_clinica": "s", "formulario_afiliacion": "D"}
# Los 4 colores de arriba identifican CLASES. Las representaciones usan otro
# lenguaje visual para no colisionar: gris neutro y azul marino para la fusion.
SERIE = ["#2a78d6", "#1baf7a"]        # los dos pares trampa (color del primer miembro)
NEUTRO = "#8b909b"
ACENTO = "#1f3864"
# rampa secuencial de un solo tono (azul 100 -> 700)
AZULES = LinearSegmentedColormap.from_list(
    "azules", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
               "#256abf", "#184f95", "#0d366b"])

NOMBRE_CLASE = {
    "factura": "Factura (tabla)",
    "factura_narrativa": "Factura narrada (carta)",
    "extracto_bancario": "Extracto bancario",
    "historia_clinica": "Historia clínica",
    "formulario_afiliacion": "Form. de afiliación",
    "oficio": "Oficio",
    "articulo_cientifico": "Artículo científico",
    "certificado_laboral": "Certificado laboral",
}
ORDEN = ["factura", "factura_narrativa", "extracto_bancario", "historia_clinica",
         "formulario_afiliacion", "oficio", "articulo_cientifico", "certificado_laboral"]

TRAMPAS = [
    (("factura", "factura_narrativa"),
     "Mismo texto,\nlayout distinto"),
    (("historia_clinica", "formulario_afiliacion"),
     "Mismo layout,\ntexto distinto"),
]


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUP, "axes.facecolor": SUP, "savefig.facecolor": SUP,
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": TINTA, "axes.labelcolor": TINTA_2, "axes.edgecolor": TENUE,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": "#e7e6e1", "grid.linewidth": 0.8,
    })


# --------------------------------------------------------------------------
def leer_etiquetas(docs) -> np.ndarray:
    m = {r["archivo"]: r["clase_real"] for r in csv.DictReader(CSV_ETIQUETAS.open(encoding="utf-8"))}
    return np.array([m[d.archivo] for d in docs])


def pureza(y, et) -> float:
    return sum(np.bincount(np.unique(y[et == g], return_inverse=True)[1]).max()
               for g in np.unique(et)) / len(y)


def acierto_1nn(X, y) -> float:
    """Vecino mas cercano dejando uno fuera: mide la calidad del embedding
    sin depender del algoritmo de agrupamiento."""
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return float((y[S.argmax(axis=1)] == y).mean())


def metricas(X, y, k_real):
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)
    et_auto, k_auto, sil = agrupar(X, k=None)
    et_fijo, _, _ = agrupar(X, k=k_real)
    return {
        "k_auto": k_auto,
        "silueta": sil[k_auto],
        "ARI_auto": adjusted_rand_score(y, et_auto),
        "NMI_auto": normalized_mutual_info_score(y, et_auto),
        "ARI_k": adjusted_rand_score(y, et_fijo),
        "pureza_k": pureza(y, et_fijo),
        "1nn": acierto_1nn(X, y),
        "et_auto": et_auto,
        "et_fijo": et_fijo,
    }


# --------------------------------------------------------------------------
def fig_dataset(docs, y):
    from PIL import Image
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.2))
    for ax, clase in zip(axes.ravel(), ORDEN):
        i = int(np.where(y == clase)[0][0])
        ax.imshow(Image.open(docs[i].png))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(TENUE); s.set_linewidth(0.8)
        color = DESTACADOS.get(clase, TINTA)
        ax.set_title(NOMBRE_CLASE[clase], fontsize=10.5, color=color,
                     fontweight="bold" if clase in DESTACADOS else "normal", pad=6)
        ax.set_xlabel(f"n = {int((y == clase).sum())}", fontsize=8, color=TINTA_2)
    fig.suptitle("73 documentos, 8 tipos, una sola carpeta de entrada",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945, "En color: los dos pares diseñados como trampa "
                         "(azul/naranja comparten el texto; verde/violeta comparten el layout)",
             ha="center", fontsize=9, color=TINTA_2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(DIR_FIG / "1_dataset.png", dpi=140)
    plt.close(fig)


def fig_similitud(E, y):
    orden = np.concatenate([np.where(y == c)[0] for c in ORDEN])
    tam = [int((y == c).sum()) for c in ORDEN]
    limites = np.cumsum(tam)
    inicio = np.concatenate([[0], limites[:-1]])
    centros = inicio + np.array(tam) / 2
    pos = {c: (int(inicio[i]), int(tam[i])) for i, c in enumerate(ORDEN)}

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.6))
    for ax, pista in zip(axes, PISTAS):
        X = E[pista][orden]
        S = X @ X.T
        fuera = S[~np.eye(len(S), dtype=bool)]
        # cada modelo vive en su propio rango de coseno: escala propia por panel
        vmin, vmax = float(np.percentile(fuera, 1)), float(np.percentile(fuera, 99.5))
        im = ax.imshow(S, cmap=AZULES, vmin=vmin, vmax=vmax, interpolation="nearest")
        for b in limites[:-1]:
            ax.axhline(b - 0.5, color=SUP, lw=1.2)
            ax.axvline(b - 0.5, color=SUP, lw=1.2)
        # recuadro sobre los bloques fuera de la diagonal que deberian ser claros
        for (a, b), _ in TRAMPAS:
            (ia, na), (ib, nb) = pos[a], pos[b]
            for (x0, n0), (y0, n1) in [((ia, na), (ib, nb)), ((ib, nb), (ia, na))]:
                ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), n0, n1, fill=False,
                                           ec=TINTA, lw=1.5, ls=(0, (3, 2)), zorder=5))
        ax.set_xticks(centros)
        ax.set_xticklabels([NOMBRE_CLASE[c] for c in ORDEN], fontsize=7.5,
                           rotation=45, ha="right")
        for c, tk in zip(ORDEN, ax.get_xticklabels()):
            if c in DESTACADOS:
                tk.set_color(DESTACADOS[c]); tk.set_fontweight("bold")
        if ax is axes[0]:
            ax.set_yticks(centros)
            ax.set_yticklabels([NOMBRE_CLASE[c] for c in ORDEN], fontsize=7.5)
            for c, tk in zip(ORDEN, ax.get_yticklabels()):
                if c in DESTACADOS:
                    tk.set_color(DESTACADOS[c]); tk.set_fontweight("bold")
        else:
            ax.set_yticks([])
        ax.set_title(ETIQUETA_PISTA[pista], fontsize=10.5, pad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05,
                          pad=0.30, ticks=[vmin, vmax])
        cb.ax.set_xticklabels([f"{vmin:.2f}", f"{vmax:.2f}"], fontsize=7.5)
        cb.set_label("similitud coseno", fontsize=7.5)
        cb.outline.set_visible(False)
    fig.suptitle("Similitud entre documentos, ordenados por tipo real", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.text(0.5, 0.965, "Los bloques de la diagonal son los tipos. El recuadro punteado marca "
             "los dos pares trampa: si ese bloque está oscuro, el modelo los confunde.",
             ha="center", fontsize=9, color=TINTA_2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DIR_FIG / "2_similitud.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_vecinos(docs, E, y, clase_consulta, archivo_salida, titulo, n=3):
    """Para una consulta, los n documentos mas parecidos DE OTRO TIPO.
    El vecino inmediato siempre es del mismo tipo; lo que distingue a cada
    representacion es con quien confundiria el documento a continuacion."""
    from PIL import Image

    q = int(np.where(y == clase_consulta)[0][0])
    fig, axes = plt.subplots(len(PISTAS), n + 1, figsize=(3.1 * (n + 1), 4.0 * len(PISTAS)))
    for fila, pista in enumerate(PISTAS):
        S = E[pista] @ E[pista].T
        S[q, q] = -np.inf
        propio = float(S[q][y == clase_consulta].max())
        ajenos = [i for i in np.argsort(-S[q]) if y[i] != clase_consulta][:n]
        for col in range(n + 1):
            ax = axes[fila, col]
            i = q if col == 0 else int(ajenos[col - 1])
            ax.imshow(Image.open(docs[i].png))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                for sp in ax.spines.values():
                    sp.set_color(DESTACADOS.get(clase_consulta, TINTA)); sp.set_linewidth(2.4)
                ax.set_title(f"CONSULTA · {NOMBRE_CLASE[clase_consulta]}\n"
                             f"mejor vecino del mismo tipo: {propio:.3f}", fontsize=9,
                             fontweight="bold", color=DESTACADOS.get(clase_consulta, TINTA))
                ax.set_ylabel(ETIQUETA_PISTA[pista].replace(" (", "\n("), fontsize=9.5,
                              fontweight="bold", labelpad=10)
            else:
                clase = y[i]
                color = DESTACADOS.get(clase, TINTA_2)
                brecha = propio - S[q, i]
                for sp in ax.spines.values():
                    sp.set_color(color); sp.set_linewidth(2.0 if clase in DESTACADOS else 1.0)
                ax.set_title(f"{NOMBRE_CLASE[clase]}\n{S[q, i]:.3f}   "
                             f"(a {brecha:.3f} del propio)", fontsize=8.5, color=color,
                             fontweight="bold" if clase in DESTACADOS else "normal")
    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=0.997)
    fig.text(0.5, 0.978, "Cuanto menor sea la brecha, más cerca está la representación de "
             "meterlos en la misma carpeta.", ha="center", fontsize=9.5, color=TINTA_2)
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    fig.savefig(DIR_FIG / archivo_salida, dpi=105, bbox_inches="tight")
    plt.close(fig)


def fig_metricas(res):
    """Small multiples: una metrica por panel. La fusion se destaca; el resto queda
    en neutro, porque la comparacion es 'cada modalidad sola' contra 'las dos juntas'."""
    paneles = [("ARI_auto", "ARI  ·  sin decirle cuántos tipos hay",
                "es lo que haría un sistema real"),
               ("ARI_k", "ARI  ·  con k = 8 dado", "escenario optimista"),
               ("1nn", "Acierto del vecino más cercano", "los 4 embeddings aciertan")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    etiquetas = [ETIQUETA_PISTA[p].replace(" (", "\n(") for p in PISTAS]
    for ax, (clave, titulo, nota) in zip(axes, paneles):
        vals = [res[p][clave] for p in PISTAS]
        colores = [ACENTO if p == "multimodal" else NEUTRO for p in PISTAS]
        ax.bar(range(len(PISTAS)), vals, 0.62, color=colores, edgecolor=SUP, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=10, color=TINTA,
                    fontweight="bold" if PISTAS[i] == "multimodal" else "normal")
        ax.set_xticks(range(len(PISTAS)))
        ax.set_xticklabels(etiquetas, fontsize=8.5)
        ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_axisbelow(True); ax.yaxis.grid(True)
        ax.set_title(titulo, fontsize=11, fontweight="bold", loc="left", pad=16)
        ax.text(0, 1.035, nota, transform=ax.transAxes, fontsize=8.5, color=TINTA_2)
    axes[0].set_ylabel("1.0 = reproduce exactamente los 8 tipos reales")
    fig.suptitle("Concordancia con los tipos reales", fontsize=13.5, fontweight="bold",
                 x=0.008, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(DIR_FIG / "4_metricas.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_contingencia(res, y):
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4))
    for ax, pista in zip(axes, PISTAS):
        et = res[pista]["et_auto"]
        grupos = sorted(set(et.tolist()),
                        key=lambda g: min(ORDEN.index(c) for c in set(y[et == g])))
        M = np.array([[int(((et == g) & (y == c)).sum()) for c in ORDEN] for g in grupos])
        ax.imshow(M, cmap=AZULES, vmin=0, vmax=M.max(), interpolation="nearest", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j]:
                    ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8.5,
                            color=SUP if M[i, j] > M.max() * 0.55 else TINTA)
        ax.set_xticks(range(len(ORDEN)))
        ax.set_xticklabels([NOMBRE_CLASE[c] for c in ORDEN], rotation=45, ha="right", fontsize=7.5)
        for c, t in zip(ORDEN, ax.get_xticklabels()):
            if c in DESTACADOS:
                t.set_color(DESTACADOS[c]); t.set_fontweight("bold")
        ax.set_yticks(range(len(grupos)))
        ax.set_yticklabels([f"carpeta {i}" for i in range(len(grupos))], fontsize=7.5)
        ax.set_title(f"{ETIQUETA_PISTA[pista]}  ({len(grupos)} grupos)", fontsize=10, pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Qué metió cada representación en cada carpeta  "
                 "(una fila con dos celdas llenas = dos tipos fusionados)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(DIR_FIG / "5_contingencia.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def margenes_trampa(E, y):
    """Margen = similitud media dentro del tipo  -  similitud media con el tipo gemelo.
    Cerca de 0 (o negativo) significa que la representacion no los distingue."""
    out = {}
    for pista in PISTAS:
        X = E[pista]
        S = X @ X.T
        fila = []
        for (a, b), _ in TRAMPAS:
            ia, ib = np.where(y == a)[0], np.where(y == b)[0]
            intra = np.mean([S[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)].mean(),
                             S[np.ix_(ib, ib)][np.triu_indices(len(ib), 1)].mean()])
            cruce = S[np.ix_(ia, ib)].mean()
            fila.append(float(intra - cruce))
        out[pista] = fila
    return out


def fig_trampas(marg):
    x = np.arange(len(PISTAS))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_axisbelow(True); ax.yaxis.grid(True)
    for j, p in enumerate(PISTAS):
        vals = marg[p]
        ax.plot([x[j], x[j]], [min(vals), max(vals)], color=TENUE, lw=2.5, zorder=1,
                solid_capstyle="round")
    for i, ((a, b), etq) in enumerate(TRAMPAS):
        vals = [marg[p][i] for p in PISTAS]
        ax.scatter(x, vals, s=170, c=SERIE[i], marker=["o", "^"][i], zorder=3,
                   edgecolors=SUP, linewidths=2,
                   label=f"{NOMBRE_CLASE[a]} vs {NOMBRE_CLASE[b]}  ·  {etq.replace(chr(10), ' ').lower()}")
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(13, -3),
                        fontsize=8.5, color=TINTA_2)
    peores = [min(marg[p]) for p in PISTAS]
    for xi, v in zip(x, peores):
        ax.annotate("peor caso", (xi, v), textcoords="offset points", xytext=(0, -20),
                    fontsize=8, color=TINTA, ha="center", fontweight="bold")
    ax.axhline(0, color=TINTA_2, lw=1)
    ax.set_xlim(-0.45, len(PISTAS) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETA_PISTA[p].replace(" (", "\n(") for p in PISTAS], fontsize=9.5)
    ax.set_ylabel("margen = similitud dentro del tipo − similitud con el tipo gemelo")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.13))
    ax.set_title("Un sistema vale lo que su peor caso:\nsolo la fusión mantiene separados los dos "
                 "pares a la vez", fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(DIR_FIG / "6_trampas.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_umbral(E, y):
    """Barrido del umbral de producto punto. No basta con que una representacion
    pueda acertar: tiene que acertar en un rango de umbral que alguien pueda
    encontrar sin conocer la respuesta."""
    from sklearn.metrics import adjusted_rand_score

    umbrales = np.round(np.arange(0.30, 0.995, 0.005), 3)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.4), sharey=True, sharex=True)
    for ax, pista in zip(axes, PISTAS):
        aris, grupos = [], []
        for u in umbrales:
            et, k, _ = agrupar(E[pista], umbral=float(u), enlace="completo")
            aris.append(adjusted_rand_score(y, et))
            grupos.append(k)
        aris, grupos = np.array(aris), np.array(grupos)
        hero = pista == "multimodal"
        color = ACENTO if hero else NEUTRO

        correcto = aris >= 0.999
        if correcto.any():
            ancho = correcto.sum() * 0.005
            centro = float(umbrales[correcto].mean())
            if ancho >= 0.02:
                ax.fill_between(umbrales, 0, 1.16, where=correcto, color=color, alpha=0.18,
                                step="mid", linewidth=0)
            else:   # una franja de 0.005 no se veria: se marca con una linea
                ax.axvline(centro, color=color, lw=1.6, ls=(0, (2, 2)), zorder=2)
            ax.annotate(f"acierta exacto\nen {ancho:.3f} de umbral",
                        (min(max(centro, 0.47), 0.83), 0.34),
                        ha="center", fontsize=8.5, color=TINTA, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.34", fc=SUP, ec=color, lw=1.2))
        else:
            ax.annotate("nunca reproduce\nlos 8 tipos exactos", (0.65, 0.34), ha="center",
                        fontsize=8.5, color=TINTA_2,
                        bbox=dict(boxstyle="round,pad=0.34", fc=SUP, ec=TENUE, lw=1))
        ax.plot(umbrales, aris, color=color, lw=2.4 if hero else 1.8, zorder=3)
        ax.scatter([umbrales[aris.argmax()]], [aris.max()], s=42, color=color,
                   edgecolors=SUP, linewidths=1.6, zorder=4)
        ax.annotate(f"máx {aris.max():.3f}", (umbrales[aris.argmax()], aris.max()),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=9, color=TINTA, fontweight="bold" if hero else "normal")

        ax.set_ylim(0, 1.16); ax.set_xlim(0.30, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_axisbelow(True); ax.yaxis.grid(True)
        ax.set_title(ETIQUETA_PISTA[pista], fontsize=10.5, pad=10)
        ax.set_xlabel("umbral de producto punto")
    axes[0].set_ylabel("ARI frente a los 8 tipos reales")
    fig.suptitle("Agrupar por umbral en vez de por número de grupos", fontsize=13.5,
                 fontweight="bold", x=0.007, ha="left", y=1.10)
    fig.text(0.007, 1.03, "La zona marcada es el rango de umbral donde la representación "
             "reproduce exactamente los 8 tipos; cuanto más ancha, más fácil de encontrar sin "
             "conocer la respuesta.",
             ha="left", fontsize=9.5, color=TINTA_2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(DIR_FIG / "7_umbral.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    estilo()
    DIR_FIG.mkdir(exist_ok=True)
    docs = cargar_documentos()
    y = leer_etiquetas(docs)
    E = obtener_embeddings(docs)
    k_real = len(set(y.tolist()))

    res = {p: metricas(E[p], y, k_real) for p in PISTAS}

    enc = f"{'representacion':<30}{'k auto':>8}{'ARI auto':>10}{'NMI auto':>10}{'ARI k=8':>10}{'pureza':>9}{'1-NN':>8}"
    print("\n" + enc)
    print("-" * len(enc))
    for p in PISTAS:
        r = res[p]
        print(f"{ETIQUETA_PISTA[p]:<30}{r['k_auto']:>8}{r['ARI_auto']:>10.3f}"
              f"{r['NMI_auto']:>10.3f}{r['ARI_k']:>10.3f}{r['pureza_k']:>9.3f}{r['1nn']:>8.3f}")

    marg = margenes_trampa(E, y)
    print("\nMargen de separacion en los pares trampa (intra - cruzado):")
    for p in PISTAS:
        print(f"  {ETIQUETA_PISTA[p]:<30} texto compartido {marg[p][0]:+.3f}   "
              f"layout compartido {marg[p][1]:+.3f}")

    print("\nGenerando figuras ...")
    fig_dataset(docs, y)
    fig_similitud(E, y)
    fig_vecinos(docs, E, y, "factura_narrativa", "3_vecinos_factura_narrada.png",
                "¿Con qué OTRO tipo confundiría cada representación una FACTURA NARRADA?")
    fig_vecinos(docs, E, y, "formulario_afiliacion", "3b_vecinos_afiliacion.png",
                "¿Con qué OTRO tipo confundiría cada representación un "
                "FORMULARIO DE AFILIACIÓN?")
    fig_metricas(res)
    fig_contingencia(res, y)
    fig_trampas(marg)
    fig_umbral(E, y)
    for f in sorted(DIR_FIG.glob("*.png")):
        print("  ", f.relative_to(RAIZ))

    import json
    (RAIZ / "data" / "resultados.json").write_text(json.dumps(
        {p: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in res[p].items()} | {"margenes": marg[p]} for p in PISTAS},
        indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
