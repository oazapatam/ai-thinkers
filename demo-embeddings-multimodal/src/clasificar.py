"""
Toma TODOS los PDFs de una carpeta, calcula embeddings, los agrupa y crea
subcarpetas con los documentos ya clasificados.

Dos modos de decidir los grupos:

  --k auto / --k 8      numero de grupos (automatico por silueta, o fijo)
  --umbral 0.80         corta por PRODUCTO PUNTO entre vectores: no se dice
                        cuantos grupos hay, se dice que tan parecidos tienen
                        que ser dos documentos para caer en el mismo.

Ejemplos:
  python src/clasificar.py                                  # multimodal, k automatico
  python src/clasificar.py --umbral 0.80                    # por umbral
  python src/clasificar.py --barrido                        # que umbral usar
  python src/clasificar.py --todas --k 8 --detalle          # las 4 pistas, con vectores
  python src/clasificar.py --entrada ~/pdfs --salida ~/out  # documentos propios
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from nucleo import (ALPHA, DIR_CACHE, DIR_ENTRADA, ENLACES, ETIQUETA_PISTA,
                    MODELO_TEXTO, MODELO_VISUAL, PISTAS, RAIZ, agrupar,
                    barrido_umbral, calcular_todas, cargar_documentos,
                    estadisticas_grupos, nombrar_grupos)

DIR_SALIDA = RAIZ / "data" / "salida"
CACHE = DIR_CACHE / "embeddings.npz"
UMBRALES_BARRIDO = [round(0.50 + 0.025 * i, 3) for i in range(21)]


def obtener_embeddings(docs, alpha=ALPHA, recalcular=False, modelo_texto=MODELO_TEXTO,
                       modelo_visual=MODELO_VISUAL) -> dict[str, np.ndarray]:
    archivos = [d.archivo for d in docs]
    firma = f"{alpha}|{modelo_texto}|{modelo_visual}|{docs[0].ruta.parent.resolve()}"
    if CACHE.exists() and not recalcular:
        z = np.load(CACHE, allow_pickle=True)
        if list(z["archivos"]) == archivos and str(z["firma"]) == firma:
            print(f"Embeddings leidos de cache ({CACHE.relative_to(RAIZ)})")
            return {p: z[p] for p in PISTAS}
    print("Calculando embeddings:")
    E = calcular_todas(docs, alpha=alpha, modelo_texto=modelo_texto,
                       modelo_visual=modelo_visual)
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, archivos=np.array(archivos), firma=firma, **E)
    return E


# --------------------------------------------------------------------------
# salida en consola
# --------------------------------------------------------------------------
def _vec(v: np.ndarray, dims: int) -> str:
    cuerpo = " ".join(f"{x:+.4f}" for x in v[:dims])
    return f"[{cuerpo} …]" if len(v) > dims else f"[{cuerpo}]"


def imprimir_detalle(stats: dict, nombres: dict, dims: int, matriz: bool):
    """Muestra el vector de cada documento y su producto punto con el resto
    del grupo. Los vectores tienen norma 1: producto punto = coseno."""
    for g, info in stats.items():
        print(f"\n  grupo_{g:02d}_{nombres[g]}   ·   {info['n']} documentos   ·   "
              f"{len(info['centroide'])} dimensiones")
        if info["n"] > 1:
            print(f"    producto punto dentro del grupo   min {info['par_min']:.3f}   "
                  f"medio {info['par_medio']:.3f}   max {info['par_max']:.3f}")
            print(f"    lo mas parecido que hay fuera     {info['externo_max']:.3f}"
                  f"        margen {info['margen']:+.3f}")

        enc = (f"    {'archivo':<16}{'norma':>6}  {'vector (primeras ' + str(dims) + ')':<{9 * dims + 4}}"
               f"{'·centro':>9}{'·grupo':>9}   {'vecino mas cercano':<26}{'·fuera':>9}")
        print("\n" + enc)
        print("    " + "-" * (len(enc) - 4))
        for d in sorted(info["docs"], key=lambda r: -r["con_centroide"]):
            vecino = f"{d['vecino']:<16}{d['vecino_sim']:>7.3f}" if d["vecino"] != "-" else "-" * 23
            print(f"    {d['archivo']:<16}{d['norma']:>6.3f}  "
                  f"{_vec(d['vector'], dims):<{9 * dims + 4}}"
                  f"{d['con_centroide']:>9.3f}{d['intra_media']:>9.3f}   "
                  f"{vecino:<26}{d['externo_sim']:>9.3f}")

        if matriz and info["n"] > 1:
            print(f"\n    matriz de productos punto del grupo_{g:02d}:")
            archivos = [d["archivo"] for d in info["docs"]]
            V = np.vstack([d["vector"] for d in info["docs"]])
            S = V @ V.T
            print("    " + " " * 16 + "".join(f"{a.replace('.pdf', ''):>9}" for a in archivos))
            for a, fila in zip(archivos, S):
                print(f"    {a:<16}" + "".join(f"{v:>9.3f}" for v in fila))


def imprimir_barrido(E: dict, pistas: list[str], enlace: str):
    print("\nBarrido de umbrales — cuantos grupos salen con cada valor del producto punto")
    print("(el mismo umbral NO significa lo mismo en cada representacion: cada modelo "
          "vive en su propio rango de coseno)\n")
    for pista in pistas:
        filas = barrido_umbral(E[pista], UMBRALES_BARRIDO, enlace=enlace)
        print(f"  {ETIQUETA_PISTA[pista]}")
        print(f"    {'umbral':>8}{'grupos':>9}{'grupo mayor':>14}{'sueltos':>10}")
        anterior = None
        for f in filas:
            marca = "  <--" if anterior is not None and f["grupos"] != anterior else ""
            print(f"    {f['umbral']:>8.3f}{f['grupos']:>9}{f['mayor']:>14}"
                  f"{f['sueltos']:>10}{marca}")
            anterior = f["grupos"]
        print()


# --------------------------------------------------------------------------
# escritura en disco
# --------------------------------------------------------------------------
def organizar(docs, etiquetas: np.ndarray, pista: str, stats: dict, nombres: dict,
              raiz_salida: Path) -> Path:
    """Crea <salida>/<pista>/<carpeta>/ con los PDFs, sus vectores completos y
    la matriz de productos punto del grupo."""
    destino = raiz_salida / pista
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    carpeta_de = {g: destino / f"grupo_{g:02d}_{nombres[g]}" for g in stats}
    for c in carpeta_de.values():
        c.mkdir()

    por_doc = {d["archivo"]: (g, d) for g, info in stats.items() for d in info["docs"]}
    filas = []
    for d in docs:
        g, info_doc = por_doc[d.archivo]
        shutil.copy2(d.ruta, carpeta_de[g] / d.archivo)
        filas.append({
            "archivo": d.archivo, "grupo": g, "carpeta": carpeta_de[g].name,
            "pp_centroide": round(info_doc["con_centroide"], 4),
            "pp_medio_grupo": round(info_doc["intra_media"], 4),
            "vecino_en_grupo": info_doc["vecino"],
            "pp_vecino": round(info_doc["vecino_sim"], 4),
            "mas_parecido_fuera": info_doc["externo"],
            "pp_fuera": round(info_doc["externo_sim"], 4),
        })

    with (destino / "asignaciones.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(sorted(filas, key=lambda r: r["archivo"]))

    with (destino / "resumen_grupos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "carpeta", "documentos", "pp_min", "pp_medio", "pp_max",
                    "pp_max_fuera", "margen"])
        for g, info in stats.items():
            w.writerow([g, carpeta_de[g].name, info["n"], round(info["par_min"], 4),
                        round(info["par_medio"], 4), round(info["par_max"], 4),
                        round(info["externo_max"], 4), round(info["margen"], 4)])

    for g, info in stats.items():
        archivos = [d["archivo"] for d in info["docs"]]
        V = np.vstack([d["vector"] for d in info["docs"]])
        np.save(carpeta_de[g] / "vectores.npy", V)
        with (carpeta_de[g] / "vectores.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archivo", "norma"] + [f"v{i:04d}" for i in range(V.shape[1])])
            for a, v in zip(archivos, V):
                w.writerow([a, round(float(np.linalg.norm(v)), 6)]
                           + [round(float(x), 6) for x in v])
        with (carpeta_de[g] / "productos_punto.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + archivos)
            for a, fila in zip(archivos, V @ V.T):
                w.writerow([a] + [round(float(x), 4) for x in fila])
    return destino


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pista", default="multimodal", choices=PISTAS)
    ap.add_argument("--todas", action="store_true", help="procesa las 4 pistas")
    ap.add_argument("--k", default="auto", help="numero de grupos, o 'auto' (silueta)")
    ap.add_argument("--umbral", type=float, default=None,
                    help="producto punto minimo para que dos documentos caigan juntos "
                         "(0-1). Reemplaza a --k.")
    ap.add_argument("--enlace", default=None, choices=list(ENLACES),
                    help="criterio de union (por defecto: 'completo' con --umbral, "
                         "'promedio' con --k)")
    ap.add_argument("--barrido", action="store_true",
                    help="tabla de umbral -> numero de grupos, y termina")
    ap.add_argument("--detalle", action="store_true",
                    help="muestra el vector y los productos punto de cada documento")
    ap.add_argument("--sin-detalle", action="store_true")
    ap.add_argument("--matriz", action="store_true",
                    help="ademas, la matriz completa de productos punto por grupo")
    ap.add_argument("--dims", type=int, default=6, help="componentes del vector a mostrar")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="peso del texto en la fusion")
    ap.add_argument("--recalcular", action="store_true")
    ap.add_argument("--entrada", default=None, help="carpeta con los PDFs")
    ap.add_argument("--salida", default=None, help="carpeta donde crear las subcarpetas")
    ap.add_argument("--modelo-texto", default=MODELO_TEXTO)
    ap.add_argument("--modelo-visual", default=MODELO_VISUAL)
    args = ap.parse_args()

    entrada = Path(args.entrada).expanduser().resolve() if args.entrada else DIR_ENTRADA
    salida = Path(args.salida).expanduser().resolve() if args.salida else DIR_SALIDA
    enlace = args.enlace or ("completo" if args.umbral is not None else "promedio")
    pistas = PISTAS if args.todas else [args.pista]

    docs = cargar_documentos(entrada)
    print(f"{len(docs)} documentos en {entrada}\n")
    E = obtener_embeddings(docs, alpha=args.alpha, recalcular=args.recalcular,
                           modelo_texto=args.modelo_texto, modelo_visual=args.modelo_visual)

    if args.barrido:
        imprimir_barrido(E, pistas, enlace)
        return

    if args.umbral is not None and len(pistas) > 1:
        print("  aviso: un mismo umbral NO significa lo mismo en cada representacion; "
              "cada modelo\n         vive en su propio rango de coseno. "
              "Use --barrido para elegir uno por pista.")

    # el detalle se muestra por defecto cuando se pide una sola pista
    detalle = args.detalle or (not args.todas and not args.sin_detalle)
    k = None if args.k == "auto" else int(args.k)

    for pista in pistas:
        X = E[pista]
        etiquetas, kk, siluetas = agrupar(X, k=k, umbral=args.umbral, enlace=enlace)
        stats = estadisticas_grupos(X, docs, etiquetas)
        nombres = nombrar_grupos(docs, etiquetas)
        destino = organizar(docs, etiquetas, pista, stats, nombres, salida)

        print(f"\n{'=' * 100}")
        print(f"### {ETIQUETA_PISTA[pista]}   ·   {X.shape[1]} dimensiones, norma 1")
        if args.umbral is not None:
            print(f"    modo: umbral de producto punto >= {args.umbral:.3f}, "
                  f"enlace {enlace}")
            if enlace == "completo":
                print(f"    garantia: TODO par dentro de un grupo tiene "
                      f"producto punto >= {args.umbral:.3f}")
            print(f"    resultado: {kk} grupos (no se le dijo cuantos buscar)")
        else:
            print(f"    modo: k = {kk}" +
                  (f" elegido por silueta ({siluetas[kk]:.3f})" if k is None else " (fijo)") +
                  f", enlace {enlace}")
        print(f"    {destino}/")
        for g, info in stats.items():
            print(f"      grupo_{g:02d}_{nombres[g]:<40} {info['n']:3d} docs   "
                  f"pp interno min {info['par_min']:.3f}   margen {info['margen']:+.3f}")

        if detalle:
            imprimir_detalle(stats, nombres, args.dims, args.matriz)


if __name__ == "__main__":
    main()
