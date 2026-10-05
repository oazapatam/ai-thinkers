"""
Pestana "Conclusiones": lo que nos llevamos del demo, en cuatro bloques (que
son, para que sirven, por que son economicos y que hay que cuidar). Cada
conclusion dice en que pestana se ve.

Los numeros de "No truncar el vector" se midieron con las 32 fotos de
../text_examples/mezclados, el texto "una vaca" con la instruccion de busqueda
y el corte por defecto de "Agrupar por texto" (0.29).
"""
from __future__ import annotations

import streamlit as st


def _miles(n: int) -> str:
    """2048 -> '2 048', con el mismo espacio fino que la Intro."""
    return f"{n:,}".replace(",", " ")


def _bloques(k: int) -> list[tuple[str, list[list[tuple[str, str, str]]]]]:
    """(titulo del bloque, filas de [(icono, conclusion, explicacion)]): las
    conclusiones de una fila van lado a lado."""
    return [
        ("Qué son", [[
            (":material/psychology:", "Una neurociencia aplicada a la IA",
             "Los embeddings abren un área nueva, parecida a la neurociencia pero aplicada a la "
             "IA: en vez de medir neuronas, se estudia cómo una red representa los conceptos en "
             "sus vectores. El ángulo entre dos vectores dice qué tan cerca los pone el modelo, "
             "sea un texto, una foto o la página de un PDF (**Intro**, **Comparar archivos**)."),
        ]]),
        ("Para qué sirven", [[
            (":material/folder_open:", "Ordenan documentos empresariales mal manejados",
             "Facturas, contratos, convenios: documentos mezclados, con nombres que no dicen nada "
             "y sin etiquetas. En el demo, 73 PDFs de 8 tipos (facturas, oficios, extractos "
             "bancarios, certificados laborales, historias clínicas…) llamados `doc_001.pdf`, "
             "`doc_002.pdf`… se agrupan sin decir cuántos tipos hay: la fusión de texto y "
             "layout recupera los 8 exactos (ARI 1.0) y Qwen3-VL encuentra 7 (ARI 0.85: junta la "
             "factura en tabla con la factura redactada como carta). **Agrupar una carpeta** "
             "deja una subcarpeta por grupo."),
        ], [
            (":material/hub:", "Encuentran estructura en los datos",
             "Identifican estructuras complejas sin que nadie las etiquete, y de ahí salen "
             "conceptos para estrategia o análisis. En **Agrupar una carpeta** los grupos salen "
             "de un punto de corte, sin decir cuántos hay."),
            (":material/savings:", "Ahorran tokens a los agentes",
             "Un agente no tiene que pasarle toda la base de conocimiento a un modelo "
             "fundacional grande: con embeddings busca primero lo relevante y solo eso entra al "
             "prompt. Menos tokens para responder. En **Agrupar por texto** un texto encuentra "
             "sus archivos."),
        ]]),
        ("Por qué son económicos", [[
            (":material/bolt:", "No generan tokens, solo un vector",
             "Un LLM escribe su respuesta token por token, y cada token es otra pasada por el "
             "modelo. Un modelo de embedding lee la entrada una vez y entrega un solo vector: "
             f"aquí, {_miles(k)} números por entrada."),
            (":material/model_training:", "Más baratos de entrenar",
             "Son mucho más pequeños que los grandes LLMs: el de este demo, "
             "Qwen3-VL-Embedding-2B, tiene 2 mil millones de parámetros. Entrenarlos cuesta "
             "mucho menos cómputo."),
        ]]),
        ("Qué hay que cuidar", [[
            (":material/functions:", "Conceptos simples, mal aplicados",
             "Norma, producto punto, coseno y punto de corte son conceptos simples. Aun así, hoy "
             "se ven RAGs y bases de conocimiento mal construidos por no entender la matemática "
             "de fondo: vectores sin normalizar comparados con producto interno (en la **Intro**, "
             "la foto de la vaca sale de CLIP con norma 10.9), vectores truncados, o un punto de "
             "corte calibrado con otro modelo, otra instrucción u otras dimensiones. Nada de eso "
             "da error: la búsqueda sigue respondiendo, solo que peor."),
        ], [
            (":material/warning:", "No truncar el vector",
             f"Tienta hacerlo: Qwen3-VL entrega {_miles(2048)} dimensiones y en **pgvector** un "
             f"índice HNSW o IVFFlat sobre `vector` acepta hasta {_miles(2000)}. Pero al "
             "truncar sin renormalizar la norma deja de ser 1, y el producto interno (`<#>`, el "
             "que pgvector recomienda para vectores normalizados) ya no es el coseno. Con «una "
             f"vaca» y las 32 fotos mezcladas, a {_miles(1024)} dimensiones solo 1 de las 11 "
             "vacas pasa el corte de 0.29; con el vector completo pasan las 11. Aun "
             "renormalizando, el corte calibrado no se traslada: a 256 pasan 9 (**Agrupar por "
             "texto**, bajando las dimensiones en la barra lateral). Mejor guardar el vector "
             "completo como `halfvec(2048)`, que se indexa hasta "
             f"{_miles(4000)} dimensiones."),
        ]]),
    ]


def mostrar(k: int) -> None:
    st.markdown("### Lo que nos llevamos")
    for titulo, filas in _bloques(k):
        st.markdown(f"#### {titulo}")
        for fila in filas:
            for col, (icono, conclusion, texto) in zip(st.columns(len(fila), gap="medium"), fila):
                with col, st.container(border=True, height="stretch"):
                    st.markdown(f"**{icono} {conclusion}**")
                    st.markdown(texto)
