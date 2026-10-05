<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/henkia-logo-white.png">
    <img alt="Henkia" src="../assets/henkia-logo.png" width="320">
  </picture>
</p>

# Clasificar documentos por texto **y** por layout

Demo de embeddings multimodales: toma una carpeta con documentos PDF mezclados,
los agrupa sin supervisión y crea subcarpetas con los documentos ya clasificados.
La comparación es contra las alternativas que solo miran el texto.

El punto que demuestra: **dos documentos pueden decir lo mismo y ser tipos
distintos, o decir cosas distintas y ser el mismo tipo.** Esa diferencia está en
la estructura de la página, no en las palabras.

---

## Resultado

| Representación | Grupos que encuentra | ARI (sin decirle cuántos tipos hay) | ARI (con k = 8) |
|---|---|---|---|
| Solo texto — TF-IDF léxico | 7 | 0.850 | 0.804 |
| Solo texto — embedding denso | 7 | 0.850 | 0.853 |
| Solo imagen / layout — CLIP | 6 | 0.731 | 0.833 |
| **Multimodal (texto + layout)** | **8** | **1.000** | **1.000** |

ARI = 1.0 significa reproducir exactamente los 8 tipos reales. El corpus tiene
73 documentos en 8 tipos y **dos pares diseñados como trampa**:

- **Mismo texto, layout distinto** — `factura` (tabla con grilla) y
  `factura_narrativa` (la misma factura redactada como carta). Comparten datos,
  ítems y hasta las etiquetas de campo: *código, descripción, cantidad, valor
  unitario, subtotal, IVA 19%, total a pagar*. Los modelos de texto los fusionan.
- **Mismo layout, texto distinto** — `historia_clinica` y
  `formulario_afiliacion`: misma geometría de formulario (banda de título,
  secciones en cajas, fila de celdas, tabla, firmas), dominios léxicos
  distintos. El modelo visual los fusiona.

Solo la fusión de las dos señales separa los dos pares a la vez.

## Instalación

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --torch-backend=auto \
  torch torchvision transformers sentence-transformers \
  scikit-learn matplotlib numpy pandas reportlab pymupdf pillow \
  streamlit plotly
```

Los modelos se descargan solos la primera vez (~1 GB). Usa GPU si hay al menos
2 GB libres; si no, cae a CPU sin fallar (73 páginas tardan ~1 min en CPU).

## Uso

El corpus no está en el repositorio: se genera en local con
`generar_dataset.py`, y siempre sale igual. Las empresas, bancos, EPS y personas
de los documentos son ficticios.

```bash
cd src

# 1. genera el corpus sintético en data/entrada/  (73 PDFs, nombres sin pista)
../.venv/bin/python generar_dataset.py

# 2a. clasificar diciendo cuántos grupos hay (o dejando que lo escoja la silueta)
../.venv/bin/python clasificar.py --k auto
../.venv/bin/python clasificar.py --todas --k 8

# 2b. clasificar SIN decir cuántos grupos hay: por umbral de producto punto
../.venv/bin/python clasificar.py --barrido          # ¿qué umbral usar?
../.venv/bin/python clasificar.py --umbral 0.825

# 3. métricas y figuras en figuras/
../.venv/bin/python evaluar.py
```

### Agrupar por umbral de producto punto

Todos los vectores están normalizados a norma 1, así que **producto punto =
coseno = 1 − distancia**. En vez de decir *«hay 8 tipos»*, se dice *«dos
documentos van juntos si su producto punto llega a 0.825»*:

```bash
../.venv/bin/python clasificar.py --umbral 0.825
```

```
### Multimodal (texto + layout)   ·   896 dimensiones, norma 1
    modo: umbral de producto punto >= 0.825, enlace completo
    garantia: TODO par dentro de un grupo tiene producto punto >= 0.825
    resultado: 8 grupos (no se le dijo cuantos buscar)
      grupo_00_anexos-comite-cordial              10 docs   pp interno min 0.867   margen +0.066
      grupo_01_paciente-antecedentes-consulta     10 docs   pp interno min 0.907   margen +0.095
      ...
```

Con `--enlace completo` (el predeterminado en este modo) la garantía es exacta:
*todo* par dentro de un grupo supera el umbral. Con `--enlace promedio` la
condición aplica al promedio entre grupos, y con `simple` solo a un par.

`--barrido` imprime la tabla de umbral → número de grupos, que es la forma
práctica de elegirlo: se busca una **meseta**, un rango donde el número de
grupos no cambie.

```
      umbral   grupos   grupo mayor   sueltos
       0.775        7            19         0  <--
       0.800        8            10         0  <--
       0.825        8            10         0
       0.850        8            10         0
       0.875        8            10         0
       0.900       12            10         3  <--
```

Cuánto aguanta cada representación antes de equivocarse (ARI = 1.0 significa
reproducir exactamente los 8 tipos):

| Representación | Mejor ARI por umbral | Ancho del rango que acierta exacto |
|---|---|---|
| Solo texto — TF-IDF | 0.853 | nunca acierta exacto |
| Solo texto — denso | 1.000 | 0.070 |
| Solo imagen — CLIP | 0.983 | nunca acierta exacto |
| **Multimodal** | **1.000** | **0.080** |

Por umbral, el texto denso también llega a los 8 tipos exactos, en una franja
parecida a la de la fusión. La diferencia está cuando el número de grupos lo
escoge la silueta (la tabla de **Resultado**): ahí el texto denso junta la
factura con la factura narrada y la fusión no.

### Ver los vectores y las distancias

Con una sola pista el detalle sale por defecto (`--sin-detalle` lo apaga; con
`--todas` hay que pedirlo con `--detalle`):

```
  grupo_00_anexos-comite-cordial   ·   10 documentos   ·   896 dimensiones
    producto punto dentro del grupo   min 0.867   medio 0.916   max 0.944
    lo mas parecido que hay fuera     0.801        margen +0.066

    archivo          norma  vector (primeras 6)                                ·centro   ·grupo   vecino mas cercano        ·fuera
    doc_039.pdf      1.000  [-0.0319 +0.0539 -0.0390 -0.0329 +0.0254 +0.0153 …]   0.973    0.928   doc_013.pdf    0.944      0.749
    doc_048.pdf      1.000  [-0.0080 +0.0537 -0.0303 -0.0486 -0.0025 +0.0065 …]   0.972    0.927   doc_013.pdf    0.940      0.744
```

| Columna | Qué es |
|---|---|
| `norma` | siempre 1.000 — por eso el producto punto es el coseno |
| `·centro` | producto punto con el centroide del grupo |
| `·grupo` | producto punto medio con sus compañeros de grupo |
| `vecino mas cercano` | el documento más parecido dentro del grupo |
| `·fuera` | el documento más parecido que quedó **fuera** del grupo |
| `margen` | `pp interno mínimo − pp máximo con el exterior`; si es negativo, el grupo se solapa con otro |

`--matriz` añade la matriz completa de productos punto de cada grupo, y
`--dims 12` muestra más componentes del vector.

### Qué queda en disco

```
data/salida/multimodal/
  asignaciones.csv                      archivo → grupo, con sus productos punto
  resumen_grupos.csv                    cohesión y margen de cada grupo
  grupo_00_anexos-comite-cordial/
    doc_001.pdf … doc_071.pdf           los documentos
    vectores.npy                        matriz (n × dims) del grupo
    vectores.csv                        los mismos vectores, componente a componente
    productos_punto.csv                 matriz de productos punto entre ellos
```

### Con tus propios documentos

```bash
../.venv/bin/python clasificar.py --entrada ~/mis_pdfs --salida ~/clasificados --umbral 0.8
```

Opciones útiles:

| Opción | Para qué |
|---|---|
| `--k auto` / `--k 8` | número de grupos automático (silueta) o fijo |
| `--umbral 0.825` | producto punto mínimo para caer en el mismo grupo |
| `--barrido` | tabla de umbral → número de grupos, para elegirlo |
| `--enlace completo/promedio/simple` | criterio de unión |
| `--detalle` / `--matriz` / `--dims N` | vectores y distancias en consola |
| `--alpha 0.5` | peso del texto en la fusión (0 = solo layout, 1 = solo texto) |
| `--pista visual` | agrupar con una sola modalidad |
| `--modelo-visual google/siglip-base-patch16-384` | codificador visual alternativo |
| `--recalcular` | ignora la caché de embeddings |

Si un PDF viene escaneado (sin capa de texto), el script lo avisa: para esos la
única señal disponible es la visual, que es justamente el caso donde más pesa.

## Interfaz gráfica

`src/interfaz.py` calcula embeddings con **Qwen/Qwen3-VL-Embedding-2B
desplegado en vLLM** (`http://localhost:8001`, vectores de 2048 dimensiones con
norma 1) y tiene seis pestañas:

- **Intro**: qué es un embedding multimodal, con un diagrama de tres entradas
  reales (un texto, una foto y la página de un PDF) y el vector que sale de cada
  una.
- **Comparar archivos**: dos archivos (PDF, imagen o texto) y un tercero
  opcional; muestra los vectores, los ángulos entre ellos y una proyección 3D.
- **Agrupar una carpeta**: todos los archivos de una carpeta, agrupados por un
  punto de corte de coseno que se elige con un deslizador; dibuja los grupos en
  2D/3D y crea una subcarpeta por grupo.
- **Comparar texto e imagen**: un texto escrito contra una imagen; con varias
  descripciones, cuál se parece más a la imagen.
- **Agrupar por texto**: uno o varios textos, y cada archivo de una carpeta va
  al texto que más se le parece si pasa el punto de corte; mismos gráficos y
  una subcarpeta por texto.
- **Conclusiones**: lo que nos llevamos del demo.

```bash
make desplegar      # verifica el modelo, levanta la interfaz y abre http://localhost:8501
make estado         # qué está arriba: modelo vLLM e interfaz
make logs           # sigue el log (.run/interfaz.log)
make detener        # baja la interfaz
```

`make desplegar` también reinicia si ya estaba corriendo. No arranca ni detiene
el modelo: solo consulta si responde, y si no responde, no levanta nada.
Variables: `PUERTO=8502`, `ABRIR=no`, `URL_EMBEDDING=…` y `DIRECCION=0.0.0.0`
para abrirla desde otro equipo de la red (por defecto escucha solo en
`localhost`).

Sin make: `cd src && ../.venv/bin/streamlit run interfaz.py`.

La instrucción del embedding y las dimensiones (barra lateral) valen para
todas las pestañas.

### Intro

Un diagrama del mecanismo: «una vaca en el campo», la foto de una vaca de
`../text_examples` y la página 1 de una factura del demo (con su texto) entran
al mismo modelo, pasan por las mismas cuatro etapas (mensaje con instrucción,
tokens, transformer, último token normalizado) y salen como un vector cada una.
Los valores dibujados y los cosenos del pie se calculan con el modelo desplegado:
el texto queda más cerca de la foto de la vaca (0.51) que de la factura (0.07).

Debajo explica la **norma** (el largo del vector) y por qué con norma 1 el
producto punto es el coseno. También muestra que no todos los modelos
normalizan: con las mismas entradas, Qwen3-VL entrega norma 1.000, CLIP 10.9
(foto) y 8.8 (texto), y MiniLM 4.6 o 3.6 según el texto. Si no vienen
normalizados, hay que dividir por la norma o usar el coseno explícito
(A·B / (‖A‖ ‖B‖)); una figura muestra que al duplicar el largo se duplica el
producto punto pero no el coseno.

### Comparar archivos

| Parte | Qué es |
|---|---|
| Métricas | coseno, ángulo θ y distancia ‖A − B‖; con C, el ángulo, el coseno y la distancia de cada par |
| Instrucción | texto que guía al embedding (barra lateral); se aplica igual a todos los archivos |
| Proyección 3D | base `e₁ = A`, `e₂ = parte de B ⟂ A`, `e₃ = parte de C ⟂ ambos`: todos los ángulos son los reales, sin distorsión de PCA |
| Valores de los vectores | componentes de cada vector, producto punto de cada par y la tabla completa descargable en CSV |
| Ángulo por modalidad | θ para cada combinación texto / imagen / texto + imagen, por cada par |

Detalles que importan:

- **Todo se envía en formato `messages`**, el único que aplica la plantilla del
  modelo. Así texto, imagen y texto + imagen caen en el mismo espacio y se
  pueden comparar entre sí (un `.txt` contra una foto, por ejemplo). La misma
  frase enviada por `input` da otro vector: coseno ~0.4 con el de `messages`.
- **Matryoshka**: el selector de dimensiones (64 … 2048) trunca y renormaliza
  en local; da exactamente lo mismo que el parámetro `dimensions` del servidor.
- **La instrucción va como mensaje `system`** y cambia el vector. La plantilla
  usa *"Represent the user's input."* cuando no se envía ninguna (mandarla
  explícita da el mismo vector). Con factura contra factura narrada, el coseno
  pasa de 0.781 a 0.721 con *"Represent the document type (invoice, letter,
  form, ...) of this page."*.
- **Con dos archivos la vista 3D es plana a propósito**: dos vectores siempre
  generan un plano, en 3 o en 2048 dimensiones. Un PCA tampoco los saca de ahí
  (solo inclina el plano) y además deforma el ángulo: en factura contra factura
  narrada mostraría 9.9° donde hay 38.7°. Para una vista realmente 3D hace falta
  un tercer vector, y eso es lo que aporta el archivo C.
- A diferencia de CLIP a 224 px, Qwen3-VL **sí lee** la página a 110 dpi
  (~1 100 tokens de imagen), así que su modalidad «imagen» no es solo layout.
- **vLLM en bf16 no es del todo determinista**: el vector de un archivo depende
  un poco de con qué otras peticiones se agrupa y de lo que haya en su caché de
  prefijos (coseno ~0.9998 entre llamadas con el mismo archivo). Los ángulos
  pueden moverse unas décimas de grado entre sesiones; dentro de una sesión la
  interfaz guarda los vectores y lo que se ve es estable.

### Agrupar una carpeta

Toma los archivos de una carpeta (sin entrar a subcarpetas; por defecto, los 73
PDFs del demo), calcula sus embeddings y los agrupa con el mismo árbol
aglomerativo sobre distancia coseno que `clasificar.py`, cortado donde diga el
**punto de corte**.

| Parte | Qué es |
|---|---|
| Punto de corte | coseno mínimo para quedar en el mismo grupo. Con enlace completo es una garantía: todo par de documentos dentro de un grupo lo supera |
| Dendrograma | el árbol de uniones con el eje en coseno y la línea de corte: lo que cuelga por debajo queda en un mismo grupo |
| Grupos en 2D / 3D | PCA o t-SNE; cada grupo con su color y su forma, y su número escrito en el centro |
| Tabla | por grupo: documentos, coseno mínimo y medio, lo más parecido que quedó afuera y el margen |
| Generar carpetas | `<salida>/qwen3vl/grupo_XX_nombre/` con los documentos, sus vectores y la matriz de cosenos, más `asignaciones.csv`, `resumen_grupos.csv` y `corte.json` con los parámetros usados |

- Los embeddings se calculan una vez por carpeta, modalidad e instrucción
  (`data/cache/qwen3vl_<hash>.npz`); después, mover el punto de corte es
  instantáneo.
- Un archivo que no tiene la modalidad elegida va con la que tenga (una imagen
  suelta no tiene texto): todo cae en el mismo espacio.
- Si la carpeta, o la de arriba, trae un `etiquetas.csv` (`archivo,clase_real`)
  que cubre todos sus archivos, se muestra el ARI contra esos tipos reales. Así
  pasa con `data/entrada` y con `../text_examples/mezclados`.
- Con el corpus del demo y texto + imagen, la partición es estable entre
  0.525 y 0.600: 7 grupos, ARI 0.85. Qwen3-VL lee la página, así que junta
  factura con factura narrada (la fusión MiniLM + CLIP del demo sí las separa).
- Con las fotos de `../text_examples/mezclados` y enlace completo, cualquier
  corte entre 0.17 y 0.34 da los 3 tipos exactos (ARI 1.0): entre dos fotos el
  coseno es mucho más bajo que entre dos documentos, por eso el deslizador baja
  a 0.05.
- «Generar carpetas» reemplaza `qwen3vl/` solo si es una salida anterior de la
  herramienta (tiene `asignaciones.csv`); si no, no la toca.
- Con más de 9 grupos, los que sobran van en gris en el gráfico: solo tres
  colores de la paleta se distinguen entre sí en un scatter, así que se
  combinan con tres formas.

### Comparar texto e imagen

A la izquierda se escribe un texto; a la derecha se sube una imagen (de un PDF
se usa solo la imagen de la página 1). El modelo pone los dos en el mismo
espacio, así que se comparan por coseno y ángulo, con la misma proyección 3D y
los mismos valores que en «Comparar archivos». Con varias descripciones, una
por línea, un gráfico las ordena por su parecido con la imagen y se elige cuál
ver en detalle.

**La instrucción del texto decide si funciona.** En búsqueda, la consulta lleva
una instrucción que dice qué buscar y la imagen, la genérica. Con la genérica en
los dos lados:

| Descripción | contra la imagen de una factura |
|---|---|
| con *"Represent the user's input."* | «una factura de venta…» 0.198 — gana «una historia clínica» (0.472) |
| con *"Find the document image that matches the description."* | «una factura de venta…» **0.707** — gana la correcta |

Por eso el texto trae esa instrucción por defecto (editable en la pestaña) y la
imagen usa la de la barra lateral. Si el campo se vacía, el texto va con la
genérica; debajo del campo se ve cuál se está usando, y «Usar la recomendada» la
devuelve. También funciona con fotos: contra la de un
gato negro, «un gato negro sentado» 0.55, «un gato blanco» 0.38, «un perro
jugando en el parque» 0.22.

### Agrupar por texto

Se escriben uno o varios textos, uno por línea. Cada archivo de la carpeta (por
defecto, `../text_examples/mezclados`) va al texto que más se le parece, si su
coseno con él llega al **punto de corte**; si no, queda sin asignar. Con un solo
texto se separan los archivos que se le parecen del resto; con varios, se
reparten entre ellos. El texto lleva la misma instrucción de búsqueda que en
«Comparar texto e imagen».

| Parte | Qué es |
|---|---|
| Reparto | una barra por archivo con su coseno contra el texto más parecido, del color de ese texto, y la línea de corte |
| Grupos en 2D / 3D | PCA o t-SNE como en «Agrupar una carpeta»; cada texto aparece como una ✕ entre sus archivos |
| Pureza | con `etiquetas.csv`: de los archivos asociados, cuántos son del tipo real más común de su grupo |
| Generar carpetas | `<salida>/qwen3vl_texto/grupo_XX_<texto>/`, más `grupo_XX_sin-asignar/` si queda alguno |

Con las 32 fotos mezcladas y el corte por defecto (0.29):

| Textos | Resultado |
|---|---|
| «una vaca» | las 11 vacas asociadas, las otras 21 sin asignar (pureza 100%) |
| «un árbol», «una vaca», «un gato» | cada foto a su texto: 10, 11 y 11 (pureza 100%) |

**Por qué importa:** con un texto se puede buscar en una base de conocimiento
por lo que muestran sus imágenes, no solo por lo que dicen sus palabras. Una
descripción como «una factura con tabla de productos» o «un diagrama de
arquitectura» encuentra las páginas e imágenes de la documentación que le
corresponden, aunque nadie las haya etiquetado.

El corte importa más con un solo texto: «un árbol» deja fuera todo lo ajeno
solo si el corte está entre 0.28 y 0.32. Con PCA ajustado a las fotos, la ✕ de
cada texto cae hacia sus fotos pero más cerca del centro, porque texto e imagen
no ocupan la misma región del espacio.

### Conclusiones

Lo que nos llevamos, en cuatro bloques. Cada conclusión dice en qué pestaña se ve.

| Bloque | Conclusión |
|---|---|
| Qué son | una neurociencia aplicada a la IA: se estudia cómo una red representa los conceptos en sus vectores |
| Para qué sirven | ordenan documentos empresariales mal manejados (facturas, contratos, convenios): el corpus del demo se agrupa sin decir cuántos tipos hay |
| | encuentran estructuras complejas sin etiquetas, de donde salen conceptos para estrategia o análisis |
| | ahorran tokens a los agentes: se busca primero lo relevante y solo eso entra al prompt del modelo fundacional |
| Por qué son económicos | no generan tokens: leen la entrada una vez y entregan un solo vector |
| | son más baratos de entrenar que los grandes LLMs: el de este demo tiene 2 mil millones de parámetros |
| Qué hay que cuidar | conceptos simples, mal aplicados: hoy se ven RAGs y bases de conocimiento mal construidos por no entender la matemática de fondo (norma, producto punto, coseno, punto de corte) |
| | no truncar el vector (abajo) |

**No truncar el vector.** Tienta hacerlo: el modelo entrega 2048 dimensiones y en
pgvector un índice HNSW o IVFFlat sobre `vector` acepta hasta 2000. Al truncar
sin renormalizar la norma deja de ser 1 y el producto interno (`<#>`, el que
pgvector recomienda para vectores normalizados) ya no es el coseno. Medido con
«una vaca» (instrucción de búsqueda) contra las 32 fotos mezcladas y el corte
0.29:

| Dimensiones | Norma de cada foto sin renormalizar | Vacas que pasan, sin renormalizar | Vacas que pasan, renormalizado |
|---|---|---|---|
| 2048 | 1.000 | 11 de 11 | 11 de 11 |
| 2000 | 0.978 – 0.990 | 11 | 11 |
| 1024 | 0.666 – 0.714 | **1** | 11 |
| 256 | 0.308 – 0.376 | 0 | 9 |

Renormalizar arregla la norma, pero el corte calibrado con 2048 no se traslada.
Mejor guardar el vector completo como `halfvec(2048)`, que se indexa hasta 4000
dimensiones.

Los textos están en `src/pestana_conclusiones.py`.

### Imágenes de ejemplo: `../text_examples/`

Está en la raíz del repositorio, junto a esta carpeta.

| Ruta | Qué hay |
|---|---|
| `da3c8b…jpg`, `vaca.png` | la foto de un gato negro y la de una vaca que ya estaban en la carpeta |
| `arboles/`, `vacas/`, `gatos/` | 10 fotos de cada tipo de Wikimedia Commons (640 px) |
| `mezclados/` | las 32 en orden aleatorio, con nombres sin pista (`img_01.jpg` …); de las dos sueltas van copias a 640 px |
| `etiquetas.csv` | tipo real de cada foto de `mezclados/` y su archivo original |
| `CREDITOS.md` | autor, licencia y fuente de cada foto |

Las de Commons vienen de sus categorías de imágenes de calidad de gatos, ganado
y árboles, con licencias CC BY, CC BY-SA, FAL y GFDL; cada una conserva la de
su autor. Las dos sueltas no tienen origen ni licencia registrados.

## Cómo funciona

`src/nucleo.py` calcula cuatro representaciones del mismo corpus:

| Pista | Qué mira |
|---|---|
| `tfidf` | bolsa de palabras, sin modelo neuronal |
| `texto` | embedding denso multilingüe, por trozos de 150 palabras promediados |
| `visual` | imagen de la primera página por CLIP, a 224 px |
| `multimodal` | concatenación de las dos anteriores, normalizada |

Detalle que importa: **a 224 píxeles el codificador visual no puede leer las
letras**. Lo único que sobrevive de la página es la estructura — columnas,
grillas, cajas, bloques, espacio en blanco. Por eso la pista `visual` es una
medida limpia de layout, no un OCR encubierto.

La página se rellena a cuadrado antes de reescalar, para no perder la mitad del
documento en el recorte central que hace el preprocesador por defecto.

La fusión concatena los dos bloques ya normalizados con pesos `√α` y `√(1−α)`,
de modo que el coseno del vector fusionado es exactamente
`α · coseno_texto + (1−α) · coseno_visual`.

Agrupamiento: jerárquico aglomerativo con distancia coseno. Con `--k auto` el
número de grupos se escoge por silueta; con `--umbral` no se escoge ninguno y el
árbol se corta a la altura que fije el producto punto. Ambos modos evitan lo que
en producción no se tiene: **saber cuántos tipos de documento hay.**

## Figuras

`evaluar.py` las deja en `figuras/`, junto con `data/resultados.json`. Se
generan en local y no se suben al repositorio.

| Archivo | Qué muestra |
|---|---|
| `1_dataset.png` | los 8 tipos del corpus |
| `2_similitud.png` | matriz de similitud por representación, con los pares trampa marcados |
| `3_vecinos_factura_narrada.png` | con qué otro tipo confundiría cada representación una factura narrada |
| `3b_vecinos_afiliacion.png` | lo mismo para un formulario de afiliación |
| `4_metricas.png` | concordancia con los tipos reales |
| `5_contingencia.png` | qué metió cada representación en cada carpeta |
| `6_trampas.png` | margen de separación en los dos pares trampa |
| `7_umbral.png` | barrido del umbral: en qué rango acierta cada representación |

## Advertencias honestas

- El corpus es **sintético y controlado**: permite aislar el efecto del layout,
  no predice el desempeño sobre una colección real.
- Por umbral de producto punto, el embedding denso de texto también reproduce
  los 8 tipos, en una franja casi tan ancha como la de la fusión (0.070 contra
  0.080). La ventaja clara de la fusión aparece cuando nadie fija ni el número
  de tipos ni el umbral.
- El vecino más cercano es correcto en ~99% de los casos en las cuatro pistas.
  El fallo de las modalidades sueltas está en el agrupamiento, no en la
  recuperación uno a uno.
- CLIP no está entrenado sobre documentos. Un codificador específico de páginas
  (ColPali, ColQwen2, DSE) daría una señal de layout bastante mejor, a costa de
  unos 6 GB de descarga y de VRAM.
