"""
Genera un dataset sintetico de documentos PDF en espanol para demostrar
clasificacion con embeddings multimodales (texto + layout).

El dataset esta disenado con DOS TRAMPAS deliberadas:

  TRAMPA 1 (el texto falla) -> 'factura' vs 'factura_narrativa'
      Comparten casi todo el vocabulario (NIT, items, IVA, subtotal, total a
      pagar) pero el layout es radicalmente distinto: tabla con grilla vs
      carta en prosa a una columna.

  TRAMPA 2 (la vision falla) -> 'historia_clinica' vs 'formulario_afiliacion'
      y 'factura' vs 'extracto_bancario'
      Layouts geometricamente casi identicos (formulario con cajas / tabla
      densa de cifras) pero dominios lexicos distintos.

Solo la fusion de ambas modalidades resuelve los dos casos.
"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

W, H = A4
M = 45.0                      # margen
GRIS = colors.HexColor("#D9D9D9")
GRIS_SUAVE = colors.HexColor("#F2F2F2")
AZUL = colors.HexColor("#1F3864")
LINEA = colors.HexColor("#666666")

RAIZ = Path(__file__).resolve().parents[1]
DIR_ENTRADA = RAIZ / "data" / "entrada"
CSV_ETIQUETAS = RAIZ / "data" / "etiquetas.csv"


# --------------------------------------------------------------------------
# helpers de dibujo
# --------------------------------------------------------------------------
def t(c, x, y, s, font="Helvetica", size=9, color=colors.black):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, str(s))


def tc(c, x, y, s, font="Helvetica", size=9, color=colors.black):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawCentredString(x, y, str(s))


def tr(c, x, y, s, font="Helvetica", size=9, color=colors.black):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawRightString(x, y, str(s))


def ln(c, x, y, s, font="Helvetica", size=9, leading=13):
    """Escribe UNA linea y devuelve la y de la siguiente."""
    t(c, x, y, s, font, size)
    return y - leading


def parrafo(c, x, y, s, ancho, font="Helvetica", size=9, leading=12.5):
    """Escribe texto justificado-izquierda con salto de linea. Devuelve la y final."""
    c.setFont(font, size)
    c.setFillColor(colors.black)
    for linea in simpleSplit(s, font, size, ancho):
        c.drawString(x, y, linea)
        y -= leading
    return y


def caja(c, x, y, w, h, relleno=None, borde=LINEA, grosor=0.7):
    if relleno is not None:
        c.setFillColor(relleno)
    c.setStrokeColor(borde)
    c.setLineWidth(grosor)
    c.rect(x, y, w, h, stroke=1, fill=1 if relleno is not None else 0)


def banda(c, y, h, titulo, relleno=GRIS, size=9):
    caja(c, M, y, W - 2 * M, h, relleno=relleno)
    t(c, M + 5, y + h / 2 - 3, titulo, "Helvetica-Bold", size)


def campo(c, x, y, etiqueta, valor, ancho_etq=None, size=7.5):
    """Etiqueta pequena arriba, valor debajo (estilo formulario)."""
    t(c, x, y, etiqueta.upper(), "Helvetica", size - 0.7, colors.HexColor("#555555"))
    t(c, x, y - 10, valor, "Helvetica-Bold", size + 0.7)


def linea_h(c, x1, y, x2, grosor=0.6, color=LINEA):
    c.setStrokeColor(color)
    c.setLineWidth(grosor)
    c.line(x1, y, x2, y)


def cop(v):
    return "$" + f"{int(round(v)):,}".replace(",", ".")


# --------------------------------------------------------------------------
# pools de contenido (compartidos a proposito entre clases "trampa")
# --------------------------------------------------------------------------
EMPRESAS = [
    ("REPUESTOS DEL CAFE S.A.S.", "900.123.456-7", "Cra. 43A # 18-11, Medellin"),
    ("SUMINISTROS MEDICOS ANDINOS LTDA.", "830.987.654-3", "Av. El Dorado # 68-90, Bogota"),
    ("TECNOLOGIA Y SOFTWARE ABURRA S.A.S.", "901.445.221-9", "Cl. 10 # 32-115, Medellin"),
    ("DISTRIBUIDORA LOGISTICA DEL NORTE S.A.", "890.334.778-1", "Cl. 76 # 54-11, Barranquilla"),
    ("INGENIERIA Y MONTAJES VALLE S.A.S.", "805.221.663-4", "Cra. 100 # 11-60, Cali"),
    ("PAPELERIA Y DOTACIONES EL PROGRESO", "811.556.902-8", "Cl. 49 # 50-21, Medellin"),
]

CLIENTES = [
    ("DATOS DEL ROBLE S.A.", "999.100.201-3"), ("ALCALDIA DE VALLE ALTO", "999.100.202-1"),
    ("SALUD CORDILLERA S.A.", "999.100.203-8"), ("UNIVERSIDAD DEL MIRADOR", "999.100.204-6"),
    ("CLINICA LOS ARRAYANES S.A.", "999.100.205-4"), ("GRUPO PIEDRALTA S.A.", "999.100.206-2"),
    ("TRANSPORTES RIO SERENO", "999.100.207-9"), ("COMFAVALLE", "999.100.208-7"),
]

# mismo pool de items para 'factura' y 'factura_narrativa' -> solapamiento lexico maximo
ITEMS = [
    ("RP-1042", "Filtro de aceite para motor diesel", 45900),
    ("RP-2210", "Kit de pastillas de freno delanteras", 189500),
    ("SM-3301", "Caja de guantes de nitrilo talla M x 100", 38700),
    ("SM-4120", "Jeringa desechable 10 ml paquete x 50", 27400),
    ("TS-0090", "Licencia anual de software de gestion documental", 1250000),
    ("TS-0455", "Hora de soporte tecnico especializado", 96000),
    ("LG-7781", "Servicio de transporte de carga urbano", 310000),
    ("IM-5502", "Mantenimiento preventivo de equipo industrial", 780000),
    ("PD-1190", "Resma de papel bond tamano carta", 18900),
    ("PD-2234", "Dotacion de uniforme empresarial completo", 156000),
    ("SM-6612", "Suministro de oxigeno medicinal cilindro 6 m3", 142000),
    ("TS-7788", "Desarrollo de modulo de integracion via API", 2400000),
]

CIUDADES = ["Medellin", "Bogota D.C.", "Cali", "Barranquilla", "Bucaramanga", "Pereira"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

NOMBRES = ["Carlos Andres Restrepo Villa", "Maria Fernanda Gomez Osorio",
           "Juan Pablo Arango Mesa", "Luz Adriana Cardona Rios",
           "Santiago Alberto Munoz Pena", "Diana Carolina Herrera Lopez",
           "Jorge Enrique Salazar Duque", "Paula Andrea Zapata Ruiz",
           "Andres Felipe Ocampo Vargas", "Claudia Patricia Jimenez Toro"]

CARGOS = ["Analista de Procesos", "Coordinador Administrativo", "Ingeniero de Desarrollo",
          "Auxiliar Contable", "Especialista en Datos", "Jefe de Compras",
          "Profesional de Talento Humano", "Director de Operaciones"]

BANCOS = [("BANCO ALTIPLANO S.A.", "901"), ("BANCO DE LA SABANA", "902"),
          ("BANCO RIBERA S.A.", "903"), ("BANCO PRADERA", "904")]

MOVIMIENTOS = [
    "ABONO NOMINA EMPLEADOR", "PAGO PSE SERVICIOS PUBLICOS", "COMPRA DATAFONO SUPERTIENDA",
    "RETIRO CAJERO AUTOMATICO CC", "TRANSFERENCIA INTERBANCARIA ACH", "CUOTA DE MANEJO TARJETA",
    "PAGO OBLIGACION CREDITO ROTATIVO", "CONSIGNACION NACIONAL EN EFECTIVO",
    "DEBITO AUTOMATICO SEGURO DE VIDA", "IMPUESTO GOBIERNO 4X1000",
    "COMPRA DATAFONO ESTACION SERVICIO", "TRANSFERENCIA SUCURSAL VIRTUAL",
    "INTERESES ABONADOS CUENTA AHORROS", "PAGO PROVEEDOR FACTURA ELECTRONICA",
    "RETENCION EN LA FUENTE RENDIMIENTOS", "COMPRA EN LINEA COMERCIO ELECTRONICO",
]

# clinico
DIAGNOSTICOS = [
    ("J00X", "Rinofaringitis aguda (resfriado comun)"),
    ("I10X", "Hipertension esencial primaria"),
    ("E119", "Diabetes mellitus tipo 2 sin complicaciones"),
    ("M545", "Lumbago no especificado"),
    ("K297", "Gastritis no especificada"),
    ("J453", "Asma persistente moderada"),
]
MOTIVOS = [
    "Paciente acude por cuadro clinico de cinco dias de evolucion consistente en tos seca, "
    "odinofagia y sensacion febril no cuantificada, sin disnea asociada.",
    "Consulta por dolor lumbar de inicio insidioso, de intensidad moderada, que aumenta con la "
    "bipedestacion prolongada y mejora parcialmente con el reposo.",
    "Control de programa de riesgo cardiovascular. Refiere adherencia parcial al tratamiento "
    "farmacologico y dificultad para sostener el plan nutricional indicado.",
    "Paciente remitido por servicio de urgencias por epigastralgia recurrente asociada a "
    "pirosis y llenura posprandial de tres semanas de evolucion.",
]
EXAMENES = [
    "Paciente alerta, orientado, hidratado, afebril al momento del examen. Ruidos cardiacos "
    "ritmicos sin soplos. Murmullo vesicular conservado sin agregados. Abdomen blando, "
    "depresible, no doloroso a la palpacion, sin masas ni megalias.",
    "Mucosa oral humeda, faringe eritematosa sin exudados. Cuello movil sin adenopatias "
    "palpables. Neurologico sin signos de focalizacion. Extremidades simetricas con llenado "
    "capilar menor a dos segundos.",
    "Se observa limitacion dolorosa de la flexion del tronco. Maniobra de Lasegue negativa "
    "bilateral. Fuerza muscular conservada 5/5 en miembros inferiores. Reflejos osteotendinosos "
    "normoactivos y simetricos.",
]
PLANES = [
    "Se indica manejo sintomatico ambulatorio con acetaminofen 500 mg cada 8 horas por 3 dias, "
    "abundante hidratacion oral y signos de alarma explicados al paciente. Control en 72 horas.",
    "Se solicitan laboratorios de control: hemograma, glicemia en ayunas, perfil lipidico y "
    "creatinina. Se refuerza educacion en estilo de vida saludable y adherencia farmacologica.",
    "Se formula terapia fisica por diez sesiones, analgesia con antiinflamatorio no esteroideo "
    "por cinco dias e incapacidad laboral por dos dias. Se dan recomendaciones posturales.",
]
EPS = ["EPS VIDA ANDINA", "NUEVA SALUD EPS", "CUIDADO PLENO EPS", "ARRAYAN EPS", "SANAVIDA EPS"]
IPS = ["IPS CENTRO MEDICO LOS ALAMOS", "CLINICA SAN FRANCISCO DE ASIS",
       "UNIDAD HOSPITALARIA SANTA MARIA", "CENTRO DE ATENCION PRIMARIA EL POBLADO"]

# oficios administrativos
PARRAFOS_OFICIO = [
    "Por medio de la presente nos permitimos dar respuesta a la comunicacion radicada en "
    "nuestras oficinas, en la cual se solicita informacion sobre el estado del proceso "
    "contractual adelantado por esta entidad. Al respecto, le informamos que el expediente se "
    "encuentra en etapa de evaluacion tecnica y juridica por parte del comite designado.",
    "En atencion a lo dispuesto en el articulo 14 de la Ley 1755 de 2015, se remite la "
    "documentacion soporte requerida, la cual consta de los anexos relacionados al final de "
    "este oficio. Cualquier aclaracion adicional podra ser tramitada a traves del canal de "
    "correspondencia institucional habilitado para tal fin.",
    "Agradecemos su atencion y reiteramos nuestra disposicion para continuar fortaleciendo los "
    "canales de comunicacion entre las dos organizaciones, en el marco de los principios de "
    "transparencia, celeridad y economia que orientan la funcion administrativa.",
    "Adicionalmente, se le comunica que el comite de convivencia sesionara el proximo mes con "
    "el fin de revisar las observaciones presentadas por las partes interesadas, razon por la "
    "cual se extiende cordial invitacion a la delegacion de su dependencia.",
    "Finalmente, le solicitamos actualizar los datos de contacto del representante legal en la "
    "plataforma dispuesta por la entidad, con el proposito de garantizar la correcta "
    "notificacion de los actos administrativos que se expidan en el curso de la actuacion.",
]
ASUNTOS_OFICIO = [
    "Respuesta a solicitud de informacion contractual",
    "Remision de documentacion soporte del proceso",
    "Citacion a comite tecnico interinstitucional",
    "Actualizacion de datos del representante legal",
    "Notificacion de acto administrativo de tramite",
]

# articulos cientificos
TITULOS_ART = [
    "Evaluacion de representaciones vectoriales multimodales para la clasificacion automatica "
    "de documentos administrativos",
    "Un enfoque hibrido de aprendizaje profundo para la extraccion de informacion en historias "
    "clinicas electronicas",
    "Analisis comparativo de metodos de agrupamiento no supervisado sobre corpus documentales "
    "heterogeneos",
    "Impacto de la estructura visual en la recuperacion semantica de documentos digitalizados",
]
SECCIONES_ART = [
    ("1. Introduccion",
     "La digitalizacion masiva de acervos documentales ha incrementado la necesidad de metodos "
     "automaticos capaces de organizar grandes volumenes de archivos heterogeneos. Los enfoques "
     "clasicos basados exclusivamente en la senal textual ignoran informacion estructural "
     "relevante, lo cual degrada su desempeno cuando el vocabulario se solapa entre categorias."),
    ("2. Trabajos relacionados",
     "Diversos autores han abordado el problema mediante modelos de bolsa de palabras y "
     "representaciones densas obtenidas con transformadores. Trabajos mas recientes incorporan "
     "senales de posicion y modalidad visual, reportando mejoras consistentes en tareas de "
     "clasificacion y recuperacion sobre corpus de facturas y formularios."),
    ("3. Metodologia",
     "Se construyo un corpus controlado en el cual se manipularon de forma independiente el "
     "contenido lexico y la organizacion espacial de cada pagina. Para cada documento se "
     "calcularon representaciones textuales, visuales y una fusion normalizada de ambas, "
     "evaluando el agrupamiento resultante con indices de concordancia estandar."),
    ("4. Resultados y discusion",
     "Los resultados evidencian que la modalidad textual separa correctamente los dominios "
     "tematicos pero colapsa categorias con vocabulario compartido, mientras que la modalidad "
     "visual exhibe el comportamiento complementario. La combinacion de ambas senales alcanza "
     "el mayor indice de concordancia ajustado sobre el conjunto completo."),
    ("5. Conclusiones",
     "El estudio confirma que la estructura visual aporta informacion no redundante frente al "
     "contenido textual. Como trabajo futuro se propone incorporar modelos de codificacion "
     "conjunta entrenados especificamente sobre paginas de documentos."),
]
RELLENO_ART = [
    "La evaluacion se realizo sobre particiones independientes con el fin de controlar el efecto "
    "del azar en la asignacion inicial de los centroides.",
    "Se reporta ademas el indice de informacion mutua normalizada, que resulta menos sensible al "
    "numero de grupos obtenidos por cada configuracion.",
    "Cabe senalar que la resolucion de entrada del codificador visual impide la lectura del texto "
    "impreso, de modo que la senal aprovechada corresponde unicamente a la organizacion espacial.",
    "Los experimentos se ejecutaron sobre una unidad de procesamiento grafico de gama media, con "
    "tiempos de inferencia inferiores a un segundo por pagina.",
    "Trabajos previos sugieren que la normalizacion por bloques evita que una modalidad de mayor "
    "dimensionalidad domine la medida de similitud resultante.",
    "El corpus utilizado es de naturaleza sintetica y controlada, lo cual permite aislar el efecto "
    "de cada factor pero limita la extrapolacion directa a colecciones reales.",
    "Se observa una degradacion sistematica del desempeno textual cuando dos categorias comparten "
    "mas del sesenta por ciento de sus terminos caracteristicos.",
    "La matriz de contingencia permite identificar con precision cuales categorias son fusionadas "
    "de manera erronea por cada representacion evaluada.",
]

REFERENCIAS = [
    "Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. NAACL.",
    "Radford, A., et al. (2021). Learning transferable visual models from natural language. ICML.",
    "Xu, Y., et al. (2020). LayoutLM: Pre-training of text and layout. KDD.",
    "Faysse, M., et al. (2024). ColPali: Efficient document retrieval with vision models. arXiv.",
    "Harley, A. W., et al. (2015). Evaluation of deep CNN architectures for document image "
    "classification. ICDAR.",
]


# --------------------------------------------------------------------------
# plantillas de documento
# --------------------------------------------------------------------------
def _pie_dian(c):
    t(c, M, 38, "Resolucion DIAN 18764004512345 del 14/02/2026 - Rango FV-1 al FV-50000 - "
                "Vigencia 24 meses", "Helvetica", 6.2, colors.HexColor("#777777"))


def _items_factura(r, n):
    sel = r.sample(ITEMS, n)
    filas = []
    for cod, desc, vu in sel:
        cant = r.choice([1, 1, 2, 3, 4, 5, 8, 10, 12])
        filas.append((cod, desc, cant, vu, cant * vu))
    return filas


# --------------------------------------------------------------------------
# TRAMPA 1: 'factura' y 'factura_narrativa' se generan a partir de LOS MISMOS
# datos y comparten hasta las etiquetas de campo ("codigo", "descripcion",
# "cantidad", "valor unitario", "subtotal", "IVA", "total a pagar"...).
# Como bolsa de palabras son casi indistinguibles; solo el layout las separa.
# --------------------------------------------------------------------------
def datos_factura(r):
    emp = r.choice(EMPRESAS)
    cli = r.choice(CLIENTES)
    filas = _items_factura(r, r.randint(5, 8))
    sub = sum(f[4] for f in filas)
    iva = round(sub * 0.19)
    ret = round(sub * 0.025) if r.random() < 0.4 else 0
    return {
        "emp": emp, "cli": cli, "filas": filas, "sub": sub, "iva": iva, "ret": ret,
        "total": sub + iva - ret,
        "num": r.randint(1000, 9999),
        "dia": r.randint(1, 28), "mes": r.choice(MESES),
        "fecha": f"{r.randint(1,28):02d}/{r.randint(1,12):02d}/2026",
        "pago": r.choice(["Credito 30 dias", "Contado", "Credito 60 dias"]),
        "ciudad": r.choice(CIUDADES),
        "direccion": f"Cl. {r.randint(1,90)} # {r.randint(1,99)}-{r.randint(1,99)}",
        "telefono": f"(60) {r.randint(4,6)} {r.randint(2000000,8999999)}",
        "vendedor": r.choice(NOMBRES),
        "banco": r.choice(BANCOS)[0],
        "cuenta": str(r.randint(10000000000, 99999999999)),
        "firma": r.choice(NOMBRES),
    }


def texto_observaciones(d):
    """Bloque de texto identico en las dos plantillas."""
    return (f"El pago debe realizarse a la cuenta de ahorros No. {d['cuenta']} del banco "
            f"{d['banco']}. Esta factura de venta se asimila en todos sus efectos legales a una "
            "letra de cambio segun el articulo 774 del Codigo de Comercio. Pasados treinta dias "
            "se causaran intereses de mora a la tasa maxima legal vigente certificada por la "
            "Superintendencia Financiera de Colombia.")


def texto_items(d):
    """Los mismos items de la tabla, escritos con las mismas etiquetas de columna."""
    return " ".join(
        f"Codigo {cod}, descripcion {desc}, cantidad {cant}, valor unitario {cop(vu)}, "
        f"valor total {cop(vt)}."
        for cod, desc, cant, vu, vt in d["filas"])


def texto_totales(d):
    partes = [f"Subtotal {cop(d['sub'])}", f"IVA 19% {cop(d['iva'])}"]
    if d["ret"]:
        partes.append(f"Retencion en la fuente 2.5% {cop(d['ret'])}")
    partes.append(f"Total a pagar {cop(d['total'])}")
    return ". ".join(partes) + "."


def texto_adquiriente(d):
    return (f"Datos del adquiriente: razon social {d['cli'][0]}, NIT / CC {d['cli'][1]}, "
            f"ciudad {d['ciudad']}, direccion {d['direccion']}, telefono {d['telefono']}, "
            f"vendedor {d['vendedor']}.")


def doc_factura(c, r):
    d = datos_factura(r)
    emp, cli = d["emp"], d["cli"]
    y = H - M

    caja(c, M, y - 58, 58, 58, relleno=GRIS_SUAVE)
    tc(c, M + 29, y - 34, "".join(p[0] for p in emp[0].split()[:3]), "Helvetica-Bold", 16, AZUL)
    t(c, M + 70, y - 14, emp[0], "Helvetica-Bold", 12, AZUL)
    t(c, M + 70, y - 27, f"NIT {emp[1]} - Regimen comun", "Helvetica", 8)
    t(c, M + 70, y - 38, emp[2], "Helvetica", 8)
    t(c, M + 70, y - 49, "facturacion@empresa.com.co", "Helvetica", 8)

    bx, bw = W - M - 185, 185
    caja(c, bx, y - 58, bw, 58, relleno=colors.white)
    caja(c, bx, y - 20, bw, 20, relleno=GRIS)
    tc(c, bx + bw / 2, y - 14, "FACTURA DE VENTA", "Helvetica-Bold", 10)
    t(c, bx + 6, y - 33, "No.", "Helvetica", 8)
    tr(c, bx + bw - 6, y - 33, f"FV-{d['num']}", "Helvetica-Bold", 10, colors.HexColor("#B00000"))
    t(c, bx + 6, y - 45, "Fecha de expedicion", "Helvetica", 7.5)
    tr(c, bx + bw - 6, y - 45, d["fecha"], "Helvetica-Bold", 8)
    t(c, bx + 6, y - 55, "Forma de pago", "Helvetica", 7.5)
    tr(c, bx + bw - 6, y - 55, d["pago"], "Helvetica-Bold", 8)

    y -= 74
    caja(c, M, y - 48, W - 2 * M, 48, relleno=colors.white)
    caja(c, M, y - 14, W - 2 * M, 14, relleno=GRIS_SUAVE)
    t(c, M + 5, y - 10.5, "DATOS DEL ADQUIRIENTE", "Helvetica-Bold", 7.5)
    campo(c, M + 8, y - 24, "Razon social", cli[0])
    campo(c, M + 250, y - 24, "NIT / CC", cli[1])
    campo(c, M + 390, y - 24, "Ciudad", d["ciudad"])
    campo(c, M + 8, y - 42, "Direccion", d["direccion"])
    campo(c, M + 250, y - 42, "Telefono", d["telefono"])
    campo(c, M + 390, y - 42, "Vendedor", d["vendedor"].split()[0] + " " + d["vendedor"].split()[-1])

    y -= 66
    cols = [M, M + 60, M + 300, M + 345, M + 440, W - M]
    encabezados = ["CODIGO", "DESCRIPCION", "CANTIDAD", "VALOR UNITARIO", "VALOR TOTAL"]
    alto = 15
    caja(c, M, y - alto, W - 2 * M, alto, relleno=GRIS)
    for i, h in enumerate(encabezados):
        if i >= 2:
            tr(c, cols[i + 1] - 4, y - alto + 4.5, h, "Helvetica-Bold", 7.5)
        else:
            t(c, cols[i] + 4, y - alto + 4.5, h, "Helvetica-Bold", 7.5)
    y -= alto
    for cod, desc, cant, vu, vt in d["filas"]:
        caja(c, M, y - alto, W - 2 * M, alto, relleno=colors.white)
        for cx in cols[1:-1]:
            c.setStrokeColor(LINEA)
            c.line(cx, y, cx, y - alto)
        t(c, cols[0] + 4, y - alto + 4.5, cod, "Helvetica", 7.5)
        t(c, cols[1] + 4, y - alto + 4.5, desc[:52], "Helvetica", 7.5)
        tr(c, cols[3] - 4, y - alto + 4.5, cant, "Helvetica", 7.5)
        tr(c, cols[4] - 4, y - alto + 4.5, cop(vu), "Helvetica", 7.5)
        tr(c, cols[5] - 4, y - alto + 4.5, cop(vt), "Helvetica", 7.5)
        y -= alto

    y -= 12
    tx = W - M - 200
    totales = [("Subtotal", d["sub"], False), ("IVA 19%", d["iva"], False)]
    if d["ret"]:
        totales.append(("Retencion en la fuente 2.5%", d["ret"], False))
    totales.append(("TOTAL A PAGAR", d["total"], True))
    for etq, val, bold in totales:
        f = "Helvetica-Bold" if bold else "Helvetica"
        if bold:
            caja(c, tx, y - 16, 200, 16, relleno=GRIS)
        t(c, tx + 5, y - 11.5, etq, f, 8.5)
        tr(c, W - M - 5, y - 11.5, cop(val), f, 8.5)
        y -= 16 if bold else 13

    y -= 18
    t(c, M, y, "OBSERVACIONES", "Helvetica-Bold", 7.5)
    y = parrafo(c, M, y - 12, texto_observaciones(d), W - 2 * M, size=7.5, leading=10)
    y = parrafo(c, M, y - 4, texto_adquiriente(d), W - 2 * M, size=7.5, leading=10)
    _pie_dian(c)


def doc_factura_narrativa(c, r):
    """TRAMPA 1: los mismos datos y el mismo vocabulario que 'factura',
    presentados como carta en prosa a una columna."""
    d = datos_factura(r)
    emp, cli = d["emp"], d["cli"]
    y = H - M

    tc(c, W / 2, y - 12, emp[0], "Helvetica-Bold", 11, AZUL)
    tc(c, W / 2, y - 24, f"NIT {emp[1]} - Regimen comun", "Helvetica", 8.5)
    tc(c, W / 2, y - 35, emp[2], "Helvetica", 8.5)
    tc(c, W / 2, y - 46, "facturacion@empresa.com.co", "Helvetica", 8.5)
    linea_h(c, M, y - 54, W - M)

    y -= 78
    tr(c, W - M, y, f"{d['ciudad']}, {d['dia']} de {d['mes']} de 2026", "Helvetica", 9)
    y -= 30
    y = ln(c, M, y, "Senores", "Helvetica-Bold", 9.5, 13)
    y = ln(c, M, y, cli[0], "Helvetica-Bold", 9.5, 13)
    y = ln(c, M, y, f"NIT / CC {cli[1]}", "Helvetica", 9.5, 13)
    y = ln(c, M, y, d["ciudad"], "Helvetica", 9.5, 13)
    y -= 14
    t(c, M, y, f"Asunto: factura de venta No. FV-{d['num']}", "Helvetica-Bold", 9.5)
    y -= 24

    parrafos = [
        f"Factura de venta No. FV-{d['num']}. Fecha de expedicion {d['fecha']}. "
        f"Forma de pago {d['pago']}. {texto_adquiriente(d)}",
        "Detalle de los items facturados. " + texto_items(d),
        texto_totales(d),
        "Observaciones. " + texto_observaciones(d),
    ]
    for p_ in parrafos:
        y = parrafo(c, M, y, p_, W - 2 * M, size=9.3, leading=13.2)
        y -= 10

    y -= 16
    y = ln(c, M, y, "Cordialmente,", "Helvetica", 9.5, 13)
    y -= 40
    linea_h(c, M, y, M + 190)
    t(c, M, y - 12, d["firma"], "Helvetica-Bold", 9)
    t(c, M, y - 23, "Coordinador de Cartera y Facturacion", "Helvetica", 8.5)
    _pie_dian(c)


def doc_extracto_bancario(c, r):
    banco, cod = r.choice(BANCOS)
    y = H - M
    # banda superior de color
    caja(c, 0, y - 42, W, 42, relleno=AZUL, borde=AZUL)
    t(c, M, y - 20, banco, "Helvetica-Bold", 14, colors.white)
    t(c, M, y - 34, "EXTRACTO DE CUENTA DE AHORROS", "Helvetica", 9.5, colors.white)
    tr(c, W - M, y - 20, f"Codigo entidad {cod}", "Helvetica", 8.5, colors.white)
    tr(c, W - M, y - 33, f"Periodo: 01/{r.randint(1,9):02d}/2026 al 30/{r.randint(1,9):02d}/2026",
       "Helvetica", 8.5, colors.white)

    y -= 58
    cli = r.choice(CLIENTES)
    campo(c, M, y, "Titular", cli[0])
    campo(c, M + 250, y, "Identificacion", cli[1])
    campo(c, M + 390, y, "No. de cuenta", str(r.randint(10000000000, 99999999999)))

    # resumen en 4 celdas
    y -= 34
    saldo_ini = r.randint(500000, 20000000)
    total_deb = r.randint(300000, 9000000)
    total_cre = r.randint(300000, 12000000)
    saldo_fin = saldo_ini - total_deb + total_cre
    cw = (W - 2 * M) / 4
    for i, (etq, val) in enumerate([("SALDO INICIAL", saldo_ini), ("TOTAL DEBITOS", total_deb),
                                    ("TOTAL CREDITOS", total_cre), ("SALDO FINAL", saldo_fin)]):
        caja(c, M + i * cw, y - 30, cw, 30, relleno=GRIS_SUAVE)
        tc(c, M + i * cw + cw / 2, y - 11, etq, "Helvetica", 6.8, colors.HexColor("#555555"))
        tc(c, M + i * cw + cw / 2, y - 24, cop(val), "Helvetica-Bold", 10)

    # tabla densa de movimientos (sin bordes verticales, estilo bancario)
    y -= 48
    t(c, M, y, "DETALLE DE MOVIMIENTOS DEL PERIODO", "Helvetica-Bold", 8.5)
    y -= 14
    alto = 13.5
    caja(c, M, y - alto, W - 2 * M, alto, relleno=AZUL, borde=AZUL)
    t(c, M + 4, y - alto + 4, "FECHA", "Helvetica-Bold", 7, colors.white)
    t(c, M + 58, y - alto + 4, "DOCUMENTO", "Helvetica-Bold", 7, colors.white)
    t(c, M + 120, y - alto + 4, "DESCRIPCION DE LA TRANSACCION", "Helvetica-Bold", 7, colors.white)
    tr(c, M + 385, y - alto + 4, "DEBITO", "Helvetica-Bold", 7, colors.white)
    tr(c, M + 440, y - alto + 4, "CREDITO", "Helvetica-Bold", 7, colors.white)
    tr(c, W - M - 4, y - alto + 4, "SALDO", "Helvetica-Bold", 7, colors.white)
    y -= alto
    saldo = saldo_ini
    for i in range(r.randint(20, 26)):
        if i % 2 == 0:
            caja(c, M, y - alto, W - 2 * M, alto, relleno=GRIS_SUAVE, borde=GRIS_SUAVE)
        es_deb = r.random() < 0.6
        val = r.randint(12000, 2500000)
        saldo += -val if es_deb else val
        t(c, M + 4, y - alto + 4, f"{r.randint(1,28):02d}/{r.randint(1,9):02d}/26", "Helvetica", 7)
        t(c, M + 58, y - alto + 4, str(r.randint(100000, 999999)), "Helvetica", 7)
        t(c, M + 120, y - alto + 4, r.choice(MOVIMIENTOS)[:44], "Helvetica", 7)
        if es_deb:
            tr(c, M + 385, y - alto + 4, cop(val), "Helvetica", 7)
        else:
            tr(c, M + 440, y - alto + 4, cop(val), "Helvetica", 7)
        tr(c, W - M - 4, y - alto + 4, cop(saldo), "Helvetica", 7)
        y -= alto
    linea_h(c, M, y, W - M, 1.0)
    y -= 20
    parrafo(c, M, y,
            "Senor cliente: verifique su extracto y reporte cualquier inconsistencia dentro de "
            "los quince dias siguientes a la fecha de corte. Los saldos expresados corresponden "
            "a pesos colombianos. Este documento es de caracter informativo y no constituye "
            "titulo valor. Consulte la tasa de interes vigente en nuestros canales virtuales.",
            W - 2 * M, size=7, leading=9.5)


def _seccion_form(c, y, titulo, alto):
    caja(c, M, y - alto, W - 2 * M, alto, relleno=colors.white)
    caja(c, M, y - 13, W - 2 * M, 13, relleno=GRIS)
    t(c, M + 5, y - 9.5, titulo, "Helvetica-Bold", 7.5)
    return y - alto


def doc_historia_clinica(c, r):
    """TRAMPA 2a: geometria de formulario casi identica a 'formulario_afiliacion'."""
    ips = r.choice(IPS)
    nom = r.choice(NOMBRES)
    dx = r.choice(DIAGNOSTICOS)
    y = H - M
    tc(c, W / 2, y - 12, ips, "Helvetica-Bold", 11.5, AZUL)
    tc(c, W / 2, y - 24, f"NIT {r.randint(800,901)}.{r.randint(100,999)}.{r.randint(100,999)}-{r.randint(1,9)}"
                         f"  -  Habilitacion REPS {r.randint(100000,999999)}", "Helvetica", 7.5)
    y -= 34
    caja(c, M, y - 18, W - 2 * M, 18, relleno=GRIS)
    tc(c, W / 2, y - 13, "HISTORIA CLINICA - NOTA DE CONSULTA EXTERNA", "Helvetica-Bold", 10)
    y -= 26

    y = _seccion_form(c, y, "1. IDENTIFICACION DEL PACIENTE", 72)
    yy = y + 72 - 22
    campo(c, M + 8, yy, "Primer nombre y apellidos", nom)
    campo(c, M + 250, yy, "Tipo y numero de documento", f"CC {r.randint(10000000, 99999999)}")
    campo(c, M + 400, yy, "Fecha de nacimiento", f"{r.randint(1,28):02d}/{r.randint(1,12):02d}/{r.randint(1955,2005)}")
    yy -= 24
    campo(c, M + 8, yy, "Sexo", r.choice(["Masculino", "Femenino"]))
    campo(c, M + 100, yy, "Edad", f"{r.randint(18,85)} anos")
    campo(c, M + 170, yy, "Entidad responsable de pago", r.choice(EPS))
    campo(c, M + 340, yy, "Regimen", r.choice(["Contributivo", "Subsidiado"]))
    campo(c, M + 440, yy, "Nivel", str(r.randint(1, 3)))
    yy -= 24
    campo(c, M + 8, yy, "Direccion de residencia", f"Cl. {r.randint(1,90)} # {r.randint(1,99)}-{r.randint(1,99)}")
    campo(c, M + 250, yy, "Telefono de contacto", f"3{r.randint(10,25)} {r.randint(1000000,9999999)}")
    campo(c, M + 400, yy, "Acompanante", r.choice(NOMBRES).split()[0])
    y -= 8

    y = _seccion_form(c, y, "2. SIGNOS VITALES Y ANTROPOMETRIA", 42)
    cw = (W - 2 * M) / 6
    signos = [("T.A. (mmHg)", f"{r.randint(100,145)}/{r.randint(60,95)}"),
              ("F.C. (lpm)", str(r.randint(58, 98))),
              ("F.R. (rpm)", str(r.randint(14, 22))),
              ("Temp. (C)", f"{r.uniform(36.0, 38.4):.1f}"),
              ("Peso (kg)", f"{r.uniform(52, 98):.1f}"),
              ("Talla (cm)", str(r.randint(150, 188)))]
    for i, (etq, val) in enumerate(signos):
        cx = M + i * cw
        if i:
            c.setStrokeColor(LINEA); c.line(cx, y, cx, y + 29)
        tc(c, cx + cw / 2, y + 19, etq, "Helvetica", 6.5, colors.HexColor("#555555"))
        tc(c, cx + cw / 2, y + 7, val, "Helvetica-Bold", 9.5)
    y -= 8

    for titulo, texto, alto in [
        ("3. MOTIVO DE CONSULTA Y ENFERMEDAD ACTUAL", r.choice(MOTIVOS), 50),
        ("4. ANTECEDENTES PERSONALES Y FAMILIARES",
         "Antecedentes patologicos: " + r.choice(["niega", "hipertension arterial en manejo",
          "diabetes mellitus tipo 2", "asma en la infancia"]) +
         ". Quirurgicos: " + r.choice(["apendicectomia", "niega", "cesarea"]) +
         ". Alergicos: " + r.choice(["niega", "penicilina", "dipirona"]) +
         ". Farmacologicos: " + r.choice(["losartan 50 mg dia", "metformina 850 mg dia", "ninguno"]) +
         ". Toxicologicos: " + r.choice(["fumador activo", "niega", "consumo social de alcohol"]) + ".", 44),
        ("5. EXAMEN FISICO", r.choice(EXAMENES), 56),
    ]:
        y = _seccion_form(c, y, titulo, alto)
        parrafo(c, M + 8, y + alto - 26, texto, W - 2 * M - 16, size=7.8, leading=10)
        y -= 8

    y = _seccion_form(c, y, "6. DIAGNOSTICOS (CIE-10)", 34)
    t(c, M + 8, y + 13, f"Principal:  {dx[0]}  -  {dx[1]}", "Helvetica-Bold", 8)
    dx2 = r.choice(DIAGNOSTICOS)
    t(c, M + 300, y + 13, f"Relacionado:  {dx2[0]}  -  {dx2[1][:34]}", "Helvetica", 7.8)
    y -= 8

    y = _seccion_form(c, y, "7. PLAN DE MANEJO Y CONDUCTA", 50)
    parrafo(c, M + 8, y + 24, r.choice(PLANES), W - 2 * M - 16, size=7.8, leading=10)
    y -= 22

    linea_h(c, M + 20, y, M + 210)
    t(c, M + 20, y - 11, r.choice(NOMBRES), "Helvetica-Bold", 8)
    t(c, M + 20, y - 21, f"Medico general - Reg. {r.randint(10000,99999)}", "Helvetica", 7.5)
    linea_h(c, W - M - 210, y, W - M - 20)
    t(c, W - M - 210, y - 11, "Firma del paciente o acudiente", "Helvetica", 7.5)


def doc_formulario_afiliacion(c, r):
    """TRAMPA 2b: misma geometria que 'historia_clinica', dominio lexico administrativo."""
    ent = r.choice(EPS)
    nom = r.choice(NOMBRES)
    y = H - M
    tc(c, W / 2, y - 12, ent, "Helvetica-Bold", 11.5, AZUL)
    tc(c, W / 2, y - 24, f"NIT {r.randint(800,901)}.{r.randint(100,999)}.{r.randint(100,999)}-{r.randint(1,9)}"
                         f"  -  Vigilado por la Superintendencia Nacional de Salud", "Helvetica", 7.5)
    y -= 34
    caja(c, M, y - 18, W - 2 * M, 18, relleno=GRIS)
    tc(c, W / 2, y - 13, "FORMULARIO UNICO DE AFILIACION Y REGISTRO DE NOVEDADES",
       "Helvetica-Bold", 10)
    y -= 26

    # fila de casillas de verificacion
    caja(c, M, y - 20, W - 2 * M, 20, relleno=colors.white)
    t(c, M + 5, y - 13.5, "TIPO DE SOLICITUD:", "Helvetica-Bold", 7.5)
    opciones = ["Afiliacion inicial", "Traslado de EPS", "Actualizacion de datos",
                "Inclusion de beneficiario", "Retiro"]
    marcada = r.randrange(len(opciones))
    cx = M + 105
    for i, op in enumerate(opciones):
        caja(c, cx, y - 15, 9, 9, relleno=colors.white, grosor=0.8)
        if i == marcada:
            t(c, cx + 1.4, y - 13.5, "X", "Helvetica-Bold", 7.5)
        t(c, cx + 13, y - 13.5, op, "Helvetica", 7.3)
        cx += 18 + len(op) * 3.7
    y -= 28

    y = _seccion_form(c, y, "1. DATOS BASICOS DEL COTIZANTE", 72)
    yy = y + 72 - 22
    campo(c, M + 8, yy, "Primer nombre y apellidos", nom)
    campo(c, M + 250, yy, "Tipo y numero de documento", f"CC {r.randint(10000000, 99999999)}")
    campo(c, M + 400, yy, "Fecha de nacimiento", f"{r.randint(1,28):02d}/{r.randint(1,12):02d}/{r.randint(1955,2005)}")
    yy -= 24
    campo(c, M + 8, yy, "Sexo", r.choice(["Masculino", "Femenino"]))
    campo(c, M + 100, yy, "Estado civil", r.choice(["Soltero", "Casado", "Union libre"]))
    campo(c, M + 190, yy, "Nivel educativo", r.choice(["Tecnico", "Profesional", "Bachiller"]))
    campo(c, M + 320, yy, "Ocupacion", r.choice(CARGOS)[:22])
    campo(c, M + 440, yy, "Sisben", r.choice(["A1", "B4", "C2", "No aplica"]))
    yy -= 24
    campo(c, M + 8, yy, "Direccion de residencia", f"Cra. {r.randint(1,90)} # {r.randint(1,99)}-{r.randint(1,99)}")
    campo(c, M + 250, yy, "Telefono de contacto", f"3{r.randint(10,25)} {r.randint(1000000,9999999)}")
    campo(c, M + 400, yy, "Correo electronico", "usuario" + str(r.randint(10, 99)) + "@correo.com")
    y -= 8

    y = _seccion_form(c, y, "2. REGIMEN Y CONDICION DE AFILIACION", 42)
    cw = (W - 2 * M) / 6
    celdas = [("Regimen", r.choice(["Contributivo", "Subsidiado"])),
              ("Tipo cotizante", r.choice(["Dependiente", "Independiente"])),
              ("Categoria IBC", r.choice(["A", "B", "C"])),
              ("Salario base", cop(r.randint(1400000, 9000000))),
              ("Fecha ingreso", f"{r.randint(1,28):02d}/{r.randint(1,12):02d}/2026"),
              ("Grupo poblacional", r.choice(["Ninguno", "Victima", "Migrante"]))]
    for i, (etq, val) in enumerate(celdas):
        cxx = M + i * cw
        if i:
            c.setStrokeColor(LINEA); c.line(cxx, y, cxx, y + 29)
        tc(c, cxx + cw / 2, y + 19, etq, "Helvetica", 6.5, colors.HexColor("#555555"))
        tc(c, cxx + cw / 2, y + 7, val, "Helvetica-Bold", 8)
    y -= 8

    y = _seccion_form(c, y, "3. DATOS DEL EMPLEADOR O APORTANTE", 44)
    emp = r.choice(EMPRESAS)
    yy = y + 44 - 24
    campo(c, M + 8, yy, "Razon social del aportante", emp[0])
    campo(c, M + 290, yy, "NIT", emp[1])
    campo(c, M + 400, yy, "Actividad economica CIIU", str(r.randint(1000, 9999)))
    yy -= 20
    campo(c, M + 8, yy, "Direccion del aportante", emp[2])
    campo(c, M + 290, yy, "Ciudad", r.choice(CIUDADES))
    campo(c, M + 400, yy, "Clase de riesgo ARL", str(r.randint(1, 5)))
    y -= 8

    # tabla de beneficiarios
    y = _seccion_form(c, y, "4. BENEFICIARIOS DEL NUCLEO FAMILIAR", 74)
    cols = [M, M + 210, M + 300, M + 390, W - M]
    alto = 13
    yt = y + 74 - 13
    caja(c, M, yt - alto, W - 2 * M, alto, relleno=GRIS_SUAVE)
    for i, h in enumerate(["NOMBRE COMPLETO", "DOCUMENTO", "PARENTESCO", "FECHA NACIMIENTO"]):
        t(c, cols[i] + 4, yt - alto + 4, h, "Helvetica-Bold", 6.8)
    yt -= alto
    for _ in range(r.randint(2, 4)):
        for cxx in cols[1:-1]:
            c.setStrokeColor(LINEA); c.line(cxx, yt, cxx, yt - alto)
        linea_h(c, M, yt - alto, W - M, 0.4)
        t(c, cols[0] + 4, yt - alto + 4, r.choice(NOMBRES), "Helvetica", 7)
        t(c, cols[1] + 4, yt - alto + 4, f"CC {r.randint(10000000,99999999)}", "Helvetica", 7)
        t(c, cols[2] + 4, yt - alto + 4, r.choice(["Conyuge", "Hijo(a)", "Padre", "Madre"]), "Helvetica", 7)
        t(c, cols[3] + 4, yt - alto + 4, f"{r.randint(1,28):02d}/{r.randint(1,12):02d}/{r.randint(1990,2022)}", "Helvetica", 7)
        yt -= alto
    y -= 8

    y = _seccion_form(c, y, "5. DECLARACION Y AUTORIZACION DEL AFILIADO", 46)
    parrafo(c, M + 8, y + 22,
            "Declaro que la informacion consignada en este formulario es veraz y autorizo su "
            "verificacion. Autorizo el tratamiento de mis datos personales conforme a la Ley "
            "1581 de 2012 y manifiesto conocer los deberes y derechos como afiliado al Sistema "
            "General de Seguridad Social en Salud.", W - 2 * M - 16, size=7.6, leading=10)
    y -= 24

    linea_h(c, M + 20, y, M + 210)
    t(c, M + 20, y - 11, "Firma del cotizante", "Helvetica", 7.5)
    linea_h(c, M + 250, y, M + 400)
    t(c, M + 250, y - 11, "Firma del aportante", "Helvetica", 7.5)
    caja(c, W - M - 70, y - 20, 70, 52, relleno=colors.white)
    tc(c, W - M - 35, y - 12, "HUELLA", "Helvetica", 6.5)


def doc_oficio(c, r):
    ent = r.choice([e[0] for e in EMPRESAS] + ["SECRETARIA DE HACIENDA MUNICIPAL",
                                               "INSTITUTO DEPARTAMENTAL DE SALUD"])
    y = H - M
    tc(c, W / 2, y - 12, ent, "Helvetica-Bold", 11, AZUL)
    tc(c, W / 2, y - 24, f"NIT {r.randint(800,901)}.{r.randint(100,999)}.{r.randint(100,999)}-{r.randint(1,9)}",
       "Helvetica", 8.5)
    tc(c, W / 2, y - 35, r.choice(EMPRESAS)[2], "Helvetica", 8.5)
    tc(c, W / 2, y - 46, "Direccion Administrativa y Financiera", "Helvetica", 8.5)
    linea_h(c, M, y - 54, W - M)

    y -= 78
    t(c, M, y, f"Radicado No. {r.randint(2026000,2026999)}-{r.randint(100,999)}", "Helvetica", 8.5)
    tr(c, W - M, y, f"{r.choice(CIUDADES)}, {r.randint(1,28)} de {r.choice(MESES)} de 2026",
       "Helvetica", 9)
    y -= 34
    y = ln(c, M, y, "Senor(a)", "Helvetica", 9.5, 13)
    y = ln(c, M, y, r.choice(NOMBRES), "Helvetica-Bold", 9.5, 13)
    y = ln(c, M, y, r.choice(CARGOS), "Helvetica", 9.5, 13)
    y = ln(c, M, y, r.choice(CLIENTES)[0], "Helvetica", 9.5, 13)
    y = ln(c, M, y, r.choice(CIUDADES), "Helvetica", 9.5, 13)
    y -= 16
    t(c, M, y, f"Asunto: {r.choice(ASUNTOS_OFICIO)}", "Helvetica-Bold", 9.5)
    y -= 26
    y = parrafo(c, M, y, "Cordial saludo:", ancho=W - 2 * M, size=9.5, leading=13)
    y -= 8
    for p in r.sample(PARRAFOS_OFICIO, r.randint(3, 5)):
        y = parrafo(c, M, y, p, W - 2 * M, size=9.5, leading=13.5)
        y -= 9
    y -= 20
    y = parrafo(c, M, y, "Atentamente,", ancho=W - 2 * M, size=9.5, leading=13)
    y -= 46
    linea_h(c, M, y, M + 190)
    t(c, M, y - 12, r.choice(NOMBRES), "Helvetica-Bold", 9)
    t(c, M, y - 23, r.choice(CARGOS), "Helvetica", 8.5)
    y -= 42
    t(c, M, y, f"Anexos: {r.randint(1,6)} folios", "Helvetica", 7.5)
    t(c, M, y - 10, f"Copia: Archivo central - Proyecto: {r.choice(NOMBRES).split()[0][:1]}."
                    f"{r.choice(NOMBRES).split()[-1]}", "Helvetica", 7.5)


def doc_articulo_cientifico(c, r):
    y = H - M
    titulo = r.choice(TITULOS_ART)
    yy = y - 6
    for ln_ in simpleSplit(titulo, "Helvetica-Bold", 14, W - 2 * M - 60):
        tc(c, W / 2, yy, ln_, "Helvetica-Bold", 14)
        yy -= 17
    yy -= 6
    autores = ", ".join(f"{n.split()[0][0]}. {n.split()[-2]} {n.split()[-1]}"
                        for n in r.sample(NOMBRES, 3))
    tc(c, W / 2, yy, autores, "Helvetica", 9.5); yy -= 12
    tc(c, W / 2, yy, "Departamento de Ingenieria de Sistemas, Universidad del Mirador, Medellin, Colombia",
       "Helvetica-Oblique", 8); yy -= 11
    tc(c, W / 2, yy, f"Recibido: {r.randint(1,28)} de {r.choice(MESES)} de 2026  |  "
                     f"Aceptado: {r.randint(1,28)} de {r.choice(MESES)} de 2026", "Helvetica", 7.5)
    yy -= 22

    t(c, M, yy, "Resumen", "Helvetica-Bold", 9)
    yy = parrafo(c, M, yy - 12,
                 "Este trabajo estudia el aporte de la informacion estructural de la pagina en "
                 "la organizacion automatica de corpus documentales. Se comparan "
                 "representaciones vectoriales de naturaleza textual, visual y multimodal sobre "
                 "un corpus controlado en el que el contenido lexico y la disposicion espacial "
                 "se manipulan de manera independiente. Los resultados muestran que la fusion "
                 "de modalidades supera de forma consistente a cada senal por separado, "
                 "especialmente en categorias con alto solapamiento de vocabulario.",
                 W - 2 * M, size=8.2, leading=10.5)
    yy -= 4
    t(c, M, yy, "Palabras clave: " + ", ".join(r.sample(
        ["embeddings multimodales", "clasificacion documental", "analisis de layout",
         "recuperacion de informacion", "aprendizaje no supervisado", "vision por computador"], 4)),
      "Helvetica-Oblique", 7.8)
    yy -= 14
    linea_h(c, M, yy, W - M)

    # ---- flujo a dos columnas ----
    gap, y_inf = 20, 58.0
    colw = (W - 2 * M - gap) / 2
    xs = [M, M + colw + gap]
    y_sup = yy - 16
    estado = {"col": 0, "y": y_sup}

    def salto(alto_necesario):
        if estado["y"] - alto_necesario < y_inf:
            if estado["col"] == 0:
                estado["col"], estado["y"] = 1, y_sup
                return True
            return False
        return True

    def bloque_texto(texto, font="Helvetica", size=8, leading=10.2):
        for ln_ in simpleSplit(texto, font, size, colw):
            if not salto(leading):
                return False
            t(c, xs[estado["col"]], estado["y"], ln_, font, size)
            estado["y"] -= leading
        return True

    for sec, base in SECCIONES_ART:
        if not salto(24):
            break
        t(c, xs[estado["col"]], estado["y"], sec, "Helvetica-Bold", 9)
        estado["y"] -= 12
        cuerpo = base + " " + " ".join(r.sample(RELLENO_ART, 3))
        if not bloque_texto(cuerpo):
            break
        estado["y"] -= 9

    # figura simulada (ocupa el ancho de una columna)
    if salto(96):
        yb = estado["y"]
        caja(c, xs[estado["col"]], yb - 74, colw, 74, relleno=GRIS_SUAVE)
        for b in range(6):
            hh = 10 + r.random() * 52
            caja(c, xs[estado["col"]] + 14 + b * (colw - 28) / 6, yb - 68,
                 (colw - 34) / 8, hh, relleno=AZUL, borde=AZUL)
        estado["y"] = yb - 82
        bloque_texto("Figura 1. Indice de concordancia ajustado obtenido por cada "
                     "configuracion de representacion evaluada.", size=7, leading=9)
        estado["y"] -= 8

    if salto(20):
        t(c, xs[estado["col"]], estado["y"], "Referencias", "Helvetica-Bold", 8.5)
        estado["y"] -= 11
        for i, ref in enumerate(r.sample(REFERENCIAS, 4), 1):
            if not bloque_texto(f"[{i}] {ref}", size=6.8, leading=8.4):
                break
            estado["y"] -= 2

    tc(c, W / 2, 32, f"Revista de Ingenieria y Ciencia Computacional  -  Vol. {r.randint(10,18)}, "
                     f"No. {r.randint(1,4)}, 2026  -  pp. {r.randint(10,80)}-{r.randint(81,140)}",
       "Helvetica", 6.8, colors.HexColor("#777777"))


def doc_certificado_laboral(c, r):
    emp = r.choice(EMPRESAS)
    nom = r.choice(NOMBRES)
    cargo = r.choice(CARGOS)
    salario = r.randint(2500000, 12000000)
    y = H - M - 20
    tc(c, W / 2, y, emp[0], "Helvetica-Bold", 13, AZUL); y -= 15
    tc(c, W / 2, y, f"NIT {emp[1]}", "Helvetica", 9); y -= 12
    tc(c, W / 2, y, emp[2], "Helvetica", 9); y -= 12
    tc(c, W / 2, y, "Area de Gestion Humana", "Helvetica", 9)
    y -= 60

    tc(c, W / 2, y, "LA EMPRESA CERTIFICA", "Helvetica-Bold", 15)
    linea_h(c, W / 2 - 105, y - 6, W / 2 + 105, 1.1)
    y -= 60

    y = parrafo(c, M + 30, y,
                f"Que el(la) senor(a) {nom}, identificado(a) con cedula de ciudadania numero "
                f"{r.randint(10000000, 99999999)}, labora en nuestra organizacion mediante "
                f"contrato de trabajo a termino {r.choice(['indefinido', 'fijo'])} desde el "
                f"{r.randint(1,28)} de {r.choice(MESES)} de {r.randint(2015, 2025)}, "
                f"desempenando el cargo de {cargo}.",
                W - 2 * M - 60, size=10.5, leading=17)
    y -= 24
    y = parrafo(c, M + 30, y,
                f"Su asignacion salarial mensual actual corresponde a la suma de {cop(salario)} "
                f"({r.choice(['mas', 'sin'])} auxilio de transporte legal), pagadera de forma "
                f"{r.choice(['quincenal', 'mensual'])}. El(la) trabajador(a) se encuentra "
                f"afiliado(a) al Sistema General de Seguridad Social en Salud y Pensiones y a "
                f"la Administradora de Riesgos Laborales correspondiente.",
                W - 2 * M - 60, size=10.5, leading=17)
    y -= 24
    y = parrafo(c, M + 30, y,
                "La presente certificacion se expide a solicitud del interesado(a) para los "
                "fines que estime convenientes y no constituye promesa de continuidad de la "
                "relacion laboral ni compromiso adicional por parte del empleador.",
                W - 2 * M - 60, size=10.5, leading=17)
    y -= 46
    y = parrafo(c, M + 30, y,
                f"Se expide en {r.choice(CIUDADES)}, a los {r.randint(1,28)} dias del mes de "
                f"{r.choice(MESES)} de 2026.", W - 2 * M - 60, size=10.5, leading=17)

    y -= 90
    linea_h(c, W / 2 - 100, y, W / 2 + 100)
    tc(c, W / 2, y - 14, r.choice(NOMBRES), "Helvetica-Bold", 10)
    tc(c, W / 2, y - 27, "Director(a) de Gestion Humana", "Helvetica", 9)
    # sello
    c.setStrokeColor(colors.HexColor("#2E5C9A")); c.setLineWidth(1.6)
    c.circle(M + 62, y - 6, 40, stroke=1, fill=0)
    tc(c, M + 62, y + 2, "GESTION", "Helvetica-Bold", 7.5, colors.HexColor("#2E5C9A"))
    tc(c, M + 62, y - 8, "HUMANA", "Helvetica-Bold", 7.5, colors.HexColor("#2E5C9A"))
    tc(c, M + 62, y - 20, "2026", "Helvetica", 6.5, colors.HexColor("#2E5C9A"))
    tc(c, W / 2, 40, "Verifique la autenticidad de este certificado en nuestro portal de "
                     "empleados.", "Helvetica", 7, colors.HexColor("#777777"))


# --------------------------------------------------------------------------
# catalogo: clase -> (funcion, cantidad, nota didactica)
# --------------------------------------------------------------------------
CATALOGO = {
    "factura":               (doc_factura,              10, "tabla con grilla + totales"),
    "factura_narrativa":     (doc_factura_narrativa,     9, "TRAMPA 1: texto de factura, layout de carta"),
    "extracto_bancario":     (doc_extracto_bancario,     9, "tabla densa de cifras"),
    "historia_clinica":      (doc_historia_clinica,     10, "TRAMPA 2: formulario con cajas (clinico)"),
    "formulario_afiliacion": (doc_formulario_afiliacion, 9, "TRAMPA 2: formulario con cajas (administrativo)"),
    "oficio":                (doc_oficio,               10, "carta a una columna"),
    "articulo_cientifico":   (doc_articulo_cientifico,   8, "dos columnas"),
    "certificado_laboral":   (doc_certificado_laboral,   8, "una columna, mucho espacio en blanco"),
}


def main(semilla: int = 7):
    if DIR_ENTRADA.exists():
        shutil.rmtree(DIR_ENTRADA)
    DIR_ENTRADA.mkdir(parents=True, exist_ok=True)

    tareas = []
    for clase, (fn, n, _) in CATALOGO.items():
        tareas += [(clase, fn, i) for i in range(n)]

    rnd = random.Random(semilla)
    rnd.shuffle(tareas)   # nombres de archivo sin pista de la clase

    filas = []
    for idx, (clase, fn, i) in enumerate(tareas, start=1):
        nombre = f"doc_{idx:03d}.pdf"
        ruta = DIR_ENTRADA / nombre
        c = canvas.Canvas(str(ruta), pagesize=A4)
        c.setTitle(nombre)
        fn(c, random.Random(semilla * 1000 + idx))
        c.showPage()
        c.save()
        filas.append({"archivo": nombre, "clase_real": clase})

    with CSV_ETIQUETAS.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["archivo", "clase_real"])
        wr.writeheader()
        wr.writerows(sorted(filas, key=lambda d: d["archivo"]))

    print(f"Generados {len(filas)} PDFs en {DIR_ENTRADA}")
    for clase, (_, n, nota) in CATALOGO.items():
        print(f"  {clase:24s} {n:3d}  ({nota})")
    print(f"Etiquetas de referencia (solo para evaluar): {CSV_ETIQUETAS}")


if __name__ == "__main__":
    main()
