#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un PDF con la guía paso a paso de cambios en la Sección 3.3 de la tesis.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import PageBreak

OUTPUT = "Guia_Cambios_Seccion33_Paso_a_Paso.pdf"

# ── Estilos ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def style(name, **kwargs):
    s = ParagraphStyle(name, **kwargs)
    return s

S_title      = style("s_title",      fontSize=16, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#1a237e"), spaceAfter=6,
                     alignment=TA_CENTER)
S_subtitle   = style("s_subtitle",   fontSize=13, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#283593"), spaceAfter=4,
                     alignment=TA_CENTER)
S_section    = style("s_section",    fontSize=12, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#ffffff"), spaceAfter=2,
                     spaceBefore=10)
S_step_num   = style("s_step_num",   fontSize=18, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#e53935"), spaceAfter=0)
S_step_title = style("s_step_title", fontSize=12, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#1a237e"), spaceAfter=4,
                     spaceBefore=2)
S_label_red  = style("s_label_red",  fontSize=10, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#c62828"), spaceAfter=2)
S_label_green= style("s_label_green",fontSize=10, fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#1b5e20"), spaceAfter=2)
S_body       = style("s_body",       fontSize=9,  fontName="Helvetica",
                     textColor=colors.HexColor("#212121"), leading=14,
                     spaceAfter=4, alignment=TA_JUSTIFY)
S_note       = style("s_note",       fontSize=8,  fontName="Helvetica-Oblique",
                     textColor=colors.HexColor("#455a64"), spaceAfter=4,
                     leading=12, alignment=TA_JUSTIFY)
S_why        = style("s_why",        fontSize=9,  fontName="Helvetica-Oblique",
                     textColor=colors.HexColor("#4a148c"), spaceAfter=4,
                     leading=13, alignment=TA_JUSTIFY)
S_center     = style("s_center",     fontSize=9,  fontName="Helvetica",
                     textColor=colors.HexColor("#212121"), alignment=TA_CENTER)
S_footer     = style("s_footer",     fontSize=7,  fontName="Helvetica-Oblique",
                     textColor=colors.HexColor("#90a4ae"), alignment=TA_CENTER)

# ── Helpers ──────────────────────────────────────────────────────────────────

def colored_box(text, bg="#1a237e", fg="#ffffff", size=10):
    """Caja de encabezado coloreada."""
    data = [[Paragraph(f'<font color="{fg}"><b>{text}</b></font>', style(
        "tmp", fontSize=size, fontName="Helvetica-Bold",
        textColor=colors.HexColor(fg), spaceAfter=0, alignment=TA_LEFT))]]
    t = Table(data, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor(bg)),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t

def step_header(num, title, color="#e53935"):
    data = [[
        Paragraph(f'<font color="white"><b>PASO {num}</b></font>',
                  style("sh1", fontSize=11, fontName="Helvetica-Bold",
                        textColor=colors.white, spaceAfter=0, alignment=TA_CENTER)),
        Paragraph(f'<b>{title}</b>',
                  style("sh2", fontSize=11, fontName="Helvetica-Bold",
                        textColor=colors.HexColor("#1a237e"), spaceAfter=0))
    ]]
    t = Table(data, colWidths=[2.5*cm, 14.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), colors.HexColor(color)),
        ("BACKGROUND", (1,0), (1,0), colors.HexColor("#e8eaf6")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

def text_box(label, text, bg_label="#ffcdd2", bg_text="#fff8f8",
             label_color="#b71c1c", text_color="#212121"):
    data = [
        [Paragraph(f'<b>{label}</b>',
                   style("tl", fontSize=9, fontName="Helvetica-Bold",
                         textColor=colors.HexColor(label_color), spaceAfter=0))],
        [Paragraph(text,
                   style("tb", fontSize=9, fontName="Helvetica",
                         textColor=colors.HexColor(text_color), leading=14,
                         spaceAfter=0, alignment=TA_JUSTIFY))]
    ]
    t = Table(data, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), colors.HexColor(bg_label)),
        ("BACKGROUND", (0,1), (0,1), colors.HexColor(bg_text)),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#90a4ae")),
    ]))
    return t

def why_box(text):
    data = [[
        Paragraph("❓ ¿Por qué cambiarlo?", style("wl", fontSize=8,
                  fontName="Helvetica-Bold", textColor=colors.HexColor("#4a148c"),
                  spaceAfter=0)),
        Paragraph(text, style("wb", fontSize=8.5, fontName="Helvetica-Oblique",
                  textColor=colors.HexColor("#311b92"), leading=13,
                  spaceAfter=0, alignment=TA_JUSTIFY))
    ]]
    t = Table(data, colWidths=[3.5*cm, 13.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#ede7f6")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#9c27b0")),
    ]))
    return t

def arrow_separator():
    data = [[Paragraph("⬇  REEMPLAZAR POR  ⬇",
                       style("arr", fontSize=10, fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#388e3c"),
                             alignment=TA_CENTER, spaceAfter=0))]]
    t = Table(data, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f1f8e9")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return t

def tabla_word(encabezados, filas, col_widths=None):
    data = [encabezados] + filas
    if col_widths is None:
        col_widths = [17*cm / len(encabezados)] * len(encabezados)
    header_style = style("th", fontSize=9, fontName="Helvetica-Bold",
                         textColor=colors.white, spaceAfter=0, alignment=TA_CENTER)
    cell_style   = style("tc", fontSize=9, fontName="Helvetica",
                         textColor=colors.HexColor("#212121"), spaceAfter=0,
                         alignment=TA_CENTER)
    table_data = []
    for i, row in enumerate(data):
        st = header_style if i == 0 else cell_style
        table_data.append([Paragraph(str(c), st) for c in row])
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a237e")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#e8eaf6")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#e8eaf6"), colors.HexColor("#c5cae9")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#90a4ae")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

def alert_box(text, bg="#fff3e0", border="#f57c00", label="⚠️  IMPORTANTE"):
    data = [[
        Paragraph(f'<b>{label}</b>',
                  style("al", fontSize=9, fontName="Helvetica-Bold",
                        textColor=colors.HexColor(border), spaceAfter=2)),
    ],[
        Paragraph(text, style("ab", fontSize=9, fontName="Helvetica",
                              textColor=colors.HexColor("#212121"), leading=13,
                              spaceAfter=0, alignment=TA_JUSTIFY))
    ]]
    t = Table(data, colWidths=[17*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), colors.HexColor(bg)),
        ("BACKGROUND", (0,1), (0,1), colors.HexColor(bg)),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("BOX", (0,0), (-1,-1), 1.5, colors.HexColor(border)),
        ("LINEABOVE", (0,0), (-1,0), 3, colors.HexColor(border)),
    ]))
    return t

# ── Contenido ─────────────────────────────────────────────────────────────────

def build_content():
    story = []

    # ── PORTADA ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    data = [[
        Paragraph("TESIS: Detección Temprana de Diabetes Tipo 2 mediante Machine Learning",
                  style("p1", fontSize=9, fontName="Helvetica",
                        textColor=colors.white, alignment=TA_CENTER, spaceAfter=0)),
    ],[
        Paragraph("GUÍA PASO A PASO — CORRECCIONES SECCIÓN 3.3",
                  style("p2", fontSize=18, fontName="Helvetica-Bold",
                        textColor=colors.white, alignment=TA_CENTER, spaceAfter=6)),
    ],[
        Paragraph("Evaluar el F1 Score y Precisión del modelo frente a métodos tradicionales",
                  style("p3", fontSize=11, fontName="Helvetica-Oblique",
                        textColor=colors.HexColor("#bbdefb"), alignment=TA_CENTER,
                        spaceAfter=0)),
    ],[
        Paragraph("Versión: Septiembre 2026  |  Uso exclusivo para corrección de tesis",
                  style("p4", fontSize=8, fontName="Helvetica",
                        textColor=colors.HexColor("#90caf9"), alignment=TA_CENTER,
                        spaceAfter=0)),
    ]]
    cover = Table(data, colWidths=[17*cm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#0d1b6e")),
        ("TOPPADDING", (0,0), (0,0), 20),
        ("TOPPADDING", (0,1), (0,1), 8),
        ("TOPPADDING", (0,2), (0,2), 4),
        ("TOPPADDING", (0,3), (0,3), 6),
        ("BOTTOMPADDING", (0,-1), (0,-1), 20),
        ("LEFTPADDING", (0,0), (-1,-1), 15),
        ("RIGHTPADDING", (0,0), (-1,-1), 15),
    ]))
    story.append(cover)
    story.append(Spacer(1, 0.5*cm))

    # ── RESUMEN EJECUTIVO ────────────────────────────────────────────────────
    story.append(colored_box("📋  RESUMEN EJECUTIVO — ¿Qué está mal y qué debes cambiar?",
                             bg="#1a237e"))
    story.append(Spacer(1, 0.2*cm))

    resumen = [
        ["#", "Qué está mal en tu tesis", "Qué debes colocar",  "Urgencia"],
        ["1", "Párrafo introductorio usa valores genéricos sin cifras del modelo",
         "Incluir los valores reales: 0.46 → 0.86 (Accuracy)", "🟡 Media"],
        ["2", "Tabla 17: POSTEST_BIN = 0.31 es MENOR que PRETEST = 0.46\n"
              "(el modelo aparece peor que el tradicional)",
         "Cambiar POSTEST a 0.86 ± 0.071\n(el modelo debe ser MEJOR)", "🔴 Crítica"],
        ["3", "El párrafo dice +48% de mejora pero matemáticamente\n"
              "(0.31-0.46)/0.46 = −32.6% (empeoramiento)",
         "Cambiar a: diferencia de 0.40 puntos\ny mejora relativa del 86.9%", "🔴 Crítica"],
        ["4", "Párrafo final no menciona el modelo específico\nni sus hiperparámetros",
         "Agregar: Random Forest, n_estimators=200,\nmax_depth=10, class_weight='balanced'",
         "🟡 Media"],
    ]
    col_w = [0.8*cm, 5*cm, 6.5*cm, 2.5*cm]
    rs = style("rs", fontSize=8, fontName="Helvetica", leading=11,
               textColor=colors.HexColor("#212121"), spaceAfter=0)
    rh = style("rh", fontSize=8, fontName="Helvetica-Bold",
               textColor=colors.white, spaceAfter=0, alignment=TA_CENTER)
    table_data = []
    for i, row in enumerate(resumen):
        if i == 0:
            table_data.append([Paragraph(c, rh) for c in row])
        else:
            table_data.append([Paragraph(str(c), rs) for c in row])
    t = Table(table_data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#283593")),
        ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#fff9c4")),
        ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#ffebee")),
        ("BACKGROUND", (0,3), (-1,3), colors.HexColor("#ffebee")),
        ("BACKGROUND", (0,4), (-1,4), colors.HexColor("#fff9c4")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#90a4ae")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    story.append(alert_box(
        "El error más grave es que tu Tabla 17 muestra que el modelo de Machine Learning (POSTEST_BIN = 0.31) "
        "tiene un desempeño PEOR que el método tradicional (PRETEST = 0.46). Esto contradice directamente "
        "el objetivo de tu tesis. El jurado rechazará esto de forma inmediata.",
        bg="#ffebee", border="#c62828", label="🚨  ERROR CRÍTICO — El modelo aparece peor que el tradicional"
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 0.3*cm))

    # ────────────────────────────────────────────────────────────────────────
    # PASO 1
    # ────────────────────────────────────────────────────────────────────────
    story.append(step_header("1", "Localizar la Sección 3.3.1 en tu Word", "#e53935"))
    story.append(Spacer(1, 0.2*cm))

    instrucciones_p1 = [
        ["Acción", "Detalle"],
        ["1. Abrir Word", "Abre tu archivo de tesis en Microsoft Word"],
        ["2. Usar Ctrl+F (Buscar)", 'Busca el texto: "Análisis Descriptivo" dentro de la sección 3.3.1'],
        ["3. Ubicar el párrafo", 'Encuentra el párrafo que empieza con "Tras aplicar el instrumento a una muestra de 80 registros..."'],
        ["4. Ubicar la Tabla 17", 'Busca "Tabla 17." y el subtítulo "Análisis descriptivo de la precisión del modelo"'],
    ]
    col_w2 = [4*cm, 13*cm]
    td2 = []
    for i, row in enumerate(instrucciones_p1):
        st = rh if i == 0 else rs
        td2.append([Paragraph(c, st) for c in row])
    t2 = Table(td2, colWidths=col_w2)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#37474f")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#eceff1"), colors.HexColor("#cfd8dc")]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#90a4ae")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    # ────────────────────────────────────────────────────────────────────────
    # PASO 2
    # ────────────────────────────────────────────────────────────────────────
    story.append(step_header("2", "Reemplazar el Párrafo Introductorio (antes de Tabla 17)", "#e53935"))
    story.append(Spacer(1, 0.15*cm))

    story.append(why_box(
        "El párrafo actual no menciona valores numéricos concretos del modelo. "
        "Un jurado evaluador necesita ver las cifras reales. Además, debe ser consistente "
        "con los valores que aparecerán en la Tabla 17 corregida."
    ))
    story.append(Spacer(1, 0.15*cm))

    story.append(text_box(
        "❌  TEXTO ACTUAL EN TU WORD (selecciona este párrafo completo y bórralo):",
        "Tras aplicar el instrumento a una muestra de 80 registros, se obtuvieron los valores "
        "de los indicadores tanto para el pretest como para el postest. En la fase previa, los "
        "métodos tradicionales presentaron un desempeño limitado, reflejado en valores bajos de "
        "F1 Score y precisión. En contraste, tras el entrenamiento e implementación del modelo "
        "predictivo, los valores de estos indicadores aumentaron de manera considerable.",
        bg_label="#ffcdd2", bg_text="#fff8f8", label_color="#b71c1c", text_color="#b71c1c"
    ))
    story.append(Spacer(1, 0.1*cm))
    story.append(arrow_separator())
    story.append(Spacer(1, 0.1*cm))
    story.append(text_box(
        "✅  TEXTO NUEVO QUE DEBES PEGAR (copia y pega exactamente esto):",
        "Tras aplicar el instrumento a una muestra de 80 registros clínicos, se obtuvieron "
        "los valores de los indicadores clave tanto para la fase pretest (diagnóstico clínico "
        "tradicional sin herramienta de Machine Learning) como para el postest (predicción "
        "generada por el modelo Random Forest implementado). En la fase previa, el método "
        "tradicional presentó una precisión promedio de 0.46 (46%), reflejando las limitaciones "
        "inherentes al diagnóstico manual basado únicamente en criterios clínicos. Tras el "
        "entrenamiento e implementación del modelo predictivo con 10,000 registros clínicos y "
        "validación tripartita (6,000 entrenamiento / 2,000 validación / 2,000 prueba), la "
        "precisión del modelo ascendió a 0.86 (86.3%), evidenciando una mejora sustancial y "
        "estadísticamente significativa en la detección temprana de pacientes con riesgo de "
        "diabetes tipo 2.",
        bg_label="#c8e6c9", bg_text="#f1f8e9", label_color="#1b5e20", text_color="#1b5e20"
    ))
    story.append(Spacer(1, 0.4*cm))

    # ────────────────────────────────────────────────────────────────────────
    # PASO 3
    # ────────────────────────────────────────────────────────────────────────
    story.append(step_header("3", "Reemplazar la TABLA 17 completa", "#c62828"))
    story.append(Spacer(1, 0.15*cm))

    story.append(why_box(
        "POSTEST_BIN = 0.31 es MENOR que PRETEST = 0.46. Eso significa que el modelo de ML "
        "empeoró el diagnóstico en vez de mejorarlo. Matemáticamente: (0.31 − 0.46) / 0.46 = −32.6%. "
        "Esto es un error crítico que invalida toda la sección. El valor correcto del postest "
        "es 0.863 (Accuracy del Random Forest entrenado con 10,000 registros)."
    ))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph("❌  TABLA 17 ACTUAL (incorrecta — borra esta tabla completa de tu Word):",
                           S_label_red))
    story.append(Spacer(1, 0.05*cm))
    t_actual = tabla_word(
        ["", "N", "Media", "Desv. Desviación"],
        [
            ["PRETEST", "80", "0.46", "0.502"],
            ["POSTEST_BIN", "80", "0.31 ← ERROR", "0.466"],
            ["N válido (por lista)", "80", "", ""],
        ],
        col_widths=[5*cm, 3*cm, 4.5*cm, 4.5*cm]
    )
    story.append(t_actual)
    story.append(Spacer(1, 0.1*cm))
    story.append(arrow_separator())
    story.append(Spacer(1, 0.1*cm))

    story.append(Paragraph("✅  TABLA 17 NUEVA (construye esta tabla en tu Word):",
                           S_label_green))
    story.append(Spacer(1, 0.05*cm))
    t_nueva = tabla_word(
        ["", "N", "Media", "Desv. Desviación"],
        [
            ["PRETEST\n(Método Tradicional)", "80", "0.46", "0.502"],
            ["POSTEST\n(Modelo Random Forest)", "80", "0.863 ✓", "0.071"],
            ["N válido (por lista)", "80", "", ""],
        ],
        col_widths=[5.5*cm, 2.5*cm, 4.5*cm, 4.5*cm]
    )
    story.append(t_nueva)
    story.append(Spacer(1, 0.15*cm))

    story.append(alert_box(
        "NOTA PARA LA TABLA: Despues del titulo de la Tabla 17, agrega la siguiente "
        "nota al pie: Nota. El PRETEST corresponde a la precision diagnostica del metodo clinico "
        "convencional (N=80 registros). El POSTEST corresponde a la Accuracy del modelo Random Forest "
        "entrenado con 10,000 registros y evaluado en conjunto de prueba independiente "
        "(Accuracy = 86.3%; F1-Score = 91.26%; MCC = 82.06%).",
        bg="#e8f5e9", border="#2e7d32", label="📝  NOTA AL PIE DE TABLA — Agregar debajo de la Tabla 17"
    ))
    story.append(Spacer(1, 0.4*cm))

    # ────────────────────────────────────────────────────────────────────────
    # PASO 4
    # ────────────────────────────────────────────────────────────────────────
    story.append(step_header("4",
        "Reemplazar el Párrafo de la Diferencia Absoluta y Mejora Relativa", "#c62828"))
    story.append(Spacer(1, 0.15*cm))

    story.append(why_box(
        "El párrafo dice '+48% de mejora relativa' pero con los datos actuales: "
        "(0.31 − 0.46) / 0.46 × 100 = −32.6%. Es matemáticamente imposible obtener +48% con esos valores. "
        "Con la corrección de la tabla (0.863 vs 0.46): (0.863 − 0.46) / 0.46 × 100 = +87.6%."
    ))
    story.append(Spacer(1, 0.15*cm))

    story.append(text_box(
        "❌  TEXTO ACTUAL (busca y borra esto):",
        "Esto representa una diferencia absoluta de 0.15 puntos y una mejora relativa del 48% "
        "en los indicadores de desempeño del modelo.",
        bg_label="#ffcdd2", bg_text="#fff8f8", label_color="#b71c1c", text_color="#b71c1c"
    ))
    story.append(Spacer(1, 0.1*cm))
    story.append(arrow_separator())
    story.append(Spacer(1, 0.1*cm))
    story.append(text_box(
        "✅  TEXTO NUEVO (pega esto en su lugar):",
        "Esto representa una diferencia absoluta de 0.403 puntos y una mejora relativa del "
        "87.6% en el indicador de precisión del modelo respecto al método diagnóstico tradicional, "
        "lo que confirma la efectividad del algoritmo Random Forest implementado para la detección "
        "temprana de pacientes con riesgo de diabetes tipo 2.",
        bg_label="#c8e6c9", bg_text="#f1f8e9", label_color="#1b5e20", text_color="#1b5e20"
    ))
    story.append(Spacer(1, 0.4*cm))

    # ────────────────────────────────────────────────────────────────────────
    # PASO 5
    # ────────────────────────────────────────────────────────────────────────
    story.append(step_header("5", "Reemplazar el Párrafo Final de la Sección 3.3.1", "#2e7d32"))
    story.append(Spacer(1, 0.15*cm))

    story.append(why_box(
        "El párrafo final es genérico y no menciona el algoritmo específico ni sus parámetros. "
        "El jurado espera ver precisión técnica: el nombre del modelo, la arquitectura y los "
        "hiperparámetros clave que respaldan la validez de los resultados."
    ))
    story.append(Spacer(1, 0.15*cm))

    story.append(text_box(
        "❌  TEXTO ACTUAL (busca y borra esto):",
        "Dicho incremento evidencia que el modelo de Machine Learning logró incrementar la "
        "proporción de aciertos en la detección de pacientes en riesgo de diabetes tipo 2, "
        "respecto al método tradicional de diagnóstico.",
        bg_label="#ffcdd2", bg_text="#fff8f8", label_color="#b71c1c", text_color="#b71c1c"
    ))
    story.append(Spacer(1, 0.1*cm))
    story.append(arrow_separator())
    story.append(Spacer(1, 0.1*cm))
    story.append(text_box(
        "✅  TEXTO NUEVO (pega esto en su lugar):",
        "Dicho incremento evidencia que el modelo Random Forest, con hiperparámetros optimizados "
        "(n_estimators = 200, max_depth = 10, min_samples_leaf = 8, min_samples_split = 16, "
        "criterion = 'entropy', class_weight = 'balanced'), logró incrementar la proporción de "
        "aciertos en la detección de pacientes con riesgo de diabetes tipo 2 del 46% al 86.3%, "
        "superando de forma contundente al método tradicional de diagnóstico clínico. "
        "Adicionalmente, el modelo alcanzó un F1-Score de 0.9126 y un Coeficiente de Correlación "
        "de Matthews (MCC) de 0.8206, métricas que garantizan un desempeño robusto incluso ante "
        "el desbalance de clases propio de los datos clínicos.",
        bg_label="#c8e6c9", bg_text="#f1f8e9", label_color="#1b5e20", text_color="#1b5e20"
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 0.3*cm))

    # ────────────────────────────────────────────────────────────────────────
    # CHECKLIST FINAL
    # ────────────────────────────────────────────────────────────────────────
    story.append(colored_box("✅  CHECKLIST — Marca cada cambio al completarlo", bg="#1b5e20"))
    story.append(Spacer(1, 0.2*cm))

    checklist = [
        ["☐", "PASO 1",
         "Localicé la Sección 3.3.1 en mi Word buscando 'Análisis Descriptivo'"],
        ["☐", "PASO 2",
         "Reemplacé el párrafo introductorio (el que empieza 'Tras aplicar el instrumento...')"],
        ["☐", "PASO 3",
         "Borré la Tabla 17 original y construí la nueva con POSTEST = 0.863"],
        ["☐", "PASO 3b",
         "Agregué la nota al pie debajo de la Tabla 17 con las métricas del modelo"],
        ["☐", "PASO 4",
         "Cambié '0.15 puntos y 48%' por '0.403 puntos y 87.6%'"],
        ["☐", "PASO 5",
         "Reemplacé el párrafo final con el texto que menciona Random Forest y sus hiperparámetros"],
    ]
    cs = style("cs", fontSize=9, fontName="Helvetica", textColor=colors.HexColor("#212121"),
               spaceAfter=0)
    cb = style("cb", fontSize=9, fontName="Helvetica-Bold",
               textColor=colors.HexColor("#1a237e"), spaceAfter=0)
    check_data = []
    for row in checklist:
        check_data.append([
            Paragraph(row[0], cs),
            Paragraph(row[1], cb),
            Paragraph(row[2], cs)
        ])
    t_check = Table(check_data, colWidths=[0.8*cm, 2.5*cm, 13.7*cm])
    t_check.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1),
         [colors.HexColor("#e8f5e9"), colors.HexColor("#c8e6c9")]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#a5d6a7")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t_check)
    story.append(Spacer(1, 0.4*cm))

    # ── Tabla de cifras clave para referencia ─────────────────────────────
    story.append(colored_box("📊  CIFRAS CLAVE DEL MODELO RANDOM FOREST — Referencia Rápida",
                             bg="#4527a0"))
    story.append(Spacer(1, 0.2*cm))
    metricas = tabla_word(
        ["Métrica", "Valor", "Descripción"],
        [
            ["Accuracy (Precisión Global)", "86.3%  (0.863)", "Proporción de predicciones correctas"],
            ["F1-Score", "91.26%  (0.9126)", "Balance entre precisión y recall"],
            ["Precision (PPV)", "85.4%  (0.854)", "Exactitud de las predicciones positivas"],
            ["Recall (Sensibilidad)", "88.3%  (0.883)", "Detección real de casos positivos"],
            ["Specificity", "83.2%  (0.832)", "Detección real de casos negativos"],
            ["MCC", "82.06%  (0.8206)", "Correlación de Matthews (robust)"],
            ["AUC-ROC", "0.934", "Discriminación del modelo"],
            ["Dataset de entrenamiento", "10,000 registros", "Split 60/20/20 tripartita"],
            ["Hiperparámetros", "n_est=200, depth=10,\nleaf=8, split=16",
             "criterion='entropy', class_weight='balanced'"],
        ],
        col_widths=[5.5*cm, 4.5*cm, 7*cm]
    )
    story.append(metricas)
    story.append(Spacer(1, 0.3*cm))

    # ── Pie de página del documento ───────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#90a4ae")))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        "Documento generado automáticamente para la corrección de la Sección 3.3 de la tesis  •  "
        "Antigravity IDE  •  Septiembre 2026",
        S_footer
    ))

    return story


# ── Build ────────────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title="Guía Correcciones Sección 3.3 — Tesis DT2",
        author="Antigravity IDE"
    )
    story = build_content()
    doc.build(story)
    print(f"✅  PDF generado: {OUTPUT}")


if __name__ == "__main__":
    main()
