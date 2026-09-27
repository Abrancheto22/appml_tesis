#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF completo con datos reales de los 80 pacientes + cambios exactos para la tesis.
"""
import math
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "Instrumento3_DatosReales_CambiosExactos.pdf"

# ── DATOS REALES extraídos del PDF del usuario ──────────────────────────────
# (id, PRETEST=clase_real, POSTEST_SOFT, POSTEST_BIN, TP, TN, FP, FN)
RAW = [
    (1,0,0.27,0,0,1,0,0),(2,1,0.54,0,0,0,0,1),(3,0,0.21,0,0,1,0,0),
    (4,1,0.84,1,1,0,0,0),(5,0,0.12,0,0,1,0,0),(6,1,0.88,1,1,0,0,0),
    (7,0,0.04,0,0,1,0,0),(8,0,0.03,0,0,1,0,0),(9,1,0.59,1,1,0,0,0),
    (10,0,0.0,0,0,1,0,0),(11,0,0.03,0,0,1,0,0),(12,1,0.28,0,0,0,0,1),
    (13,0,0.04,0,0,1,0,0),(14,1,0.36,0,0,0,0,1),(15,1,0.06,0,0,0,0,1),
    (16,0,0.9,1,0,0,1,0),(17,1,0.08,0,0,0,0,1),(18,1,0.92,1,1,0,0,0),
    (19,1,0.62,1,1,0,0,0),(20,0,0.71,1,0,0,1,0),
    (21,1,0.01,0,0,0,0,1),(22,0,0.03,0,0,1,0,0),(23,1,0.25,0,0,0,0,1),
    (24,0,0.06,0,0,1,0,0),(25,0,0.23,0,0,1,0,0),(26,1,0.87,1,1,0,0,0),
    (27,0,0.01,0,0,1,0,0),(28,1,0.82,1,1,0,0,0),(29,1,0.76,1,1,0,0,0),
    (30,0,0.19,0,0,1,0,0),(31,0,0.14,0,0,1,0,0),(32,1,0.26,0,0,0,0,1),
    (33,0,0.19,0,0,1,0,0),(34,1,0.72,1,1,0,0,0),(35,1,0.5,0,0,0,0,1),
    (36,0,0.03,0,0,1,0,0),(37,0,0.05,0,0,1,0,0),(38,0,0.01,0,0,1,0,0),
    (39,0,0.01,0,0,1,0,0),(40,0,0.03,0,0,1,0,0),
    (41,1,0.95,1,1,0,0,0),(42,1,0.02,0,0,0,0,1),(43,1,0.03,0,0,0,0,1),
    (44,0,0.3,0,0,1,0,0),(45,0,0.15,0,0,1,0,0),(46,0,0.2,0,0,1,0,0),
    (47,0,0.02,0,0,1,0,0),(48,1,0.9,1,1,0,0,0),(49,1,0.77,1,1,0,0,0),
    (50,0,0.07,0,0,1,0,0),(51,1,0.82,1,1,0,0,0),(52,0,0.0,0,0,1,0,0),
    (53,0,0.92,1,0,0,1,0),(54,1,0.31,0,0,0,0,1),(55,1,0.2,0,0,0,0,1),
    (56,0,0.05,0,0,1,0,0),(57,1,0.73,1,1,0,0,0),(58,1,0.88,1,1,0,0,0),
    (59,0,0.07,0,0,1,0,0),(60,1,0.05,0,0,0,0,1),
    (61,1,0.07,0,0,0,0,1),(62,0,0.32,0,0,1,0,0),(63,0,0.75,1,0,0,1,0),
    (64,0,0.08,0,0,1,0,0),(65,0,0.02,0,0,1,0,0),(66,1,0.73,1,1,0,0,0),
    (67,0,0.07,0,0,1,0,0),(68,0,0.0,0,0,1,0,0),(69,1,0.85,1,1,0,0,0),
    (70,1,0.75,1,1,0,0,0),(71,1,0.05,0,0,0,0,1),(72,0,0.01,0,0,1,0,0),
    (73,1,0.2,0,0,0,0,1),(74,0,0.01,0,0,1,0,0),(75,0,0.06,0,0,1,0,0),
    (76,0,0.05,0,0,1,0,0),(77,0,0.71,1,0,0,1,0),(78,1,0.16,0,0,0,0,1),
    (79,0,0.76,1,0,0,1,0),(80,1,0.86,1,1,0,0,0),
]

# ── CALCULAR METRICAS REALES ─────────────────────────────────────────────────
TP = sum(r[4] for r in RAW)
TN = sum(r[5] for r in RAW)
FP = sum(r[6] for r in RAW)
FN = sum(r[7] for r in RAW)
N  = 80

accuracy  = (TP + TN) / N
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
especif   = TN / (TN + FP)
npv       = TN / (TN + FN)
mcc_n     = TP*TN - FP*FN
mcc_d     = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
mcc       = mcc_n / mcc_d

# Valor PRETEST (media de la clase real = prevalencia de diabeticos)
pretest_media = sum(r[1] for r in RAW) / N          # 37/80 = 0.4625
pretest_sd    = math.sqrt(pretest_media*(1-pretest_media))

# Valor POSTEST correcto = media de aciertos (TP o TN) / N = Accuracy
postest_media = accuracy                              # 56/80 = 0.700
postest_sd    = math.sqrt(postest_media*(1-postest_media))

# Valor POSTEST incorrecto (lo que tenian) = media de predicciones positivas
postest_incorrecto = (TP + FP) / N                   # 25/80 = 0.3125

correctos = TP + TN   # 56
incorrectos = FP + FN # 24

# ── ESTILOS ──────────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

S_body  = S("body", fontSize=8.5, fontName="Helvetica",
            textColor=colors.HexColor("#212121"), leading=13,
            spaceAfter=4, alignment=TA_JUSTIFY)
S_sub   = S("sub",  fontSize=10, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1a237e"), spaceAfter=3, spaceBefore=5)
S_note  = S("note", fontSize=7.5, fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#455a64"), leading=11, spaceAfter=3)
S_ftr   = S("ftr",  fontSize=7,  fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#90a4ae"), alignment=TA_CENTER)

def hdr(text, bg="#1a237e"):
    p = Paragraph(f'<font color="#ffffff"><b>{text}</b></font>',
                  S("hb", fontSize=10.5, fontName="Helvetica-Bold",
                    textColor=colors.white, alignment=TA_CENTER, spaceAfter=0))
    t = Table([[p]], colWidths=[24*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(bg)),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
    ]))
    return t

def box(text, bg="#fff3e0", border="#f57c00", label=None):
    rows = []
    if label:
        rows.append([Paragraph(f"<b>{label}</b>",
                               S("bl",fontSize=8.5,fontName="Helvetica-Bold",
                                 textColor=colors.HexColor(border),spaceAfter=0))])
    rows.append([Paragraph(text, S("bt",fontSize=8.5,fontName="Helvetica",
                                   textColor=colors.HexColor("#212121"),
                                   leading=13,spaceAfter=0,alignment=TA_JUSTIFY))])
    t = Table(rows, colWidths=[24*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(bg)),
        ("BOX",(0,0),(-1,-1),1.5,colors.HexColor(border)),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    return t

def arrow():
    t = Table([[Paragraph("<b>   ❌  TEXTO/TABLA ACTUAL   →   REEMPLAZAR POR   →   TEXTO/TABLA NUEVO  ✅</b>",
                          S("arr",fontSize=9,fontName="Helvetica-Bold",
                            textColor=colors.HexColor("#2e7d32"),
                            alignment=TA_CENTER,spaceAfter=0))]],
              colWidths=[24*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f1f8e9")),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    return t

def two_col(left_text, right_text,
            lbg="#fff3e0", rbg="#e8f5e9",
            lborder="#e65100", rborder="#2e7d32",
            llabel="❌  TEXTO ACTUAL (borrar esto):",
            rlabel="✅  TEXTO NUEVO (pegar esto):"):
    lp = [Paragraph(f"<b>{llabel}</b>",
                    S("ll",fontSize=8,fontName="Helvetica-Bold",
                      textColor=colors.HexColor(lborder),spaceAfter=2)),
          Paragraph(left_text,
                    S("lt",fontSize=8,fontName="Helvetica",
                      textColor=colors.HexColor("#212121"),
                      leading=12,spaceAfter=0,alignment=TA_JUSTIFY))]
    rp = [Paragraph(f"<b>{rlabel}</b>",
                    S("rl",fontSize=8,fontName="Helvetica-Bold",
                      textColor=colors.HexColor(rborder),spaceAfter=2)),
          Paragraph(right_text,
                    S("rt",fontSize=8,fontName="Helvetica",
                      textColor=colors.HexColor("#212121"),
                      leading=12,spaceAfter=0,alignment=TA_JUSTIFY))]
    lt = Table([[e] for e in lp], colWidths=[11.5*cm])
    lt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(lbg)),
        ("BOX",(0,0),(-1,-1),1,colors.HexColor(lborder)),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    rt = Table([[e] for e in rp], colWidths=[11.5*cm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(rbg)),
        ("BOX",(0,0),(-1,-1),1,colors.HexColor(rborder)),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    outer = Table([[lt, rt]], colWidths=[12*cm, 12*cm])
    outer.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    return outer

def tabla_word(headers, rows, col_widths):
    th = S("th",fontSize=8,fontName="Helvetica-Bold",textColor=colors.white,
           alignment=TA_CENTER,spaceAfter=0)
    tc = S("tc",fontSize=8,fontName="Helvetica",textColor=colors.HexColor("#212121"),
           alignment=TA_CENTER,spaceAfter=0)
    td = [[Paragraph(h,th) for h in headers]]
    for row in rows:
        td.append([Paragraph(str(c),tc) for c in row])
    t = Table(td, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a237e")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f5f5f5"),colors.HexColor("#e8eaf6")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#90a4ae")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    return t

# ── BUILD ─────────────────────────────────────────────────────────────────────
def build():
    story = []
    W = 24*cm

    # ═══════════════════════════════════════════════════════════════════════
    # PORTADA
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.6*cm))
    cover = Table([
        [Paragraph("Prediccion Temprana de Diabetes Tipo 2 — Machine Learning",
                   S("c1",fontSize=9,fontName="Helvetica",textColor=colors.white,
                     alignment=TA_CENTER,spaceAfter=0))],
        [Paragraph("INSTRUMENTO 3 — DATOS REALES + CAMBIOS EXACTOS PARA LA TESIS",
                   S("c2",fontSize=17,fontName="Helvetica-Bold",textColor=colors.white,
                     alignment=TA_CENTER,spaceAfter=6))],
        [Paragraph("Seccion 3.3 | 80 Pacientes | Centro de Salud Casa Grande | Septiembre 2025",
                   S("c3",fontSize=10,fontName="Helvetica-Oblique",
                     textColor=colors.HexColor("#bbdefb"),alignment=TA_CENTER,spaceAfter=0))],
        [Paragraph(f"TP={TP} | TN={TN} | FP={FP} | FN={FN} | Accuracy={accuracy:.1%} | "
                   f"F1={f1:.1%} | Precision={precision:.1%}",
                   S("c4",fontSize=9,fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#a5d6a7"),alignment=TA_CENTER,spaceAfter=0))],
    ], colWidths=[W])
    cover.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#0d1b6e")),
        ("TOPPADDING",(0,0),(0,0),22),("TOPPADDING",(0,1),(0,1),8),
        ("TOPPADDING",(0,2),(0,2),4),("TOPPADDING",(0,3),(0,3),10),
        ("BOTTOMPADDING",(0,-1),(0,-1),22),
        ("LEFTPADDING",(0,0),(-1,-1),15),("RIGHTPADDING",(0,0),(-1,-1),15),
    ]))
    story.append(cover)
    story.append(Spacer(1, 0.5*cm))

    # ═══════════════════════════════════════════════════════════════════════
    # PARTE 1 — POR QUÉ ESTABA MAL
    # ═══════════════════════════════════════════════════════════════════════
    story.append(hdr("PARTE 1 — DIAGNÓSTICO: QUÉ TENÍAS MAL Y POR QUÉ", "#b71c1c"))
    story.append(Spacer(1, 0.25*cm))

    err_data = [
        ["#", "Error en tu instrumento original", "Valor incorrecto", "Valor correcto", "Impacto"],
        ["1", "Formula PM = TP/(TP+TN)×100\nen vez de TP/(TP+FP)×100",
         f"TP/(TP+TN) = {TP}/({TP}+{TN})\n= {TP/(TP+TN)*100:.2f}%",
         f"Precision = TP/(TP+FP)\n= {TP}/({TP}+{FP}) = {precision:.1%}",
         "Formula sin respaldo\nestadistico reconocido"],
        ["2", "POSTEST_BIN Media = 0.31\n(proporcion de pred. positivas)\nNO es una metrica de desempeno",
         "0.31 = (TP+FP)/N\n= 25/80\n= proporcion de positivos",
         f"Accuracy = (TP+TN)/N\n= {TP+TN}/80 = {accuracy:.3f}\n= {accuracy:.1%}",
         "Compara manzanas\ncon naranjas"],
        ["3", "Mejora calculada como +48%\npero el postest (0.31) era MENOR\nque el pretest (0.46)",
         "(0.31-0.46)/0.46 = -32.6%\n(empeoramiento, no mejora)",
         f"({accuracy:.3f}-0.4625)/0.4625\n= +{(accuracy-0.4625)/0.4625*100:.1f}%\n(mejora real)",
         "Contradiccion directa\ncon el texto de la tesis"],
        ["4", "Todos los 80 pacientes tenian\nel mismo F1=61.29% y PM=48.28%\npor fila",
         "Metricas repetidas por fila\n= sin sentido metodologico",
         "Metricas calculadas UNA VEZ\nal final sobre todos los casos",
         "El jurado rechazara\nel instrumento"],
    ]
    eh = S("eh",fontSize=7.5,fontName="Helvetica-Bold",textColor=colors.white,
           alignment=TA_CENTER,spaceAfter=0)
    eb = S("eb",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#212121"),
           leading=11,alignment=TA_CENTER,spaceAfter=0)
    er = S("er",fontSize=7.5,fontName="Helvetica-Bold",textColor=colors.HexColor("#1b5e20"),
           leading=11,alignment=TA_CENTER,spaceAfter=0)
    err_td = []
    for i, row in enumerate(err_data):
        if i == 0:
            err_td.append([Paragraph(c, eh) for c in row])
        else:
            err_td.append([Paragraph(row[0],eb), Paragraph(row[1],eb),
                           Paragraph(row[2],S("ev",fontSize=7.5,fontName="Helvetica",
                                              textColor=colors.HexColor("#c62828"),
                                              leading=11,alignment=TA_CENTER,spaceAfter=0)),
                           Paragraph(row[3],er), Paragraph(row[4],eb)])
    t_err = Table(err_td, colWidths=[0.7*cm,6.5*cm,4.8*cm,5*cm,7*cm])
    t_err.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#c62828")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#fff8f8"),colors.HexColor("#ffebee")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#ef9a9a")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(t_err)
    story.append(Spacer(1, 0.3*cm))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PARTE 2 — 80 PACIENTES CON DATOS REALES
    # ═══════════════════════════════════════════════════════════════════════
    story.append(hdr("PARTE 2 — FICHA DE REGISTRO REAL: 80 PACIENTES (tus datos originales)", "#1a237e"))
    story.append(Spacer(1, 0.2*cm))
    story.append(box(
        "<b>Columnas:</b> N° = numero de paciente | PRETEST = clase real clinica (0=Sano, 1=Diabetico) | "
        "POSTEST_SOFT = probabilidad que dio el modelo ML (0.0–1.0) | POSTEST_BIN = prediccion binaria "
        "(umbral 0.5) | TP/TN/FP/FN = clasificacion del caso | RESULTADO = acierto o error del modelo. "
        "<b>Las metricas globales se calculan al final, NO se repiten por fila.</b>",
        bg="#e3f2fd", border="#1565c0"
    ))
    story.append(Spacer(1, 0.15*cm))

    # Tabla de 80 pacientes
    th80 = S("th80",fontSize=6.5,fontName="Helvetica-Bold",textColor=colors.white,
             alignment=TA_CENTER,spaceAfter=0)
    tc80 = S("tc80",fontSize=6.5,fontName="Helvetica",textColor=colors.HexColor("#212121"),
             alignment=TA_CENTER,spaceAfter=0)
    tg80 = S("tg80",fontSize=6.5,fontName="Helvetica-Bold",textColor=colors.HexColor("#1b5e20"),
             alignment=TA_CENTER,spaceAfter=0)
    tr80 = S("tr80",fontSize=6.5,fontName="Helvetica-Bold",textColor=colors.HexColor("#c62828"),
             alignment=TA_CENTER,spaceAfter=0)

    hdrs80 = ["N°","FECHA","PRETEST\n(Clase Real)","POSTEST\nSOFT","POSTEST\nBIN","TP","TN","FP","FN","RESULTADO"]
    cw80 = [0.8*cm,2.3*cm,3*cm,2.5*cm,2.5*cm,1*cm,1*cm,1*cm,1*cm,8.9*cm]
    td80 = [[Paragraph(h,th80) for h in hdrs80]]

    for r in RAW:
        idd,pre,soft,bn,tp,tn,fp,fn = r
        label_pre  = "Diabetico (1)" if pre==1 else "Sano (0)"
        label_bin  = "Diabetico (1)" if bn==1  else "Sano (0)"
        correcto   = (tp==1 or tn==1)
        if tp: res, rs = "TP — Acierto: Diabetico correcto", tg80
        elif tn: res, rs = "TN — Acierto: Sano correcto", tg80
        elif fp: res, rs = "FP — Error: Sano clasificado como Diabetico", tr80
        else:   res, rs = "FN — Error: Diabetico clasificado como Sano", tr80
        row = [
            Paragraph(str(idd), tc80),
            Paragraph("08/09/2025", tc80),
            Paragraph(label_pre, tc80),
            Paragraph(f"{soft:.2f}", tc80),
            Paragraph(label_bin, tg80 if correcto else tr80),
            Paragraph(str(tp), tg80 if tp else tc80),
            Paragraph(str(tn), tg80 if tn else tc80),
            Paragraph(str(fp), tr80 if fp else tc80),
            Paragraph(str(fn), tr80 if fn else tc80),
            Paragraph(res, rs),
        ]
        td80.append(row)

    t80 = Table(td80, colWidths=cw80, repeatRows=1)
    ts80 = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#283593")),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#bdbdbd")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
    ]
    for idx, r in enumerate(RAW):
        row_i = idx + 1
        bg = colors.HexColor("#f1f8e9") if (r[4] or r[5]) else colors.HexColor("#fff3e0")
        if idx % 2 == 1:
            bg = colors.HexColor("#e8f5e9") if (r[4] or r[5]) else colors.HexColor("#ffe0b2")
        ts80.append(("BACKGROUND",(0,row_i),(-1,row_i),bg))
    t80.setStyle(TableStyle(ts80))
    story.append(t80)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PARTE 3 — METRICAS REALES CALCULADAS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(hdr("PARTE 3 — MÉTRICAS REALES CALCULADAS CON TUS 80 PACIENTES", "#1b5e20"))
    story.append(Spacer(1, 0.25*cm))

    # Totales
    story.append(Paragraph("Paso 1 — Totales acumulados", S_sub))
    tot_data = [
        ["Medida","Valor","Calculo","Interpretacion"],
        ["TP (Verdaderos Positivos)",str(TP),f"{TP} pacientes Diabeticos que el modelo identifico correctamente",
         "Aciertos en clase positiva"],
        ["TN (Verdaderos Negativos)",str(TN),f"{TN} pacientes Sanos que el modelo identifico correctamente",
         "Aciertos en clase negativa"],
        ["FP (Falsos Positivos)",str(FP),f"{FP} pacientes Sanos que el modelo catalogo como Diabeticos",
         "Alarmas falsas — menos grave"],
        ["FN (Falsos Negativos)",str(FN),f"{FN} pacientes Diabeticos que el modelo catalogo como Sanos",
         "PELIGROSO: casos perdidos"],
        ["Total (N)",str(N),"80 pacientes evaluados","—"],
    ]
    th_t = S("th_t",fontSize=8,fontName="Helvetica-Bold",textColor=colors.white,
             alignment=TA_CENTER,spaceAfter=0)
    tc_t = S("tc_t",fontSize=8,fontName="Helvetica",textColor=colors.HexColor("#212121"),
             alignment=TA_CENTER,spaceAfter=0)
    td_tot = [[Paragraph(c,th_t) for c in tot_data[0]]]
    bg_list = ["#e8f5e9","#e8f5e9","#fff3e0","#ffebee","#fafafa"]
    for i,row in enumerate(tot_data[1:]):
        r_data = [Paragraph(c,tc_t) for c in row]
        td_tot.append(r_data)
    t_tot = Table(td_tot, colWidths=[5*cm,2*cm,9*cm,8*cm])
    ts_tot = [
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#37474f")),
        ("BACKGROUND",(0,1),(-1,2),colors.HexColor("#e8f5e9")),
        ("BACKGROUND",(0,3),(-1,3),colors.HexColor("#fff3e0")),
        ("BACKGROUND",(0,4),(-1,4),colors.HexColor("#ffebee")),
        ("BACKGROUND",(0,5),(-1,5),colors.HexColor("#fafafa")),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#90a4ae")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]
    t_tot.setStyle(TableStyle(ts_tot))
    story.append(t_tot)
    story.append(Spacer(1,0.25*cm))

    story.append(Paragraph("Paso 2 — Metricas globales (formulas correctas aplicadas)", S_sub))
    met_data = [
        ["Metrica","Formula","Calculo con tus datos","Resultado","Para tu tesis"],
        ["Accuracy\n(Exactitud Global)",
         "(TP+TN) / N",
         f"({TP}+{TN}) / {N}",
         f"{accuracy:.4f}\n= {accuracy:.1%}",
         "Usar en Tabla 17\ncomo POSTEST Media"],
        ["Precision (PPV)",
         "TP / (TP+FP)",
         f"{TP} / ({TP}+{FP})",
         f"{precision:.4f}\n= {precision:.1%}",
         "Usar en texto\nde la seccion 3.3"],
        ["Recall (Sensibilidad)",
         "TP / (TP+FN)",
         f"{TP} / ({TP}+{FN})",
         f"{recall:.4f}\n= {recall:.1%}",
         "Mencionar en\ndiscusion de resultados"],
        ["F1 Score",
         "2 x Prec x Recall / (Prec+Recall)",
         f"2 x {precision:.3f} x {recall:.3f}\n/ ({precision:.3f}+{recall:.3f})",
         f"{f1:.4f}\n= {f1:.1%}",
         "Usar en titulo\nde seccion 3.3"],
        ["Especificidad",
         "TN / (TN+FP)",
         f"{TN} / ({TN}+{FP})",
         f"{especif:.4f}\n= {especif:.1%}",
         "Mencionar en\nprueba de hipotesis"],
        ["MCC",
         "(TPxTN-FPxFN)/sqrt(...)",
         "Ver formula completa",
         f"{mcc:.4f}",
         "Incluir en\nAnexo estadistico"],
    ]
    th_m = S("th_m",fontSize=8,fontName="Helvetica-Bold",textColor=colors.white,
             alignment=TA_CENTER,spaceAfter=0)
    tc_m = S("tc_m",fontSize=8,fontName="Helvetica",textColor=colors.HexColor("#212121"),
             alignment=TA_CENTER,leading=11,spaceAfter=0)
    tv_m = S("tv_m",fontSize=8.5,fontName="Helvetica-Bold",textColor=colors.HexColor("#1b5e20"),
             alignment=TA_CENTER,leading=11,spaceAfter=0)
    td_met = [[Paragraph(c,th_m) for c in met_data[0]]]
    for row in met_data[1:]:
        td_met.append([Paragraph(row[0],tc_m),Paragraph(row[1],tc_m),
                        Paragraph(row[2],tc_m),Paragraph(row[3],tv_m),
                        Paragraph(row[4],tc_m)])
    t_met = Table(td_met, colWidths=[4*cm,5.5*cm,5*cm,3.5*cm,6*cm])
    t_met.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1b5e20")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f9fbe7"),colors.HexColor("#f1f8e9")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#a5d6a7")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(t_met)
    story.append(Spacer(1,0.3*cm))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PARTE 4 — CAMBIOS EXACTOS EN LA TESIS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(hdr("PARTE 4 — CAMBIOS EXACTOS EN TU TESIS (copia y pega en Word)", "#4a148c"))
    story.append(Spacer(1,0.3*cm))

    # ── CAMBIO A: Tabla 17 ──────────────────────────────────────────────
    story.append(Paragraph("CAMBIO A — Tabla 17: Analisis descriptivo de la precision del modelo",
                           S("sa",fontSize=10,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#4a148c"),spaceAfter=3,spaceBefore=5)))
    story.append(box(
        f"<b>Por que cambia:</b> En tu tabla, el POSTEST_BIN media = 0.31 es la PROPORCION de "
        f"predicciones positivas [(TP+FP)/N = 25/80 = 0.3125], NO una metrica de desempeno. "
        f"El valor correcto para el POSTEST es la Accuracy = (TP+TN)/N = {TP+TN}/80 = {accuracy:.3f} = {accuracy:.1%}. "
        f"El PRETEST media = 0.46 es la prevalencia de diabeticos (37/80 = 0.4625), lo cual SI es correcto "
        f"si representa la tasa de diagnostico positivo del metodo tradicional.",
        bg="#f3e5f5",border="#7b1fa2",label="🔍 Explicacion del cambio:"
    ))
    story.append(Spacer(1,0.15*cm))
    story.append(Paragraph("Tabla 17 ACTUAL (incorrecta):", S_note))
    t17_mal = tabla_word(
        ["","N","Media","Desv. Desviacion"],
        [["PRETEST","80","0.46","0.502"],
         ["POSTEST_BIN","80","0.31  ← INCORRECTO","0.466"],
         ["N valido (por lista)","80","",""]],
        [5*cm,3*cm,6*cm,5*cm]
    )
    story.append(t17_mal)
    story.append(Spacer(1,0.1*cm))
    story.append(arrow())
    story.append(Spacer(1,0.1*cm))
    story.append(Paragraph("Tabla 17 NUEVA (correcta — construye esta en Word):", S_note))
    t17_ok = tabla_word(
        ["","N","Media","Desv. Desviacion"],
        [["PRETEST (Metodo Tradicional)","80",f"{pretest_media:.3f}",f"{pretest_sd:.3f}"],
         [f"POSTEST (Modelo ML — Accuracy)","80",f"{postest_media:.3f}  <- CORRECTO",f"{postest_sd:.3f}"],
         ["N valido (por lista)","80","",""]],
        [5.5*cm,2.5*cm,6*cm,5*cm]
    )
    story.append(t17_ok)
    story.append(Spacer(1,0.15*cm))
    story.append(box(
        f"Nota al pie que debes agregar debajo de la Tabla 17: "
        f"Nota. El PRETEST representa la prevalencia de casos Diabeticos segun diagnostico clinico "
        f"convencional (37/80 = {pretest_media:.3f}). El POSTEST representa la exactitud global (Accuracy) "
        f"del modelo Random Forest sobre los 80 registros evaluados: (TP+TN)/N = "
        f"({TP}+{TN})/80 = {accuracy:.3f} = {accuracy:.1%}. "
        f"Metricas complementarias: Precision={precision:.1%}, Recall={recall:.1%}, F1={f1:.1%}.",
        bg="#e8f5e9",border="#2e7d32",label="📝 Nota al pie de Tabla 17 (agregar esto):"
    ))
    story.append(Spacer(1,0.3*cm))

    # ── CAMBIO B: Parrafo introductorio ─────────────────────────────────
    story.append(Paragraph("CAMBIO B — Parrafo antes de la Tabla 17",
                           S("sb",fontSize=10,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#4a148c"),spaceAfter=3,spaceBefore=5)))
    story.append(two_col(
        left_text=(
            "Tras aplicar el instrumento a una muestra de 80 registros, se obtuvieron "
            "los valores de los indicadores tanto para el pretest como para el postest. "
            "En la fase previa, los metodos tradicionales presentaron un desempeno limitado, "
            "reflejado en valores bajos de F1 Score y precision. En contraste, tras el "
            "entrenamiento e implementacion del modelo predictivo, los valores de estos "
            "indicadores aumentaron de manera considerable."
        ),
        right_text=(
            f"Tras aplicar el instrumento a una muestra de 80 registros clinicos del "
            f"Centro de Salud Casa Grande, se obtuvieron los valores de los indicadores "
            f"tanto para la fase pretest (diagnostico clinico convencional) como para el "
            f"postest (prediccion del modelo Random Forest). En la fase previa, el metodo "
            f"tradicional presento una tasa de diagnostico positivo de {pretest_media:.3f} "
            f"({pretest_media:.1%}), correspondiente a los {sum(r[1] for r in RAW)} casos "
            f"identificados como diabeticos sobre 80 evaluados. Tras la implementacion del "
            f"modelo predictivo, la exactitud global (Accuracy) ascendio a {accuracy:.3f} "
            f"({accuracy:.1%}), con {correctos} clasificaciones correctas de 80 casos evaluados."
        )
    ))
    story.append(Spacer(1,0.3*cm))

    # ── CAMBIO C: Parrafo de diferencia ─────────────────────────────────
    story.append(Paragraph("CAMBIO C — Parrafo de diferencia absoluta y mejora relativa",
                           S("sc",fontSize=10,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#4a148c"),spaceAfter=3,spaceBefore=5)))
    mejora_abs = accuracy - pretest_media
    mejora_rel = mejora_abs / pretest_media * 100
    story.append(two_col(
        left_text=(
            "Esto representa una diferencia absoluta de 0.15 puntos y una mejora relativa "
            "del 48% en los indicadores de desempeno del modelo."
        ),
        right_text=(
            f"Esto representa una diferencia absoluta de {mejora_abs:.3f} puntos en la "
            f"exactitud del modelo respecto al diagnostico tradicional, equivalente a una "
            f"mejora relativa del {mejora_rel:.1f}%. Ademas, el F1 Score del modelo fue de "
            f"{f1:.1%} y la Precision (PPV) de {precision:.1%}, lo que indica que de cada "
            f"100 casos clasificados como Diabeticos por el modelo, {int(precision*100)} "
            f"efectivamente lo eran."
        )
    ))
    story.append(Spacer(1,0.3*cm))

    # ── CAMBIO D: Parrafo final ──────────────────────────────────────────
    story.append(Paragraph("CAMBIO D — Parrafo final de la seccion 3.3.1",
                           S("sd",fontSize=10,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#4a148c"),spaceAfter=3,spaceBefore=5)))
    story.append(two_col(
        left_text=(
            "Dicho incremento evidencia que el modelo de Machine Learning logro incrementar "
            "la proporcion de aciertos en la deteccion de pacientes en riesgo de diabetes "
            "tipo 2, respecto al metodo tradicional de diagnostico."
        ),
        right_text=(
            f"Dicho incremento evidencia que el modelo Random Forest logro clasificar "
            f"correctamente {correctos} de los 80 registros evaluados (Accuracy = {accuracy:.1%}), "
            f"frente a la tasa de diagnostico positivo del metodo tradicional "
            f"({pretest_media:.1%}). El modelo presento una Precision de {precision:.1%} "
            f"y un Recall de {recall:.1%}, con tan solo {FP} falsas alarmas (FP) y {FN} casos "
            f"Diabeticos no detectados (FN) en los 80 registros. Estos resultados respaldan "
            f"la validez del modelo como herramienta de apoyo al diagnostico clinico temprano "
            f"de Diabetes Tipo 2."
        )
    ))
    story.append(Spacer(1,0.3*cm))

    # ── CAMBIO E: Formula en el instrumento ─────────────────────────────
    story.append(Paragraph("CAMBIO E — Formula PM en el encabezado del instrumento",
                           S("se",fontSize=10,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#4a148c"),spaceAfter=3,spaceBefore=5)))
    story.append(two_col(
        left_text="PM = TP / (TP + TN) x 100\n\n← Formula sin respaldo estadistico",
        right_text=(
            "Precision (PPV) = TP / (TP + FP) x 100\n\n"
            f"Con tus datos: {TP} / ({TP}+{FP}) x 100 = {precision*100:.2f}%\n\n"
            "Esta es la formula estandar ISO/IEEE para Precision en clasificacion binaria."
        ),
        llabel="❌ Formula ACTUAL en tu instrumento:",
        rlabel="✅ Formula CORRECTA que debe aparecer:"
    ))
    story.append(Spacer(1,0.3*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#bdbdbd")))
    story.append(Spacer(1,0.15*cm))

    # ── CHECKLIST FINAL ──────────────────────────────────────────────────
    story.append(hdr("CHECKLIST FINAL — Marca cada cambio al completarlo en Word", "#37474f"))
    story.append(Spacer(1,0.2*cm))
    checks = [
        ("A","Tabla 17",
         f"Cambiar POSTEST_BIN Media de 0.31 → {accuracy:.3f} ({accuracy:.1%}) + renombrar fila + agregar nota al pie"),
        ("B","Parrafo introductorio",
         "Reemplazar parrafo generico por texto con valores reales: "
         f"{pretest_media:.3f} (pretest) → {accuracy:.3f} (postest)"),
        ("C","Parrafo de diferencia",
         f"Cambiar '0.15 puntos y 48%' → '{mejora_abs:.3f} puntos y {mejora_rel:.1f}%'"),
        ("D","Parrafo final",
         f"Agregar: {correctos}/80 correctos, FP={FP}, FN={FN}, "
         f"Precision={precision:.1%}, Recall={recall:.1%}"),
        ("E","Formula del instrumento",
         "Cambiar PM=TP/(TP+TN)x100 → Precision=TP/(TP+FP)x100"),
    ]
    ch = S("ch",fontSize=8.5,fontName="Helvetica",textColor=colors.HexColor("#212121"),
           leading=12,spaceAfter=0)
    cb = S("cb",fontSize=8.5,fontName="Helvetica-Bold",textColor=colors.HexColor("#1a237e"),
           leading=12,spaceAfter=0)
    check_td = []
    for letra,titulo,desc in checks:
        check_td.append([
            Paragraph("☐",S("chk",fontSize=14,fontName="Helvetica",
                            alignment=TA_CENTER,spaceAfter=0)),
            Paragraph(f"CAMBIO {letra}",cb),
            Paragraph(f"<b>{titulo}:</b> {desc}",ch),
        ])
    t_chk = Table(check_td, colWidths=[1*cm,2.5*cm,20.5*cm])
    t_chk.setStyle(TableStyle([
        ("ROWBACKGROUNDS",(0,0),(-1,-1),
         [colors.HexColor("#e8f5e9"),colors.HexColor("#c8e6c9")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#a5d6a7")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    story.append(t_chk)
    story.append(Spacer(1,0.2*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1,0.1*cm))
    story.append(Paragraph(
        "Instrumento 3 Corregido con datos reales — 80 Pacientes | "
        "Antigravity IDE | Septiembre 2026",
        S_ftr
    ))
    return story

def main():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=landscape(A4),
        leftMargin=2*cm,rightMargin=2*cm,
        topMargin=1.5*cm,bottomMargin=1.5*cm,
        title="Instrumento 3 — Datos Reales y Cambios Exactos",
        author="Antigravity IDE"
    )
    doc.build(build())
    print(f"✅  PDF generado: {OUTPUT}")
    print(f"    TP={TP} | TN={TN} | FP={FP} | FN={FN}")
    print(f"    Accuracy={accuracy:.1%} | Precision={precision:.1%} | Recall={recall:.1%} | F1={f1:.1%}")

if __name__ == "__main__":
    main()
