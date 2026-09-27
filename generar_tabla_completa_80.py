#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera SOLO la tabla del Instrumento 3 con los 80 pacientes completos,
sin cortes, en formato landscape A4. Una fila por paciente.
"""
import math
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT = "Instrumento3_Tabla_Completa_80Pacientes.pdf"

# ── DATOS REALES ──────────────────────────────────────────────────────────────
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

# ── MÉTRICAS ──────────────────────────────────────────────────────────────────
TP=19; TN=37; FP=6; FN=18; N=80
accuracy  = (TP+TN)/N
precision = TP/(TP+FP)
recall    = TP/(TP+FN)
f1        = 2*precision*recall/(precision+recall)
especif   = TN/(TN+FP)
mcc_v     = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

# ── ESTILOS ───────────────────────────────────────────────────────────────────
def S(name,**kw): return ParagraphStyle(name,**kw)

# Estilos de celda
TH  = S("th",  fontSize=7.5, fontName="Helvetica-Bold",   textColor=colors.white,
         alignment=TA_CENTER, spaceAfter=0, leading=10)
TC  = S("tc",  fontSize=7.5, fontName="Helvetica",        textColor=colors.HexColor("#212121"),
         alignment=TA_CENTER, spaceAfter=0, leading=10)
TG  = S("tg",  fontSize=7.5, fontName="Helvetica-Bold",   textColor=colors.HexColor("#1b5e20"),
         alignment=TA_CENTER, spaceAfter=0, leading=10)
TR  = S("tr",  fontSize=7.5, fontName="Helvetica-Bold",   textColor=colors.HexColor("#c62828"),
         alignment=TA_CENTER, spaceAfter=0, leading=10)
TT  = S("tt",  fontSize=7.5, fontName="Helvetica-Bold",   textColor=colors.HexColor("#0d1b6e"),
         alignment=TA_CENTER, spaceAfter=0, leading=10)
TRE = S("tre", fontSize=7.5, fontName="Helvetica",        textColor=colors.HexColor("#212121"),
         alignment=TA_LEFT,   spaceAfter=0, leading=10)
TRG = S("trg", fontSize=7.5, fontName="Helvetica-Bold",   textColor=colors.HexColor("#1b5e20"),
         alignment=TA_LEFT,   spaceAfter=0, leading=10)
TRR = S("trr", fontSize=7.5, fontName="Helvetica-Bold",   textColor=colors.HexColor("#c62828"),
         alignment=TA_LEFT,   spaceAfter=0, leading=10)
S_title = S("stitle", fontSize=13, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0d1b6e"), alignment=TA_CENTER, spaceAfter=4)
S_sub   = S("ssub",   fontSize=9,  fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#455a64"), alignment=TA_CENTER, spaceAfter=6)
S_note  = S("snote",  fontSize=8,  fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#546e7a"), alignment=TA_JUSTIFY,
            leading=11, spaceAfter=0)
S_ftr   = S("sftr",   fontSize=7,  fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#90a4ae"), alignment=TA_CENTER)

# Anchos de columna — landscape A4 (27.7 cm util)
CW = [1*cm, 2.3*cm, 3*cm, 2.3*cm, 2.8*cm,
      1.1*cm, 1.1*cm, 1.1*cm, 1.1*cm, 10.9*cm]

def build():
    story = []

    # ── Encabezado del documento ──────────────────────────────────────────
    inst_hdr_data = [[
        Paragraph(
            "<b>INSTRUMENTO 3:</b>  Evaluacion del Desempeno del Modelo "
            "(F1 Score — Precision del Modelo y Diabetes)",
            S("ih", fontSize=10, fontName="Helvetica-Bold",
              textColor=colors.HexColor("#0d1b6e"), alignment=TA_CENTER, spaceAfter=0))
    ]]
    inst_hdr = Table(inst_hdr_data, colWidths=[sum(CW)])
    inst_hdr.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),1.5,colors.HexColor("#1a237e")),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    story.append(inst_hdr)

    # Ficha de registro
    meta = Table([
        [Paragraph("<b>TESIS</b>",
                   S("ml",fontSize=8,fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#0d1b6e"),alignment=TA_CENTER,spaceAfter=0)),
         Paragraph("Prediccion Temprana de Diabetes Tipo 2 aplicando Machine Learning "
                   "en Centro de Salud Casa Grande",
                   S("mv",fontSize=8,fontName="Helvetica",
                     textColor=colors.HexColor("#212121"),alignment=TA_LEFT,spaceAfter=0))],
        [Paragraph("<b>INVESTIGADORES</b>",
                   S("ml2",fontSize=8,fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#0d1b6e"),alignment=TA_CENTER,spaceAfter=0)),
         Paragraph("Ordonez Reyes Abraham Benjamin y Quispe Sanchez Edward Steven",
                   S("mv2",fontSize=8,fontName="Helvetica",
                     textColor=colors.HexColor("#212121"),alignment=TA_LEFT,spaceAfter=0))],
        [Paragraph("<b>FECHA DE INICIO</b>",
                   S("ml3",fontSize=8,fontName="Helvetica-Bold",
                     textColor=colors.HexColor("#0d1b6e"),alignment=TA_CENTER,spaceAfter=0)),
         Paragraph("08/09/2025   |   FECHA FIN: 08/09/2025",
                   S("mv3",fontSize=8,fontName="Helvetica",
                     textColor=colors.HexColor("#212121"),alignment=TA_LEFT,spaceAfter=0))],
    ], colWidths=[3.5*cm, sum(CW)-3.5*cm])
    meta.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#e8eaf6")),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#90a4ae")),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(meta)

    # Fila de formulas
    form_row = Table([[
        Paragraph("<b>VARIABLE</b>", TH),
        Paragraph("<b>INDICADOR</b>", TH),
        Paragraph("<b>FORMULAS CORRECTAS</b>", TH),
        Paragraph("<b>TECNICA</b>", TH),
        Paragraph("<b>INSTRUMENTO</b>", TH),
    ],[
        Paragraph("Modelo en\nMachine\nLearning",
                  S("fv",fontSize=7.5,fontName="Helvetica",
                    textColor=colors.HexColor("#212121"),alignment=TA_CENTER,
                    leading=10,spaceAfter=0)),
        Paragraph("F1 Score y\nPrecision\ndel modelo",
                  S("fv2",fontSize=7.5,fontName="Helvetica",
                    textColor=colors.HexColor("#212121"),alignment=TA_CENTER,
                    leading=10,spaceAfter=0)),
        Paragraph(
            "Accuracy = (TP+TN) / (TP+TN+FP+FN)\n"
            "Precision (PPV) = TP / (TP+FP)\n"
            "Recall = TP / (TP+FN)\n"
            "F1 = 2 x Precision x Recall / (Precision + Recall)",
            S("ff",fontSize=7,fontName="Helvetica",
              textColor=colors.HexColor("#212121"),alignment=TA_LEFT,
              leading=11,spaceAfter=0)),
        Paragraph("Observacion",
                  S("fv3",fontSize=7.5,fontName="Helvetica",
                    textColor=colors.HexColor("#212121"),alignment=TA_CENTER,
                    leading=10,spaceAfter=0)),
        Paragraph("Ficha de\nregistro",
                  S("fv4",fontSize=7.5,fontName="Helvetica",
                    textColor=colors.HexColor("#212121"),alignment=TA_CENTER,
                    leading=10,spaceAfter=0)),
    ]], colWidths=[3*cm,3*cm,12*cm,2.8*cm,3*cm])
    form_row.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#283593")),
        ("BACKGROUND",(0,1),(-1,1),colors.HexColor("#fafafa")),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#90a4ae")),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(form_row)
    story.append(Spacer(1,0.2*cm))

    # ── TABLA PRINCIPAL: 80 PACIENTES ────────────────────────────────────
    headers = [
        Paragraph("N°", TH),
        Paragraph("FECHA", TH),
        Paragraph("PRETEST\n(Clase Real)", TH),
        Paragraph("POSTEST\nSOFT", TH),
        Paragraph("POSTEST\nBIN", TH),
        Paragraph("TP", TH),
        Paragraph("TN", TH),
        Paragraph("FP", TH),
        Paragraph("FN", TH),
        Paragraph("RESULTADO DEL MODELO", TH),
    ]
    table_data = [headers]

    row_styles = []  # (row_index, is_correct)

    for idx, r in enumerate(RAW):
        idd, pre, soft, bn, tp, tn, fp, fn = r
        row_i = idx + 1

        label_pre = "Diabetico (1)" if pre == 1 else "Sano (0)"
        label_bin = "Diabetico (1)" if bn == 1 else "Sano (0)"
        correct = (tp == 1 or tn == 1)

        bin_style = TG if correct else TR

        if tp:
            res, res_s = "TP — Acierto: Diabetico detectado correctamente", TRG
        elif tn:
            res, res_s = "TN — Acierto: Sano detectado correctamente", TRG
        elif fp:
            res, res_s = "FP — Error: Sano predicho como Diabetico", TRR
        else:
            res, res_s = "FN — Error: Diabetico predicho como Sano", TRR

        row = [
            Paragraph(str(idd), TC),
            Paragraph("08/09/2025", TC),
            Paragraph(label_pre, TC),
            Paragraph(f"{soft:.2f}", TC),
            Paragraph(label_bin, bin_style),
            Paragraph(str(tp), TG if tp else TC),
            Paragraph(str(tn), TG if tn else TC),
            Paragraph(str(fp), TR if fp else TC),
            Paragraph(str(fn), TR if fn else TC),
            Paragraph(res, res_s),
        ]
        table_data.append(row)
        row_styles.append((row_i, correct))

    # Fila de TOTALES
    table_data.append([
        Paragraph("TOTAL", TT),
        Paragraph("—", TT),
        Paragraph("—", TT),
        Paragraph("—", TT),
        Paragraph("—", TT),
        Paragraph(str(TP), S("tTP",fontSize=8,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#1b5e20"),alignment=TA_CENTER,spaceAfter=0)),
        Paragraph(str(TN), S("tTN",fontSize=8,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#1b5e20"),alignment=TA_CENTER,spaceAfter=0)),
        Paragraph(str(FP), S("tFP",fontSize=8,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#c62828"),alignment=TA_CENTER,spaceAfter=0)),
        Paragraph(str(FN), S("tFN",fontSize=8,fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#c62828"),alignment=TA_CENTER,spaceAfter=0)),
        Paragraph(f"N={N}  |  Correctos={TP+TN}  |  Errores={FP+FN}", TT),
    ])

    # Fila de METRICAS GLOBALES
    table_data.append([
        Paragraph("METRICAS\nGLOBALES", TT),
        Paragraph(f"Accuracy\n{accuracy:.1%}",
                  S("mA",fontSize=7.5,fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#1b5e20"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(f"Precision\n{precision:.1%}",
                  S("mP",fontSize=7.5,fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#1b5e20"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(f"Recall\n{recall:.1%}",
                  S("mR",fontSize=7.5,fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#e65100"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(f"F1 Score\n{f1:.1%}",
                  S("mF",fontSize=7.5,fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#1b5e20"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(f"Espec.\n{especif:.1%}",
                  S("mE",fontSize=7.5,fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#1b5e20"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(f"MCC\n{mcc_v:.4f}",
                  S("mM",fontSize=7.5,fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#1b5e20"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(f"TP={TP}\nFP={FP}",
                  S("mTP",fontSize=7.5,fontName="Helvetica",
                    textColor=colors.HexColor("#212121"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(f"TN={TN}\nFN={FN}",
                  S("mTN",fontSize=7.5,fontName="Helvetica",
                    textColor=colors.HexColor("#212121"),alignment=TA_CENTER,spaceAfter=0,leading=10)),
        Paragraph(
            f"Formulas:  Accuracy=(TP+TN)/N  |  Precision=TP/(TP+FP)  |  "
            f"Recall=TP/(TP+FN)  |  F1=2xPxR/(P+R)",
            S("mDesc",fontSize=7,fontName="Helvetica-Oblique",
              textColor=colors.HexColor("#37474f"),alignment=TA_LEFT,spaceAfter=0,leading=10)),
    ])

    # Construir estilos de la tabla
    total_rows = len(table_data)
    tot_row  = total_rows - 2
    met_row  = total_rows - 1

    ts = [
        ("BACKGROUND", (0,0), (-1,0),    colors.HexColor("#1a237e")),   # header azul
        ("BACKGROUND", (0,tot_row), (-1,tot_row), colors.HexColor("#e3f2fd")),  # totales
        ("BACKGROUND", (0,met_row), (-1,met_row), colors.HexColor("#e8f5e9")),  # metricas
        ("GRID",       (0,0), (-1,-1),   0.35, colors.HexColor("#bdbdbd")),
        ("LINEABOVE",  (0,tot_row), (-1,tot_row), 1.5, colors.HexColor("#1a237e")),
        ("VALIGN",     (0,0), (-1,-1),   "MIDDLE"),
        ("LEFTPADDING",  (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
    ]

    # Color por fila (verde=correcto, naranja=error)
    for row_i, correct in row_styles:
        if correct:
            bg = colors.HexColor("#f1f8e9") if row_i % 2 == 1 else colors.HexColor("#e8f5e9")
        else:
            bg = colors.HexColor("#fff8f8") if row_i % 2 == 1 else colors.HexColor("#ffebee")
        ts.append(("BACKGROUND", (0,row_i), (-1,row_i), bg))

    t_main = Table(table_data, colWidths=CW, repeatRows=1)
    t_main.setStyle(TableStyle(ts))
    story.append(t_main)
    story.append(Spacer(1, 0.15*cm))

    # Nota al pie
    nota = Table([[Paragraph(
        f"<b>Nota.</b> N = {N} registros clinicos. PRETEST = clase real confirmada por historia "
        f"clinica (0=Sano, 1=Diabetico). POSTEST_SOFT = probabilidad continua del modelo "
        f"Random Forest (0.0-1.0). POSTEST_BIN = clasificacion binaria con umbral 0.5. "
        f"TP = Verdadero Positivo | TN = Verdadero Negativo | FP = Falso Positivo | "
        f"FN = Falso Negativo. Las metricas globales se calculan sobre el total acumulado "
        f"de los {N} casos, no por fila individual.",
        S_note
    )]], colWidths=[sum(CW)])
    nota.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f5f5f5")),
        ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#90a4ae")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(nota)
    story.append(Spacer(1,0.1*cm))
    story.append(Paragraph(
        "Instrumento 3 — Tabla Completa 80 Pacientes | "
        "Prediccion Temprana DT2 | Centro de Salud Casa Grande | Septiembre 2025",
        S_ftr
    ))
    return story


def main():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=landscape(A4),
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
        title="Instrumento 3 — Tabla Completa 80 Pacientes",
        author="Antigravity IDE"
    )
    doc.build(build())
    print(f"✅  PDF generado: {OUTPUT}")
    print(f"    TP={TP} | TN={TN} | FP={FP} | FN={FN} | N={N}")
    print(f"    Accuracy={accuracy:.1%} | Precision={precision:.1%} | "
          f"Recall={recall:.1%} | F1={f1:.1%}")


if __name__ == "__main__":
    main()
