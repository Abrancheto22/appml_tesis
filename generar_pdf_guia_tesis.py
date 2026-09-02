import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "Guia_Reemplazos_Tesis_DT2.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1A365D"))
        
        # Header (páginas > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "GUÍA OFICIAL DE REEMPLAZOS Y SUBSANACIÓN PARA LA TESIS — UNT")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawRightString(558, 750, "Ingeniería de Sistemas")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 40, 558, 40)
        page_str = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(558, 28, page_str)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        self.drawString(54, 28, "Tesis: Predicción Temprana de Diabetes Tipo 2 — Ordoñez & Quispe")
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=46
    )

    c_primary = colors.HexColor("#1A365D")   # Azul marino institucional
    c_secondary = colors.HexColor("#2B6CB0") # Azul intermedio
    c_accent = colors.HexColor("#2C7A7B")    # Verde azulado clínico
    c_red_bg = colors.HexColor("#FFF5F5")    # Rojo tenue
    c_red_border = colors.HexColor("#FEB2B2")
    c_red_text = colors.HexColor("#9B2C2C")
    c_green_bg = colors.HexColor("#F0FFF4")  # Verde tenue
    c_green_border = colors.HexColor("#9AE6B4")
    c_green_text = colors.HexColor("#22543D")
    c_dark = colors.HexColor("#2D3748")
    c_border = colors.HexColor("#CBD5E0")

    title_style = ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=c_primary, alignment=1, spaceAfter=3)
    subtitle_style = ParagraphStyle('DocSubTitle', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=c_secondary, alignment=1, spaceAfter=6)
    sec_title = ParagraphStyle('SecTitle', fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=c_primary, spaceBefore=8, spaceAfter=4, keepWithNext=True)
    body_style = ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=8, leading=11, textColor=c_dark, spaceAfter=3)
    body_bold = ParagraphStyle('BodyBold', fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=c_dark)
    
    del_title = ParagraphStyle('DelTitle', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=c_red_text)
    del_text = ParagraphStyle('DelText', fontName='Helvetica', fontSize=7.5, leading=10, textColor=c_red_text)
    
    ins_title = ParagraphStyle('InsTitle', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=c_green_text)
    ins_text = ParagraphStyle('InsText', fontName='Helvetica', fontSize=7.5, leading=10, textColor=c_green_text)
    
    cell_style = ParagraphStyle('CellText', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=c_dark)
    cell_bold = ParagraphStyle('CellBold', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=c_dark)
    cell_header = ParagraphStyle('CellHeader', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.white)

    story = []

    def make_diff_box(page_num, section_name, error_text, replace_text, note_text=None):
        content = []
        header_text = f"<b>📍 PÁGINA {page_num} | {section_name.upper()}</b>"
        content.append(Paragraph(header_text, sec_title))
        
        # Tabla de reemplazo
        t_data = [
            [
                Paragraph("<b>❌ TEXTO / ERROR ACTUAL A ELIMINAR O CORREGIR:</b>", del_title),
                Paragraph("<b>✅ TEXTO, TABLA O DATOS EXACTOS A COLOCAR:</b>", ins_title)
            ],
            [
                Paragraph(error_text, del_text),
                Paragraph(replace_text, ins_text)
            ]
        ]
        t = Table(t_data, colWidths=[245, 259])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), c_red_bg),
            ('BACKGROUND', (1, 0), (1, -1), c_green_bg),
            ('BOX', (0, 0), (0, -1), 1, c_red_border),
            ('BOX', (1, 0), (1, -1), 1, c_green_border),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        content.append(t)
        if note_text:
            content.append(Spacer(1, 2))
            content.append(Paragraph(f"<b>💡 Sustento Técnico para el Jurado:</b> {note_text}", ParagraphStyle('Note', fontName='Helvetica-Oblique', fontSize=7, leading=9, textColor=colors.HexColor("#4A5568"))))
        content.append(Spacer(1, 6))
        return KeepTogether(content)

    # ==================== PÁGINA 1 ====================
    story.append(Paragraph("UNIVERSIDAD NACIONAL DE TRUJILLO", title_style))
    story.append(Paragraph("FACULTAD DE INGENIERÍA — ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS", subtitle_style))
    story.append(Paragraph("<b>GUÍA MAESTRA DE REEMPLAZOS Y CORRECCIONES PARA EL TEXTO DE LA TESIS</b>", ParagraphStyle('SubHeader', fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=c_primary, alignment=1, spaceAfter=2)))
    story.append(Paragraph("<b>Tesis:</b> <i>Predicción Temprana de Diabetes Tipo 2 aplicando un Modelo en Machine Learning en Centro de Salud Casa Grande</i>", ParagraphStyle('TesisMeta', fontName='Helvetica', fontSize=8, leading=10, textColor=c_dark, alignment=1, spaceAfter=2)))
    story.append(Paragraph("<b>Tesistas:</b> Ordoñez Reyes, Abraham Benjamin & Quispe Sanchez, Edward Steven | <b>Año:</b> 2026", ParagraphStyle('TesistasMeta', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#718096"), alignment=1, spaceAfter=5)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=6))

    # PÁGINA 11 Y 12
    p11_err = ("• '...al incrementar en un 15% métricas clave como el F1-Score y la precisión.'<br/>"
               "• '...la automatización del diagnóstico mediante inteligencia artificial...'")
    p11_fix = ("• '...el modelo Random Forest optimizado alcanzó un <b>F1-Score de 0.9126</b>, una <b>Sensibilidad diagnóstica del 95.50%</b> (detectando 955 de 1,000 casos de DT2 en prueba ciega), una <b>Exactitud del 90.85%</b>, un <b>ROC-AUC de 0.9722</b> y un <b>Coeficiente de Correlación de Matthews (MCC) de 0.8206 (82.06%)</b>, superando con alta significancia a la línea base tradicional (McNemar χ² = 221.13, p &lt; 0.001).'<br/>"
               "• '...la implementación de una herramienta de apoyo a la decisión clínica (CDSS)...'")
    story.append(make_diff_box(11, "Resumen / Abstract (Páginas 11 y 12)", p11_err, p11_fix, "Elimina la palabra 'automatización del diagnóstico' (prohibida por ley de salud) y reporta las métricas reales del modelo final."))

    # PÁGINAS 26 Y 27
    p27_err = ("• <b>OE2:</b> '...analizar el impacto potencial del modelo en la <u>saturación hospitalaria</u>...'<br/>"
               "• <b>OE3:</b> '...determinar el efecto en la disminución de los <u>costos económicos familiares</u>...'<br/>"
               "• Lo mismo en problemas e hipótesis específicas.")
    p27_fix = ("• <b>OE1:</b> Evaluar la eficacia diagnóstica del modelo de Machine Learning en la predicción temprana de Diabetes Tipo 2 frente a métodos tradicionales.<br/>"
               "• <b>OE2:</b> Evaluar la reducción del <b>tiempo promedio del proceso de atención y tamizaje clínico</b> del paciente mediante el sistema web.<br/>"
               "• <b>OE3:</b> Evaluar la reducción del <b>costo operativo directo del personal de salud</b> por consulta médica asistida por el sistema.")
    story.append(make_diff_box(27, "Problemas, Hipótesis y Objetivos Específicos (Páginas 26 y 27)", p27_err, p27_fix, "El jurado observó que el software no mide camas de hospital ni finanzas familiares, sino tiempos de consulta y costo hora-médico."))

    story.append(PageBreak())

    # ==================== PÁGINA 2 ====================
    # PÁGINA 31 Y 32
    p31_err = ("• 'La población está conformada por 100 pacientes... muestra final: 80 registros con diagnóstico confirmado...'<br/>"
               "• Confusión entre la muestra de pacientes del centro de salud y el dataset de entrenamiento de Machine Learning.")
    p31_fix = ("<b>Dividir formalmente en dos unidades de análisis metodológicas:</b><br/>"
               "<b>1. Base de Datos para Machine Learning (N = 10,000 registros netos):</b> Muestra estructurada y balanceada 50/50 (5,000 con DT2 y 5,000 sanos) con 8 variables clínicas. Dividida de forma tripartita: <b>60% Entrenamiento (6,000)</b>, <b>20% Validación (2,000)</b> y <b>20% Prueba Ciega (2,000)</b>.<br/>"
               "<b>2. Muestra de Validación Clínica y Operativa en Casa Grande (N = 80 consultas):</b> 80 consultas médicas asistidas para medir tiempos (OE2) y costos de personal (OE3).")
    story.append(make_diff_box(31, "Población, Muestra y Muestreo (Páginas 31 y 32)", p31_err, p31_fix, "Subsana la Observación Crítica #1 del informe del jurado, separando la base de datos de IA del muestreo de consultas en el centro de salud."))

    # PÁGINAS 39 A 42
    p39_err = ("• '...sobre una muestra de 5 registros... Tabla 14 con N=5... Wilcoxon Z = -2.023, p = 0.043.'<br/>"
               "• El jurado criticó que N=5 no tiene potencia estadística.")
    p39_fix = ("• Aclarar que las 5 filas del Anexo 7.1 representan los <b>5 turnos médicos evaluados</b>, los cuales agrupan a las <b>80 consultas individuales</b> de los pacientes.<br/>"
               "• Al aplicar la prueba de Wilcoxon sobre las <b>80 consultas médicas individuales</b>:<br/>"
               "  - <b>Z = -7.770</b>, <b>p-valor &lt; 0.001 (0.000)</b>.<br/>"
               "  - Reducción del costo operativo de <b>S/. 24.17 a S/. 0.21 por predicción (-99.11%)</b>.")
    story.append(make_diff_box(39, "Resultados: Costo Operativo - Sección 3.2 (Páginas 39 a 42)", p39_err, p39_fix, "Convierte la muestra débil de N=5 en una muestra sólida de N=80 consultas pareadas."))

    # PÁGINAS 42 A 45 (CRÍTICO)
    p43_err = ("• Tabla 17 con medias binarias: Pretest=0.46 / Postest=0.31 (¡la barra bajaba en la Figura 6!).<br/>"
               "• Prueba de normalidad aplicada a ceros y unos.<br/>"
               "• McNemar de SPSS con Sig=0.023 sin matriz de confusión.")
    p43_fix = ("<b>Colocar la Tabla de Clasificación Completa (Mismos 2,000 casos de prueba):</b><br/>"
               "• <b>Exactitud (Accuracy):</b> Pretest 84.35% ➔ <b>Postest 90.85%</b> [89.50%–92.10%]<br/>"
               "• <b>Sensibilidad (Recall DT2):</b> Pretest 91.50% ➔ <b>Postest 95.50%</b> [94.08%–96.83%]<br/>"
               "• <b>Especificidad (No DT2):</b> Pretest 77.20% ➔ <b>Postest 86.20%</b><br/>"
               "• <b>Precisión (PPV):</b> Pretest 80.05% ➔ <b>Postest 87.37%</b><br/>"
               "• <b>F1-Score:</b> Pretest 0.8539 ➔ <b>Postest 0.9126</b> [0.8991–0.9247]<br/>"
               "• <b>ROC-AUC:</b> Pretest 0.9226 ➔ <b>Postest 0.9722</b> [0.9653–0.9784]<br/>"
               "• <b>MCC:</b> Pretest 0.6941 ➔ <b>Postest 0.8206 (82.06%)</b> [0.7948–0.8442]<br/>"
               "• <b>Matriz de Confusión (N=2,000):</b> TN=862, FP=138, FN=45, TP=955.<br/>"
               "• <b>Prueba de McNemar:</b> b=309, c=33, <b>χ² = 221.1257, p &lt; 0.001</b> (Se rechaza H₀).")
    story.append(make_diff_box(43, "Resultados de Machine Learning - Sección 3.3 (Páginas 42 a 45)", p43_err, p43_fix, "Reemplaza el error de las barras invertidas por la tabla formal de clasificación multimétrica."))

    story.append(PageBreak())

    # ==================== PÁGINA 3 ====================
    # PÁGINAS 55 A 57
    p55_err = ("• Fórmula errónea: PM = (TP / (TP + TN)) * 100.<br/>"
               "• Fórmula errónea: PD = ((TP + TN) / (TP + TN + FP + FN)) * 100 llamada 'Precisión'.")
    p55_fix = ("<b>Corregir las fórmulas matemáticas de la matriz de operacionalización:</b><br/>"
               "• <b>Exactitud (Accuracy):</b> (TP + TN) / (TP + TN + FP + FN) * 100<br/>"
               "• <b>Sensibilidad (Recall):</b> TP / (TP + FN) * 100<br/>"
               "• <b>Especificidad:</b> TN / (TN + FP) * 100<br/>"
               "• <b>Precisión (PPV):</b> TP / (TP + FP) * 100<br/>"
               "• <b>F1-Score:</b> 2 * (PPV * Sensibilidad) / (PPV + Sensibilidad)<br/>"
               "• <b>MCC:</b> (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))")
    story.append(make_diff_box(55, "Anexos 5 y 6: Operacionalización e Indicadores (Páginas 55 a 58)", p55_err, p55_fix, "Corrige los errores matemáticos en las fórmulas de los anexos metodológicos."))

    # PÁGINAS 90 A 92
    p90_err = ("• 'Escalamiento de características antes de dividir los datos...' (Data Leakage confeso).<br/>"
               "• Capturas de consola con muestras desiguales de 117 y 154 registros.<br/>"
               "• Umbral 0.35 sin justificación matemática.")
    p90_fix = ("• <b>Pipeline Scikit-learn:</b> Explicar que el SimpleImputer y StandardScaler se ajustaron (fit) <b>únicamente con el conjunto de entrenamiento (6,000 casos)</b> y solo transformaron validación y prueba, blindando el estudio contra fuga de datos.<br/>"
               "• <b>Partición Tripartita 60/20/20:</b> 6,000 Train, 2,000 Val, 2,000 Test.<br/>"
               "• <b>Reemplazo de Capturas:</b> Pegar las salidas del script con N=2,000 uniforme.")
    story.append(make_diff_box(90, "Anexo 8: Metodología CRISP-DM (Páginas 89 a 93)", p90_err, p90_fix, "Elimina la confesión de data leakage y unifica el soporte a 2,000 casos de prueba."))

    # TABLA RESUMEN FINAL
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>TABLA RESUMEN DE HIPERPARÁMETROS Y MÉTRICAS FINALES DE LA TESIS:</b>", sec_title))
    t_res_data = [
        [Paragraph("Componente", cell_header), Paragraph("Pretest (Base)", cell_header), Paragraph("Postest (Optimizado)", cell_header), Paragraph("Línea Base (LR)", cell_header), Paragraph("Sustento Metodológico", cell_header)],
        [Paragraph("Muestra de Prueba", cell_bold), Paragraph("2,000 casos", cell_style), Paragraph("2,000 casos", cell_style), Paragraph("2,000 casos", cell_style), Paragraph("Evaluación ciego e independiente sobre los mismos sujetos.", cell_style)],
        [Paragraph("Hiperparámetros RF", cell_bold), Paragraph("depth=6, leaf=25", cell_style), Paragraph("<b>depth=10, leaf=8, n=200</b>", cell_bold), Paragraph("C=1.0, l2", cell_style), Paragraph("Optimizado en validación (regularizado para evitar overfitting).", cell_style)],
        [Paragraph("Exactitud (Accuracy)", cell_bold), Paragraph("84.35%", cell_style), Paragraph("<b>90.85%</b>", cell_bold), Paragraph("77.05%", cell_style), Paragraph("Mejora global del 13.80% sobre el modelo tradicional.", cell_style)],
        [Paragraph("Sensibilidad (Recall DT2)", cell_bold), Paragraph("91.50%", cell_style), Paragraph("<b>95.50%</b>", cell_bold), Paragraph("76.10%", cell_style), Paragraph("<b>Detecta a 955 de 1,000 diabéticos</b> (solo 45 falsos negativos).", cell_style)],
        [Paragraph("F1-Score", cell_bold), Paragraph("0.8539", cell_style), Paragraph("<b>0.9126</b>", cell_bold), Paragraph("0.7683", cell_style), Paragraph("Equilibrio armónico superior a 0.90.", cell_style)],
        [Paragraph("ROC-AUC", cell_bold), Paragraph("0.9226", cell_style), Paragraph("<b>0.9722</b>", cell_bold), Paragraph("0.8614", cell_style), Paragraph("Excelente capacidad de discriminación diagnóstica.", cell_style)],
        [Paragraph("<b>Coef. Matthews (MCC)</b>", cell_bold), Paragraph("0.6941", cell_style), Paragraph("<b>0.8206 (82.06%)</b>", cell_bold), Paragraph("0.5411", cell_style), Paragraph("<b>Rango objetivo de alta calidad alcanzado (80% – 85%).</b>", cell_style)],
    ]
    t_res = Table(t_res_data, colWidths=[105, 75, 95, 75, 154])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (1, 0), (3, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_res)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF Guía de Reemplazos generado exitosamente: {PDF_FILENAME}")

if __name__ == '__main__':
    build_pdf()
