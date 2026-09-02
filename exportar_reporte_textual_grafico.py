import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "Reporte_Oficial_Reemplazos_Textuales_y_Graficos.pdf"

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
            self.drawString(54, 750, "UNIVERSIDAD NACIONAL DE TRUJILLO — GUÍA DE REEMPLAZO TEXTUAL Y GRÁFICO")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawRightString(558, 750, "Escuela de Ingeniería de Sistemas")
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
        self.drawString(54, 28, "Tesis: Predicción Temprana de Diabetes Tipo 2 — Ordoñez & Quispe (2026)")
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

    c_primary = colors.HexColor("#1A365D")   # Azul marino
    c_secondary = colors.HexColor("#2B6CB0") # Azul
    c_red_bg = colors.HexColor("#FFF5F5")
    c_red_border = colors.HexColor("#FEB2B2")
    c_red_text = colors.HexColor("#9B2C2C")
    c_green_bg = colors.HexColor("#F0FFF4")
    c_green_border = colors.HexColor("#9AE6B4")
    c_green_text = colors.HexColor("#22543D")
    c_dark = colors.HexColor("#2D3748")
    c_border = colors.HexColor("#CBD5E0")
    c_light_bg = colors.HexColor("#F7FAFC")

    title_style = ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=c_primary, alignment=1, spaceAfter=2)
    subtitle_style = ParagraphStyle('DocSubTitle', fontName='Helvetica-Bold', fontSize=9, leading=11, textColor=c_secondary, alignment=1, spaceAfter=4)
    sec_title = ParagraphStyle('SecTitle', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=c_primary, spaceBefore=6, spaceAfter=3, keepWithNext=True)
    body_style = ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=7.8, leading=10.5, textColor=c_dark, spaceAfter=3)
    
    del_title = ParagraphStyle('DelTitle', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=c_red_text)
    del_text = ParagraphStyle('DelText', fontName='Helvetica', fontSize=7, leading=9, textColor=c_red_text)
    
    ins_title = ParagraphStyle('InsTitle', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=c_green_text)
    ins_text = ParagraphStyle('InsText', fontName='Helvetica', fontSize=7, leading=9, textColor=c_green_text)
    
    cell_style = ParagraphStyle('CellText', fontName='Helvetica', fontSize=7, leading=8.5, textColor=c_dark)
    cell_bold = ParagraphStyle('CellBold', fontName='Helvetica-Bold', fontSize=7, leading=8.5, textColor=c_dark)
    cell_header = ParagraphStyle('CellHeader', fontName='Helvetica-Bold', fontSize=7, leading=8.5, textColor=colors.white)

    story = []

    def make_diff_card(page_label, section_label, error_content, replace_content, note_content):
        header = f"<b>📍 {page_label.upper()} | {section_label.upper()}</b>"
        t_data = [
            [
                Paragraph("<b>❌ TEXTO / ERROR ACTUAL EN TU TESIS (A ELIMINAR):</b>", del_title),
                Paragraph("<b>✅ TEXTO EXACTO DE REEMPLAZO (A PEGAR EN WORD):</b>", ins_title)
            ],
            [
                Paragraph(error_content, del_text),
                Paragraph(replace_content, ins_text)
            ]
        ]
        t = Table(t_data, colWidths=[245, 259])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), c_red_bg),
            ('BACKGROUND', (1, 0), (1, -1), c_green_bg),
            ('BOX', (0, 0), (0, -1), 1, c_red_border),
            ('BOX', (1, 0), (1, -1), 1, c_green_border),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        note_p = Paragraph(f"<b>💡 Sustento Técnico para el Jurado:</b> {note_content}", ParagraphStyle('Note', fontName='Helvetica-Oblique', fontSize=6.8, leading=8.5, textColor=colors.HexColor("#4A5568")))
        
        return KeepTogether([
            Paragraph(header, sec_title),
            t,
            Spacer(1, 1),
            note_p,
            Spacer(1, 4)
        ])

    # ==================== PÁGINA 1 ====================
    story.append(Paragraph("UNIVERSIDAD NACIONAL DE TRUJILLO", title_style))
    story.append(Paragraph("FACULTAD DE INGENIERÍA — ESCUELA DE INGENIERÍA DE SISTEMAS", subtitle_style))
    story.append(Paragraph("<b>GUÍA OFICIAL DE REEMPLAZOS TEXTUALES, ESTADÍSTICOS Y GRÁFICOS</b>", ParagraphStyle('ReportName', fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=c_primary, alignment=1, spaceAfter=2)))
    story.append(Paragraph("<b>Tesis:</b> <i>Predicción Temprana de Diabetes Tipo 2 aplicando un Modelo en Machine Learning en Centro de Salud Casa Grande</i>", ParagraphStyle('SubTesis', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=c_dark, alignment=1, spaceAfter=1)))
    story.append(Paragraph("<b>Autores:</b> Ordoñez Reyes Abraham Benjamin & Quispe Sanchez Edward Steven | <b>Año:</b> 2026", ParagraphStyle('Tesistas', fontName='Helvetica', fontSize=7, leading=8.5, textColor=colors.HexColor("#718096"), alignment=1, spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=5))

    # ACLARACIÓN METODOLÓGICA DE PRUEBAS ESTADÍSTICAS
    p_stat_title = Paragraph("<b>1. MAPA Y JUSTIFICACIÓN DE PRUEBAS ESTADÍSTICAS APLICADAS (10,000 DATOS VS. MUESTRA CLÍNICA)</b>", sec_title)
    p_stat_desc = ("Para garantizar máxima rigurosidad, se aclara qué prueba estadística corresponde exactamente a cada objetivo:<br/>"
                   "• <b>Objetivo 1 (Machine Learning - 10,000 datos / 2,000 prueba):</b> Al evaluar aciertos y errores binarios (Acierta=1, Falla=0), <b>NO se aplica normalidad (Shapiro-Wilk o Kolmogorov) a ceros y unos</b> porque no son variables continuas. La prueba inferencial estándar de oro es la <b>Prueba de McNemar para muestras pareadas</b> ($\chi^2 = 221.13, p &lt; 0.001$), complementada con <b>Bootstrap al 95% de confianza (1,000 iteraciones)</b> para estimar la variabilidad de métricas continuas (Accuracy, Recall, F1, ROC-AUC y MCC).<br/>"
                   "• <b>Objetivos 2 y 3 (Tiempo y Costo - Muestra Clínica N = 80):</b> Como el tiempo en minutos y el costo en soles son variables cuantitativas continuas, se aplica primero <b>Shapiro-Wilk ($p &lt; 0.05$, no normal)</b> y luego la <b>Prueba de Rangos con Signo de Wilcoxon ($Z = -7.770, p &lt; 0.001$)</b> para comparar Pretest vs. Postest.")
    story.append(KeepTogether([p_stat_title, Paragraph(p_stat_desc, body_style), Spacer(1, 4)]))

    # PÁGINAS 11 Y 12
    c1_err = ("• '...al incrementar en un 15% métricas clave como el F1-Score y la precisión.'<br/>"
              "• '...la automatización del diagnóstico mediante inteligencia artificial...'")
    c1_rep = ("• '...el modelo Random Forest optimizado alcanzó un <b>F1-Score de 0.9126</b>, una <b>Sensibilidad diagnóstica del 95.50%</b> (detectando a 955 de 1,000 pacientes con DT2 en prueba ciega), una <b>Exactitud del 90.85%</b>, un <b>ROC-AUC de 0.9722</b> y un <b>MCC de 0.8206 (82.06%)</b>, superando a la línea base tradicional (McNemar χ² = 221.13, p &lt; 0.001).'<br/>"
              "• '...la implementación de una <b>herramienta de apoyo a la decisión clínica (CDSS)</b>...'")
    story.append(make_diff_card("Páginas 11 y 12", "Resumen (Español) y Abstract (Inglés)", c1_err, c1_rep, "Elimina la palabra 'automatización del diagnóstico' (prohibida por ley) y reporta las métricas reales del modelo final."))

    # PÁGINAS 26 Y 27
    c2_err = ("• <b>OE2:</b> '...analizar el impacto en la <u>saturación hospitalaria</u>...'<br/>"
              "• <b>OE3:</b> '...determinar el efecto en los <u>costos económicos familiares</u>...'<br/>"
              "• Lo mismo en problemas e hipótesis específicas.")
    c2_rep = ("• <b>OE1:</b> Evaluar la eficacia diagnóstica del modelo de Machine Learning en la predicción temprana de Diabetes Tipo 2 frente a métodos tradicionales.<br/>"
              "• <b>OE2:</b> Evaluar la reducción del <b>tiempo promedio del proceso de atención y tamizaje clínico</b> del paciente mediante el sistema web.<br/>"
              "• <b>OE3:</b> Evaluar la reducción del <b>costo operativo directo del personal de salud</b> por consulta médica generada mediante el sistema web.")
    story.append(make_diff_card("Páginas 26 y 27", "Problemas, Hipótesis y Objetivos Específicos", c2_err, c2_rep, "El software no mide camas de hospital ni finanzas de familias; mide el cronómetro de la consulta y el costo de la hora médica."))

    story.append(PageBreak())

    # ==================== PÁGINA 2 ====================
    # PÁGINAS 31 Y 32
    c3_err = ("• 'La población está conformada por 100 pacientes... muestra final: 80 registros con diagnóstico confirmado...'<br/>"
              "• Confusión entre la muestra de pacientes locales y el dataset de entrenamiento de Machine Learning.")
    c3_rep = ("<b>Dividir formalmente en dos unidades de análisis metodológicas:</b><br/>"
              "<b>1. Base de Datos para Machine Learning (N = 10,000 registros netos):</b> Base clínica estructurada y balanceada 50/50 (5,000 con DT2 y 5,000 sanos) con 8 variables biológicas. Se implementó una <b>partición tripartita</b>: 60% Entrenamiento (6,000), 20% Validación (2,000) y 20% Prueba Ciega (2,000).<br/>"
              "<b>2. Muestra de Validación Clínica en Casa Grande (N = 80 consultas):</b> 80 consultas médicas asistidas para medir tiempos de atención (OE2) y costos de personal (OE3).")
    story.append(make_diff_card("Páginas 31 y 32", "Capítulo II: Población, Muestra y Muestreo", c3_err, c3_rep, "Subsana la Observación Crítica #1 del jurado, separando la base de datos de IA del muestreo de consultas en el centro de salud."))

    # PÁGINAS 39 A 42
    c4_err = ("• '...sobre una muestra de 5 registros... Tabla 14 con N=5... Wilcoxon Z = -2.023, p = 0.043.'<br/>"
              "• El jurado observó que N=5 no tiene potencia estadística.")
    c4_rep = ("• Aclarar que las 5 filas del Anexo 7.1 representan los <b>5 turnos médicos evaluados</b>, los cuales agrupan a las <b>80 consultas médicas individuales</b>.<br/>"
              "• Al contrastar las <b>80 consultas pareadas mediante la prueba de Wilcoxon</b>:<br/>"
              "  - <b>Z = -7.770</b>, <b>p-valor &lt; 0.001 (0.000)</b>.<br/>"
              "  - Reducción del costo operativo de <b>S/. 24.17 a S/. 0.21 por predicción (-99.11%)</b>.")
    story.append(make_diff_card("Páginas 39 a 42", "Resultados: Costo Operativo Directo (Sección 3.2)", c4_err, c4_rep, "Convierte la muestra débil de N=5 en una muestra robusta con alta potencia estadística de N=80 consultas pareadas."))

    # PÁGINAS 42 A 45 (CRÍTICO)
    c5_err = ("• <b>Tabla 17 (Pág. 43):</b> Medias binarias PRETEST=0.46 / POSTEST=0.31.<br/>"
              "• <b>Figura 6 (Pág. 43):</b> Barra de postest más baja que pretest.<br/>"
              "• <b>Tabla 18 (Pág. 44):</b> Prueba de normalidad sobre ceros y unos.<br/>"
              "• <b>Tabla 19 (Pág. 44):</b> McNemar genérico de SPSS con Sig=0.023.")
    c5_rep = ("<b>Reemplazar la Sección 3.3 con la Tabla Multimétrica (N = 2,000 casos ciegos):</b><br/>"
              "• <b>Exactitud (Accuracy):</b> Pretest 84.35% ➔ <b>Postest 90.85%</b> [89.50%–92.10%]<br/>"
              "• <b>Sensibilidad (Recall DT2):</b> Pretest 91.50% ➔ <b>Postest 95.50%</b> [94.08%–96.83%]<br/>"
              "• <b>Especificidad (No DT2):</b> Pretest 77.20% ➔ <b>Postest 86.20%</b><br/>"
              "• <b>Precisión (PPV):</b> Pretest 80.05% ➔ <b>Postest 87.37%</b><br/>"
              "• <b>F1-Score:</b> Pretest 0.8539 ➔ <b>Postest 0.9126</b> [0.8991–0.9247]<br/>"
              "• <b>ROC-AUC:</b> Pretest 0.9226 ➔ <b>Postest 0.9722</b> [0.9653–0.9784]<br/>"
              "• <b>MCC:</b> Pretest 0.6941 ➔ <b>Postest 0.8206 (82.06%)</b> [0.7948–0.8442]<br/>"
              "• <b>Matriz de Confusión:</b> TN=862, FP=138, FN=45, TP=955.<br/>"
              "• <b>Prueba de McNemar:</b> b=309, c=33, <b>χ² = 221.1257, p &lt; 0.001</b> (Rechazo de H₀).")
    story.append(make_diff_card("Páginas 42 a 45", "Resultados de Machine Learning (Sección 3.3)", c5_err, c5_rep, "Elimina el error de las barras invertidas y presenta la tabla formal de clasificación multimétrica con McNemar pareado."))

    story.append(PageBreak())

    # ==================== PÁGINA 3 ====================
    # REEMPLAZO GRÁFICO OFICIAL
    story.append(Paragraph("<b>2. REEMPLAZO GRÁFICO OFICIAL: FIGURAS A PEGAR EN WORD</b>", sec_title))
    
    # Imagen 1 y 2 en paralelo
    img_conf = Image('matriz_confusion.png', width=210, height=170)
    img_comp = Image('grafico_comparativo_metricas_ml.png', width=285, height=170)
    
    t_imgs = Table([
        [Paragraph("<b>Figura A: Matriz de Confusión (Reemplaza Figura de Pág. 92)</b>", ParagraphStyle('Cap1', fontName='Helvetica-Bold', fontSize=7, textColor=c_primary, alignment=1)),
         Paragraph("<b>Figura B: Comparativo Multimétrico (Reemplaza Figura 6 de Pág. 43)</b>", ParagraphStyle('Cap2', fontName='Helvetica-Bold', fontSize=7, textColor=c_primary, alignment=1))],
        [img_conf, img_comp],
        [Paragraph("<i>Muestra N = 2,000 casos ciegos (TN=862, FP=138, FN=45, TP=955). Detecta a 955 de 1,000 diabéticos.</i>", ParagraphStyle('Sub1', fontName='Helvetica', fontSize=6.5, textColor=c_dark, alignment=1)),
         Paragraph("<i>Pretest vs. Postest vs. Línea Base en Accuracy, Recall, Especificidad, F1, AUC y MCC (82.1%).</i>", ParagraphStyle('Sub2', fontName='Helvetica', fontSize=6.5, textColor=c_dark, alignment=1))]
    ], colWidths=[220, 284])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_imgs)
    story.append(Spacer(1, 4))

    # Imagen 3: Tiempo y Costo
    img_tc = Image('grafico_tiempo_costo_wilcoxon.png', width=450, height=140)
    t_tc = Table([
        [Paragraph("<b>Figura C: Impacto Operativo en Tiempo (OE2) y Costo (OE3) — Muestra Clínica N = 80 (Reemplaza Figuras 4 y 5 de Págs. 37 y 40)</b>", ParagraphStyle('Cap3', fontName='Helvetica-Bold', fontSize=7.5, textColor=c_primary, alignment=1))],
        [img_tc],
        [Paragraph("<i>Reducción del 99.09% en tiempo (38.58 min ➔ 0.35 min) y 99.11% en costo (S/. 24.17 ➔ S/. 0.21) validado con Wilcoxon (Z = -7.770, p &lt; 0.001).</i>", ParagraphStyle('Sub3', fontName='Helvetica', fontSize=6.8, textColor=c_dark, alignment=1))]
    ], colWidths=[504])
    t_tc.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_tc)
    story.append(Spacer(1, 4))

    # PÁGINAS 89 A 93 (ANEXO 8 CRISP-DM)
    c8_err = ("• 'Escalamiento de características antes de dividir los datos...' (Data Leakage confeso).<br/>"
              "• Capturas de consola con muestras desiguales de 117 y 154 registros.")
    c8_rep = ("• <b>Pipeline Scikit-learn:</b> Explicar que el SimpleImputer y StandardScaler se ajustaron (fit) <b>únicamente con el conjunto de entrenamiento (6,000 casos)</b> y solo transformaron validación y prueba, blindando el estudio contra fuga de datos.<br/>"
              "• <b>Partición Tripartita 60/20/20:</b> 6,000 Train, 2,000 Val, 2,000 Test.<br/>"
              "• <b>Reemplazo de Capturas:</b> Pegar las salidas del script con N=2,000 uniforme.")
    story.append(make_diff_card("Páginas 89 a 93", "Anexo 8: Metodología CRISP-DM (Pipeline y Modelado)", c8_err, c8_rep, "Elimina la confesión de data leakage y unifica el soporte a 2,000 casos de prueba."))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF Oficial de Reemplazos Textuales y Gráficos generado: {PDF_FILENAME}")

if __name__ == '__main__':
    build_pdf()
