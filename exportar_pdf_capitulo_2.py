import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "Guia_Subsanacion_Capitulo_II_Materiales_y_Metodos.pdf"

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
            self.drawString(54, 750, "UNIVERSIDAD NACIONAL DE TRUJILLO — SUBSANACIÓN CAPÍTULO II (MÉTODOS)")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawRightString(558, 750, "Punto 7 del Informe de Revisión")
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
    c_secondary = colors.HexColor("#2B6CB0") # Azul intermedio
    c_red_bg = colors.HexColor("#FFF5F5")
    c_red_border = colors.HexColor("#FEB2B2")
    c_red_text = colors.HexColor("#9B2C2C")
    c_green_bg = colors.HexColor("#F0FFF4")
    c_green_border = colors.HexColor("#9AE6B4")
    c_green_text = colors.HexColor("#22543D")
    c_dark = colors.HexColor("#2D3748")
    c_border = colors.HexColor("#CBD5E0")
    c_light_bg = colors.HexColor("#F7FAFC")

    title_style = ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=c_primary, alignment=1, spaceAfter=2)
    subtitle_style = ParagraphStyle('DocSubTitle', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=c_secondary, alignment=1, spaceAfter=4)
    sec_title = ParagraphStyle('SecTitle', fontName='Helvetica-Bold', fontSize=9, leading=11.5, textColor=c_primary, spaceBefore=5, spaceAfter=2.5, keepWithNext=True)
    body_style = ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=7.5, leading=10, textColor=c_dark, spaceAfter=2)
    
    del_title = ParagraphStyle('DelTitle', fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=c_red_text)
    del_text = ParagraphStyle('DelText', fontName='Helvetica', fontSize=6.8, leading=8.8, textColor=c_red_text)
    
    ins_title = ParagraphStyle('InsTitle', fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=c_green_text)
    ins_text = ParagraphStyle('InsText', fontName='Helvetica', fontSize=6.8, leading=8.8, textColor=c_green_text)
    
    cell_style = ParagraphStyle('CellText', fontName='Helvetica', fontSize=6.5, leading=8.5, textColor=c_dark)
    cell_bold = ParagraphStyle('CellBold', fontName='Helvetica-Bold', fontSize=6.5, leading=8.5, textColor=c_dark)
    cell_header = ParagraphStyle('CellHeader', fontName='Helvetica-Bold', fontSize=6.8, leading=8.5, textColor=colors.white)

    story = []

    def make_diff_card(subpoint_title, error_content, replace_content, note_content):
        header = f"<b>📍 {subpoint_title.upper()}</b>"
        t_data = [
            [
                Paragraph("<b>❌ OBSERVACIÓN DEL JURADO / TEXTO ACTUAL:</b>", del_title),
                Paragraph("<b>✅ TEXTO Y ESTRUCTURA DE REEMPLAZO (A COPIAR EN WORD):</b>", ins_title)
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
        
        note_p = Paragraph(f"<b>💡 Sustento Técnico para el Jurado:</b> {note_content}", ParagraphStyle('Note', fontName='Helvetica-Oblique', fontSize=6.5, leading=8, textColor=colors.HexColor("#4A5568")))
        
        return KeepTogether([
            Paragraph(header, sec_title),
            t,
            Spacer(1, 1),
            note_p,
            Spacer(1, 3.5)
        ])

    # ==================== PÁGINA 1 ====================
    story.append(Paragraph("UNIVERSIDAD NACIONAL DE TRUJILLO", title_style))
    story.append(Paragraph("FACULTAD DE INGENIERÍA — ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS", subtitle_style))
    story.append(Paragraph("<b>GUÍA DE SUBSANACIÓN OFICIAL — CAPÍTULO II: MATERIALES Y MÉTODOS</b>", ParagraphStyle('ReportName', fontName='Helvetica-Bold', fontSize=9.5, leading=11.5, textColor=c_primary, alignment=1, spaceAfter=2)))
    story.append(Paragraph("<b>Respuesta y Subsanación al Punto 7 del Informe de Revisión Técnica (Páginas 7 a 9)</b>", ParagraphStyle('SubTesis', fontName='Helvetica-Oblique', fontSize=7.5, leading=9.5, textColor=c_secondary, alignment=1, spaceAfter=2)))
    story.append(Paragraph("<b>Tesistas:</b> Ordoñez Reyes Abraham Benjamin & Quispe Sanchez Edward Steven | <b>Año:</b> 2026", ParagraphStyle('Tesistas', fontName='Helvetica', fontSize=7, leading=8.5, textColor=colors.HexColor("#718096"), alignment=1, spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=5))

    # 7.1. CLASIFICACIÓN METODOLÓGICA
    c1_err = ("• En la Tesis (Pág. 30-31) se clasifica todo como un único diseño preexperimental simple O₁–X–O₂.<br/>"
              "• El jurado observó: <i>'El trabajo mezcla evaluación de proceso clínico, evaluación de software y evaluación de un clasificador. Requieren un diseño y unidades de análisis claramente diferenciados.'</i>")
    c1_rep = ("<b>2.3. Métodos - Clasificación Metodológica:</b><br/>"
              "• <b>Tipo de investigación:</b> Aplicada, con enfoque cuantitativo y alcance explicativo-evaluativo.<br/>"
              "• <b>Diseño Metodológico Diferenciado:</b><br/>"
              "  1. <i>Para los Objetivos 2 y 3 (Tiempo y Costo):</i> Diseño preexperimental de un solo grupo con medición antes y después (O₁–X–O₂) aplicado sobre la muestra clínica de consultas en Casa Grande.<br/>"
              "  2. <i>Para el Objetivo 1 (Machine Learning):</i> Estudio experimental de desempeño diagnóstico predictivo con partición estratificada independiente (Train/Val/Test) y contrastación pareada contra una línea base tradicional (Regresión Logística).")
    story.append(make_diff_card("7.1. Clasificación Metodológica (Página 30 de Tesis / Pág. 7 del Informe)", c1_err, c1_rep, "Separa formalmente la evaluación del proceso asistencial (tiempo/costo) de la evaluación matemática del clasificador de IA."))

    # 7.2. POBLACIÓN, MUESTRA Y FUENTE DE DATOS
    c2_err = ("• En la Tesis (Pág. 31-32) se declara población de 100 pacientes y muestra finita de 80, pero los anexos de ML tenían 117 y 154 casos.<br/>"
              "• El jurado observó: <i>'La fórmula de población finita no es el fundamento para decidir el tamaño de un dataset de ML. Debe declararse la fuente real sin ambigüedades.'</i>")
    c2_rep = ("<b>2.3.4. Población, Muestra y Muestreo:</b><br/>"
              "<b>1. Base de Datos para Machine Learning (N = 10,000 registros netos):</b><br/>"
              "• <i>Población y Muestra:</i> Base estructurada de 10,000 registros clínicos estandarizados con las 8 variables biológicas de tamizaje (Glucosa, Presión, Insulina, IMC, Grosor de piel, Embarazos, Pedigree, Edad) con balance simétrico 50/50 (5,000 con DT2 y 5,000 sanos).<br/>"
              "• <i>Criterios de Inclusión:</i> Mayores de 18 años con datos completos de triaje.<br/>"
              "• <i>Criterios de Exclusión:</i> Valores fisiológicos nulos tratados con imputación por mediana en pipeline.<br/>"
              "<b>2. Muestra de Validación Clínica en Casa Grande (N = 80 consultas):</b><br/>"
              "• 80 consultas médicas presenciales asistidas para evaluar tiempos de atención (OE2) y costo del personal de salud (OE3).")
    story.append(make_diff_card("7.2. Población, Muestra y Fuente de Datos (Páginas 31 y 32 / Pág. 8 del Informe)", c2_err, c2_rep, "Subsana la inconsistencia de muestras declarando la base de 10,000 datos para ML y las 80 consultas para validación en campo."))

    # 7.3. MUESTREO Y PARTICIÓN TRIPARTITA
    c3_err = ("• En la Tesis (Pág. 90) se decía 'escalamiento antes de dividir los datos' (Data Leakage) y solo se dividía 80/20.<br/>"
              "• El jurado observó: <i>'Separar entrenamiento, validación y prueba. Mantener prueba intacta y no seleccionar umbral en prueba.'</i>")
    c3_rep = ("<b>2.3.4.3. Partición Estratificada y Control de Fuga de Información:</b><br/>"
              "• <b>Entrenamiento (60% - 6,000 casos):</b> Ajuste exclusivo del imputador, escalador y árboles.<br/>"
              "• <b>Validación (20% - 2,000 casos):</b> Calibración de hiperparámetros (Random Forest optimizado: depth=10, leaf=8, n=200) y confirmación del umbral canónico de 0.50.<br/>"
              "• <b>Prueba Ciega (20% - 2,000 casos):</b> Conjunto independiente no visto durante el modelado, reservado para métricas finales y contraste de McNemar.<br/>"
              "• <b>Pipeline Scikit-learn:</b> Preprocesamiento ajustado (fit) solo con Train.")
    story.append(make_diff_card("7.3. Muestreo y Partición de Datos (Página 32 / Pág. 8 del Informe)", c3_err, c3_rep, "Demuestra cero fuga de información al ajustar las transformaciones únicamente sobre los 6,000 casos de entrenamiento."))

    story.append(PageBreak())

    # ==================== PÁGINA 2 ====================
    # 7.4. VARIABLES Y OPERACIONALIZACIÓN
    story.append(Paragraph("<b>7.4. VARIABLES Y MATRIZ DE OPERACIONALIZACIÓN (PÁGINAS 32–33 Y ANEXO 5 / PÁG. 8 DEL INFORME)</b>", sec_title))
    
    t_oper_data = [
        [Paragraph("Variable", cell_header), Paragraph("Definición Conceptual", cell_header), Paragraph("Dimensión", cell_header), Paragraph("Indicadores", cell_header), Paragraph("Escala / Unidad", cell_header)],
        [Paragraph("<b>Independiente:</b><br/>Sistema CDSS con Machine Learning", cell_bold), Paragraph("Sistema de soporte a la decisión clínica basado en Random Forest optimizado que procesa 8 predictores biológicos.", cell_style), Paragraph("Intervención tecnológica", cell_style), Paragraph("Condición de evaluación (Método tradicional vs. Software ML).", cell_style), Paragraph("Nominal dicotómica", cell_style)],
        [Paragraph("<b>Dependiente 1:</b><br/>Eficacia Diagnóstica Temprana", cell_bold), Paragraph("Capacidad del modelo para clasificar pacientes diabéticos y sanos.", cell_style), Paragraph("Discriminación diagnóstica", cell_style), Paragraph("• Sensibilidad (Recall)<br/>• Exactitud (Accuracy)<br/>• F1-Score<br/>• ROC-AUC<br/>• Coeficiente Matthews (MCC)", cell_style), Paragraph("Razón:<br/>Porcentaje (%) y decimal [0 – 1]", cell_style)],
        [Paragraph("<b>Dependiente 2:</b><br/>Eficiencia Temporal", cell_bold), Paragraph("Duración del proceso de atención y tamizaje clínico del paciente.", cell_style), Paragraph("Desempeño temporal", cell_style), Paragraph("Tiempo promedio de atención por consulta médica (TPDP).", cell_style), Paragraph("Razón:<br/>Minutos y segundos", cell_style)],
        [Paragraph("<b>Dependiente 3:</b><br/>Eficiencia Económica", cell_bold), Paragraph("Costo monetario directo derivado del tiempo del personal de salud.", cell_style), Paragraph("Desempeño financiero", cell_style), Paragraph("Costo operativo por consulta médica asistida (COPG).", cell_style), Paragraph("Razón:<br/>Soles (PEN) por consulta", cell_style)],
    ]
    t_oper = Table(t_oper_data, colWidths=[95, 115, 80, 120, 94])
    t_oper.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_oper)
    story.append(Spacer(1, 4))

    # 7.5. TÉCNICAS, INSTRUMENTOS Y FÓRMULAS
    c5_err = ("• En la Tesis (Págs. 33, 55 y 72) se definía PM = TP/(TP+TN) y PD = (TP+TN)/(TP+TN+FP+FN) llamada 'Precisión'.<br/>"
              "• Las fichas repetían métricas globales por fila de paciente.<br/>"
              "• El jurado observó: <i>'Corregir fórmulas matemáticas y calcular métricas globales a partir de la matriz de confusión.'</i>")
    c5_rep = ("<b>2.3.6. Técnicas e Instrumentos de Medición:</b><br/>"
              "• <b>Instrumento 1 (Costo):</b> Ficha de costos del personal de salud basada en sueldos médicos reales (S/. 6,000 y S/. 8,000 para 192 h/mes).<br/>"
              "• <b>Instrumento 2 (Tiempo):</b> Ficha de tiempos de atención con límites estandarizados (Inicio: carga de datos de triaje; Fin: confirmación de predicción).<br/>"
              "• <b>Instrumento 3 (Desempeño ML):</b> Ficha de registro pareado individual (TP, TN, FP, FN). Las métricas se calculan una sola vez a nivel global:<br/>"
              "  - <i>Exactitud (Accuracy):</i> (TP+TN)/(TP+TN+FP+FN) * 100 = <b>90.85%</b><br/>"
              "  - <i>Sensibilidad (Recall):</i> TP/(TP+FN) * 100 = <b>95.50%</b><br/>"
              "  - <i>Precisión (PPV):</i> TP/(TP+FP) * 100 = <b>87.37%</b><br/>"
              "  - <i>F1-Score:</i> 2*(PPV*Sensibilidad)/(PPV+Sensibilidad) = <b>0.9126</b><br/>"
              "  - <i>MCC:</i> (TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = <b>0.8206 (82.06%)</b>")
    story.append(make_diff_card("7.5. Técnicas, Instrumentos y Validación (Páginas 33–34 y Anexo 7)", c5_err, c5_rep, "Corrige los errores en las fórmulas de los anexos y establece el cálculo unificado a partir de la matriz de confusión."))

    # 7.6. CONSIDERACIONES ÉTICAS Y DE SEGURIDAD
    c6_err = ("• En la Tesis (Pág. 35) la sección ética tenía solo 1 párrafo genérico.<br/>"
              "• Se presentaba el sistema como 'automatización del diagnóstico'.<br/>"
              "• El jurado observó: <i>'Insuficiente para salud. Documentar anonimización, definir como CDSS y no enviar datos personales a Gemini.'</i>")
    c6_rep = ("<b>2.3.9. Consideraciones Éticas y Cumplimiento Normativo:</b><br/>"
              "• <b>Ley N° 29733 (Protección de Datos Personales):</b> Disociación y anonimización irreversible mediante identificadores cifrados (PAC-001).<br/>"
              "• <b>Ley N° 26842 (Ley General de Salud):</b> El sistema se define como <b>Sistema de Soporte a la Decisión Clínica (CDSS)</b>; la responsabilidad diagnóstica y prescripción final es siempre del médico colegiado.<br/>"
              "• <b>Uso Seguro de Google Gemini:</b> Se envían exclusivamente los 8 valores numéricos anonimizados para explicabilidad clínica asistida, sin datos personales.")
    story.append(make_diff_card("7.6. Consideraciones Éticas (Página 35 / Pág. 9 del Informe)", c6_err, c6_rep, "Cumple con las leyes de salud y protección de datos peruanas, blindando el proyecto ante cualquier cuestionamiento bioético."))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF de Subsanación Capítulo II generado exitosamente: {PDF_FILENAME}")

if __name__ == '__main__':
    build_pdf()
