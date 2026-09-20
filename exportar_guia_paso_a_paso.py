import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "Guia_Paso_a_Paso_Modificaciones_Tesis.pdf"

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
            self.drawString(54, 750, "GUÍA PRÁCTICA PASO A PASO: REESTRUCTURACIÓN Y COHERENCIA DE TESIS")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawRightString(558, 750, "UNT — Ingeniería de Sistemas")
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
        self.drawString(54, 28, "Predicción Temprana de Diabetes Tipo 2 — Ordoñez Reyes & Quispe Sanchez")
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
    c_dark = colors.HexColor("#2D3748")      # Texto oscuro
    c_border = colors.HexColor("#CBD5E0")    # Borde gris
    c_red_bg = colors.HexColor("#FFF5F5")    # Rojo tenue
    c_red_border = colors.HexColor("#FEB2B2")
    c_red_text = colors.HexColor("#9B2C2C")
    c_green_bg = colors.HexColor("#F0FFF4")  # Verde tenue
    c_green_border = colors.HexColor("#9AE6B4")
    c_green_text = colors.HexColor("#22543D")

    title_style = ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=12.5, leading=15.5, textColor=c_primary, alignment=1, spaceAfter=2)
    subtitle_style = ParagraphStyle('DocSubTitle', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=c_secondary, alignment=1, spaceAfter=4)
    sec_title = ParagraphStyle('SecTitle', fontName='Helvetica-Bold', fontSize=9, leading=11.5, textColor=c_primary, spaceBefore=6, spaceAfter=2.5, keepWithNext=True)
    step_header = ParagraphStyle('StepHeader', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=c_primary, spaceBefore=6, spaceAfter=3, keepWithNext=True)
    body_style = ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=7.6, leading=10.4, textColor=c_dark, spaceAfter=2)
    body_justify = ParagraphStyle('BodyJustify', fontName='Helvetica', fontSize=7.5, leading=10.2, textColor=c_dark, alignment=4, spaceAfter=3)

    del_title = ParagraphStyle('DelTitle', fontName='Helvetica-Bold', fontSize=7.2, leading=9.2, textColor=c_red_text)
    del_text = ParagraphStyle('DelText', fontName='Helvetica', fontSize=6.8, leading=8.8, textColor=c_red_text)
    
    ins_title = ParagraphStyle('InsTitle', fontName='Helvetica-Bold', fontSize=7.2, leading=9.2, textColor=c_green_text)
    ins_text = ParagraphStyle('InsText', fontName='Helvetica', fontSize=6.8, leading=8.8, textColor=c_green_text)
    
    cell_style = ParagraphStyle('CellText', fontName='Helvetica', fontSize=6.6, leading=8.6, textColor=c_dark)
    cell_bold = ParagraphStyle('CellBold', fontName='Helvetica-Bold', fontSize=6.6, leading=8.6, textColor=c_dark)
    cell_header = ParagraphStyle('CellHeader', fontName='Helvetica-Bold', fontSize=6.8, leading=8.6, textColor=colors.white)

    story = []

    def make_step_card(step_num, page_label, step_title, what_to_remove, what_to_put, rationale):
        header_text = f"<b>PASO {step_num}: {step_title.upper()} (PÁGINAS: {page_label})</b>"
        t_data = [
            [
                Paragraph("<b>❌ LO QUE DEBES RETIRAR O CORREGIR EN TU WORD:</b>", del_title),
                Paragraph("<b>✅ LO QUE DEBES REDACTAR EXACTAMENTE:</b>", ins_title)
            ],
            [
                Paragraph(what_to_remove, del_text),
                Paragraph(what_to_put, ins_text)
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
        
        note_p = Paragraph(f"<b>💡 Criterio de Coherencia:</b> {rationale}", ParagraphStyle('Note', fontName='Helvetica-Oblique', fontSize=6.6, leading=8.3, textColor=colors.HexColor("#4A5568")))
        
        return KeepTogether([
            Paragraph(header_text, step_header),
            t,
            Spacer(1, 1),
            note_p,
            Spacer(1, 4)
        ])

    # ==================== PÁGINA 1 ====================
    story.append(Paragraph("UNIVERSIDAD NACIONAL DE TRUJILLO", title_style))
    story.append(Paragraph("FACULTAD DE INGENIERÍA — ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS", subtitle_style))
    story.append(Paragraph("<b>GUÍA PRÁCTICA PASO A PASO: REESTRUCTURACIÓN DE TESIS</b>", ParagraphStyle('ReportName', fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=c_primary, alignment=1, spaceAfter=2)))
    story.append(Paragraph("<b>Tesis:</b> <i>Predicción Temprana de Diabetes Tipo 2 aplicando un Modelo en Machine Learning en Centro de Salud Casa Grande</i>", ParagraphStyle('SubTesis', fontName='Helvetica-Oblique', fontSize=7.5, leading=9.5, textColor=c_secondary, alignment=1, spaceAfter=2)))
    story.append(Paragraph("<b>Tesistas:</b> Ordoñez Reyes Abraham Benjamin & Quispe Sanchez Edward Steven | <b>Año:</b> 2026", ParagraphStyle('Tesistas', fontName='Helvetica', fontSize=7, leading=8.5, textColor=colors.HexColor("#718096"), alignment=1, spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=5))

    story.append(Paragraph("<b>ORIENTACIÓN GENERAL PARA LOS AUTORES:</b> Esta guía contiene las modificaciones exactas ordenadas secuencialmente, redactadas en un lenguaje técnico formal, fluido y comprensible, eliminando términos artificiales o incongruencias matemáticas para garantizar el dictamen aprobatorio del jurado.", body_style))
    story.append(Spacer(1, 3))

    # PASO 1: RESUMEN Y ABSTRACT
    p1_del = ("• Frases que afirmen 'automatización del diagnóstico médico'.<br/>"
              "• La mención imprecisa de 'mejora del 15% en F1-score y precisión'.<br/>"
              "• La palabra errónea 'Palaras Clave'.")
    p1_ins = ("• <b>Redactar en Resumen (Pág. 11):</b> '...Se implementó un sistema de apoyo a la toma de decisiones clínicas (CDSS) basado en Random Forest regularizado. El modelo optimizado demostró un desempeño sólido sobre 2,000 casos de prueba independientes, alcanzando una <b>Exactitud de 90.85% (IC 95%: 89.50% – 92.10%)</b>, una <b>Sensibilidad diagnóstica de 95.50%</b> (detectando a 955 de 1,000 pacientes con diabetes), un <b>F1-Score de 0.9126</b> y un <b>Coeficiente de Correlación de Matthews (MCC) de 0.8206 (82.06%)</b>, superando de manera estadísticamente significativa a la línea base tradicional (McNemar χ² = 221.13, p &lt; 0.001). En el plano asistencial en el Centro de Salud Casa Grande, el tiempo de evaluación se redujo de 38.58 a 0.35 minutos y el costo operativo médico disminuyó de S/. 24.17 a S/. 0.21 por atención...'<br/>"
              "• <b>Traducir fielmente al Abstract (Pág. 12)</b> y corregir 'Palabras clave'.")
    story.append(make_step_card("1", "Páginas 11 y 12", "Alineación del Resumen y Abstract", p1_del, p1_ins, "Establece el alcance legal como sistema de soporte (no reemplazo del médico) y expone los resultados empíricos reales."))

    # PASO 2: REALIDAD PROBLEMÁTICA
    p2_del = ("• Párrafos genéricos sin números locales del Centro de Salud Casa Grande.<br/>"
              "• Modismos coloquiales como 'el camino es cuesta arriba' o 'manos atadas'.")
    p2_ins = ("• <b>Incorporar en Capítulo I (Págs. 13-14):</b><br/>"
              "  1. El Centro de Salud Casa Grande (I-3) brinda cobertura a más de 30,000 habitantes en Ascope, atendiendo entre 1,200 y 1,450 consultas mensuales (160 vinculadas a control metabólico).<br/>"
              "  2. El laboratorio posee recursos limitados, procesando únicamente entre 35 y 45 exámenes de glucemia en ayunas al mes.<br/>"
              "  3. La atención tradicional demanda 38.58 minutos por paciente (inicio: signos vitales en triaje; fin: registro en historia clínica física).<br/>"
              "  4. Los médicos atienden de 25 a 30 pacientes por turno, generando colas de espera de 45 a 80 minutos.<br/>"
              "  5. El costo operativo directo es de S/. 24.17 por consulta (sueldos médicos de S/. 6,000 a S/. 8,000 mensuales / 192 horas).<br/>"
              "  6. El 70% de historias clínicas continúa en soporte de papel físico y hasta un 25% omite registros continuos de IMC.")
    story.append(make_step_card("2", "Páginas 13 y 14", "Incorporación de Indicadores en la Realidad Problemática", p2_del, p2_ins, "Sustenta la necesidad del proyecto con datos cuantitativos auditables del propio establecimiento de salud."))

    story.append(PageBreak())

    # ==================== PÁGINA 2 ====================
    # PASO 3: MARCO TEÓRICO E HIPERPARÁMETROS
    p3_del = ("• Descripciones teóricas extensas que no explican cómo se controla el sobreajuste.")
    p3_ins = ("• <b>Agregar en Marco Teórico (Pág. 18):</b><br/>"
              "  Explicar que la capacidad de generalización de Random Forest ante datos nuevos depende del control estricto de sus hiperparámetros de regularización:<br/>"
              "  - <i>`n_estimators` (200):</i> estabiliza la votación del ensamble mediante agregación bootstrap (bagging).<br/>"
              "  - <i>`max_depth` (10):</i> poda controlada que frena ramas innecesarias para evitar la memorización de ruido.<br/>"
              "  - <i>`min_samples_leaf` (8):</i> exige que cada hoja terminal agrupe al menos 8 pacientes, suavizando la probabilidad diagnóstica.<br/>"
              "  - <i>`criterion = 'entropy'`:</i> maximiza la ganancia de información en variables fisiológicas continuas.<br/>"
              "  - <i>`class_weight = 'balanced'`:</i> compensa la importancia de la clase enferma para evitar falsos negativos.")
    story.append(make_step_card("3", "Página 18", "Marco Teórico de Hiperparámetros de Regularización", p3_del, p3_ins, "Le otorga fundamento ingenieril a la arquitectura del algoritmo antes de mostrar los experimentos."))

    # PASO 4: OBJETIVOS ESPECÍFICOS
    p4_del = ("• OE2: '...impacto en la saturación hospitalaria...'<br/>"
              "• OE3: '...disminución de costos económicos familiares...'<br/>"
              "• Mismas frases en problemas e hipótesis específicas.")
    p4_ins = ("• <b>Redactar en Objetivos (Págs. 26-27 y Anexo 4):</b><br/>"
              "  - <b>OE1:</b> Evaluar la eficacia diagnóstica del modelo de Machine Learning en la predicción temprana de Diabetes Tipo 2 frente a métodos tradicionales.<br/>"
              "  - <b>OE2:</b> Evaluar la reducción en el tiempo promedio del proceso de atención y tamizaje clínico tras la implementación del sistema web.<br/>"
              "  - <b>OE3:</b> Evaluar la reducción del costo operativo directo del personal de salud por consulta médica asistida por el sistema.")
    story.append(make_step_card("4", "Páginas 26 y 27", "Alineamiento de Problemas, Objetivos e Hipótesis", p4_del, p4_ins, "Garantiza coherencia total entre lo que se promete en la introducción y los datos reales obtenidos en los resultados."))

    # PASO 5: METODOLOGÍA (POBLACIÓN Y MUESTRA)
    p5_del = ("• Afirmar que se usaron 80 pacientes locales para entrenar la Inteligencia Artificial.<br/>"
              "• El párrafo de 'escalamiento antes de dividir los datos' (fuga de información).")
    p5_ins = ("• <b>Redactar en Capítulo II (Págs. 31-34):</b><br/>"
              "  <b>1. Base de Entrenamiento y Evaluación de ML (10,000 registros):</b> Datos clínicos estructurados con 8 variables biológicas, balanceados 50/50 (5,000 diabéticos y 5,000 sanos). Partición estratificada: 60% Entrenamiento (6,000), 20% Validación (2,000) y 20% Prueba Ciega (2,000).<br/>"
              "  <b>2. Muestra de Validación Clínica en Casa Grande (80 consultas):</b> Atenciones médicas pareadas evaluadas en 5 turnos de trabajo para medir tiempo y costo asistencial.<br/>"
              "  <b>3. Procedimiento (Etapa V, Pág. 34):</b> Se implementó un `Pipeline` en Scikit-Learn donde el imputador y escalador se ajustaron exclusivamente con el conjunto de entrenamiento, previniendo toda fuga de información.")
    story.append(make_step_card("5", "Páginas 31 a 34", "Separación Metodológica de Muestras y Pipeline", p5_del, p5_ins, "Resuelve la observación crítica de inconsistencia de dataset y demuestra buenas prácticas en ciencia de datos."))

    story.append(PageBreak())

    # ==================== PÁGINA 3 ====================
    # PASO 6: RESULTADOS Y ANÁLISIS ESTADÍSTICO (CRÍTICO)
    p6_del = ("• Tabla 17 con medias binarias PRETEST=0.46 y POSTEST=0.31.<br/>"
              "• La Figura 6 antigua con barras decrecientes.<br/>"
              "• La prueba de normalidad aplicada a ceros y unos en la Pág. 44.<br/>"
              "• El análisis de costos con muestra de n=5.")
    p6_ins = ("• <b>Reemplazar Sección 3.2 (Costo, Pág. 39-42):</b> Indicar que las 5 filas corresponden a turnos que agrupan las <b>80 consultas individuales</b>, obteniendo una reducción de S/. 24.17 a S/. 0.21 (Wilcoxon Z = -7.770, p &lt; 0.001).<br/>"
              "• <b>Reemplazar Sección 3.3 (Machine Learning, Pág. 42-45):</b><br/>"
              "  1. Colocar la <b>Tabla 17 con el Reporte Multimétrico</b> (Exactitud 90.85%, Sensibilidad 95.50%, Especificidad 86.20%, F1 0.9126, ROC-AUC 0.9722 y MCC 82.06%).<br/>"
              "  2. Colocar la <b>Figura 6</b> con el gráfico comparativo ascendente (`grafico_pretest_postest_actualizado.png`).<br/>"
              "  3. Agregar la <b>Figura 7</b> con la Matriz de Confusión (`matriz_confusion.png`: TN=862, FP=138, FN=45, TP=955 en 2,000 casos).<br/>"
              "  4. Redactar el contraste inferencial de <b>McNemar para muestras pareadas</b> (casos discordantes b=309, c=33, χ² = 221.13, p &lt; 0.001) para rechazar H₀.")
    story.append(make_step_card("6", "Páginas 39 a 45", "Reemplazo de Tablas, Gráficos y Pruebas Inferenciales", p6_del, p6_ins, "Elimina la contradicción visual de las barras invertidas y fundamenta el éxito del modelo con pruebas no paramétricas."))

    # PASO 7: ANEXO 8 (CRISP-DM) Y MODELADO
    p7_del = ("• El fragmento de código que solo indicaba 'n_estimators=100' sin otros hiperparámetros.<br/>"
              "• Las capturas de terminal antiguas con soportes desiguales de 117 y 154 casos y umbral 0.35.")
    p7_ins = ("• <b>Actualizar en Anexo 8 (Páginas 89 y 90):</b><br/>"
              "  Insertar la <b>Tabla Técnica de Hiperparámetros de Regularización</b>:<br/>"
              "  - Estimadores: 200 árboles de decisión.<br/>"
              "  - Profundidad máxima (`max_depth`): 10 niveles.<br/>"
              "  - Mínimo de muestras por división (`min_samples_split`): 16.<br/>"
              "  - Mínimo de muestras en nodo hoja (`min_samples_leaf`): 8.<br/>"
              "  - Criterio de división: entropía de Shannon.<br/>"
              "  - Ponderación de clases: balanceada (`class_weight = 'balanced'`).<br/>"
              "  - Umbral de decisión clínico: estándar de 0.50.<br/>"
              "  - Semilla determinista: random_state = 42.<br/>"
              "• Sustituir las capturas antiguas por los reportes unificados sobre los 2,000 casos ciegos de prueba.")
    story.append(make_step_card("7", "Páginas 89 a 93", "Documentación del Modelado en el Anexo 8 (CRISP-DM)", p7_del, p7_ins, "Cumple la exigencia del jurado de evidenciar un ajuste sistemático de hiperparámetros propio de Ingeniería de Sistemas."))

    # CUADRO DE RESUMEN FINAL
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>TABLA CONSOLIDADA DE VALORES OFICIALES PARA TODA LA TESIS:</b>", sec_title))
    t_sum_data = [
        [Paragraph("Indicador o Métrica", cell_header), Paragraph("Valor Oficial", cell_header), Paragraph("Población / Muestra", cell_header), Paragraph("Prueba Estadística Aplicada", cell_header)],
        [Paragraph("Tiempo de Atención", cell_bold), Paragraph("De 38.58 a 0.35 min (-99.09%)", cell_style), Paragraph("80 consultas en Casa Grande", cell_style), Paragraph("Wilcoxon pareado: Z = -7.770, p &lt; 0.001", cell_style)],
        [Paragraph("Costo Operativo Médico", cell_bold), Paragraph("De S/. 24.17 a S/. 0.21 (-99.11%)", cell_style), Paragraph("80 consultas en Casa Grande", cell_style), Paragraph("Wilcoxon pareado: Z = -7.770, p &lt; 0.001", cell_style)],
        [Paragraph("Exactitud (Accuracy)", cell_bold), Paragraph("90.85% [89.50% – 92.10%]", cell_style), Paragraph("2,000 casos de prueba ciegos", cell_style), Paragraph("Bootstrap 1,000 repeticiones (IC 95%)", cell_style)],
        [Paragraph("Sensibilidad (Recall DT2)", cell_bold), Paragraph("<b>95.50%</b> (955 de 1,000 diabéticos)", cell_style), Paragraph("2,000 casos de prueba ciegos", cell_style), Paragraph("Bootstrap 1,000 repeticiones (IC 95%)", cell_style)],
        [Paragraph("F1-Score", cell_bold), Paragraph("0.9126 [0.8991 – 0.9247]", cell_style), Paragraph("2,000 casos de prueba ciegos", cell_style), Paragraph("Bootstrap 1,000 repeticiones (IC 95%)", cell_style)],
        [Paragraph("ROC-AUC", cell_bold), Paragraph("0.9722 [0.9653 – 0.9784]", cell_style), Paragraph("2,000 casos de prueba ciegos", cell_style), Paragraph("Curva ROC / Bootstrap (IC 95%)", cell_style)],
        [Paragraph("Coeficiente Matthews (MCC)", cell_bold), Paragraph("<b>0.8206 (82.06%)</b>", cell_bold), Paragraph("2,000 casos de prueba ciegos", cell_style), Paragraph("Rango objetivo institucional (80% – 85%)", cell_style)],
        [Paragraph("Contraste Inferencial de ML", cell_bold), Paragraph("χ² = 221.13, p &lt; 0.001", cell_style), Paragraph("Mismos 2,000 casos pareados", cell_style), Paragraph("Prueba de McNemar con corrección (Rechazo H₀)", cell_style)],
    ]
    t_sum = Table(t_sum_data, colWidths=[120, 115, 115, 154])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_sum)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF Guía Paso a Paso generado exitosamente: {PDF_FILENAME}")

if __name__ == '__main__':
    build_pdf()
