import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = "/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/PROYECTOS_TITULO/appml_tesis/Guia_Correcciones_Exactas_Anexo8_y_Metodologia.pdf"
IMAGE_FLOWCHART = "/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/PROYECTOS_TITULO/appml_tesis/diagrama_flujo_prediccion_pipeline.png"
IMAGE_CODE = "/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/PROYECTOS_TITULO/appml_tesis/codigo_entrenamiento_pipeline.png"

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
            self.drawString(54, 752, "UNIVERSIDAD NACIONAL DE TRUJILLO — INGENIERÍA DE SISTEMAS")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#718096"))
            self.drawRightString(558, 752, "GUÍA COMPLETA DE REEMPLAZOS: ANEXO 8 Y METODOLOGÍA")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 746, 558, 746)

        # Footer (todas las páginas)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 40, 558, 40)
        page_str = f"Página {self._pageNumber} de {page_count}"
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#2B6CB0"))
        self.drawRightString(558, 28, page_str)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        self.drawString(54, 28, "Tesis: Predicción Temprana de Diabetes Tipo 2 — Ordoñez Reyes & Quispe Sánchez (2026)")
        self.restoreState()

def generar_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=44,
        bottomMargin=44
    )

    # Colores corporativos y académicos
    c_primary = colors.HexColor("#1A365D")    # Azul Marino UNT
    c_secondary = colors.HexColor("#2B6CB0")  # Azul Intermedio
    c_accent = colors.HexColor("#2C7A7B")     # Verde azulado clínico
    c_dark = colors.HexColor("#2D3748")
    c_border = colors.HexColor("#CBD5E0")
    
    # Colores para cajas de Diff (Rojo / Verde)
    c_red_bg = colors.HexColor("#FFF5F5")
    c_red_text = colors.HexColor("#9B2C2C")
    
    c_green_bg = colors.HexColor("#F0FFF4")
    c_green_text = colors.HexColor("#22543D")

    # Estilos
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', fontName='Helvetica-Bold', fontSize=12, leading=15,
        textColor=c_primary, alignment=1, spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle', fontName='Helvetica-Bold', fontSize=8.5, leading=11,
        textColor=c_secondary, alignment=1, spaceAfter=5
    )
    sec_title = ParagraphStyle(
        'SecTitle', fontName='Helvetica-Bold', fontSize=9, leading=12,
        textColor=c_primary, spaceBefore=5, spaceAfter=3, keepWithNext=True
    )
    subsec_title = ParagraphStyle(
        'SubSecTitle', fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=c_accent, spaceBefore=3, spaceAfter=2, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyCustom', fontName='Helvetica', fontSize=6.8, leading=9.2,
        textColor=c_dark, spaceAfter=3
    )

    del_title = ParagraphStyle('DelTitle', fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=c_red_text)
    del_text = ParagraphStyle('DelText', fontName='Helvetica', fontSize=6.3, leading=8.3, textColor=c_red_text)
    
    ins_title = ParagraphStyle('InsTitle', fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=c_green_text)
    ins_text = ParagraphStyle('InsText', fontName='Helvetica', fontSize=6.3, leading=8.3, textColor=c_green_text)

    cell_style = ParagraphStyle('CellText', fontName='Helvetica', fontSize=6.3, leading=8, textColor=c_dark)
    cell_bold = ParagraphStyle('CellBold', fontName='Helvetica-Bold', fontSize=6.3, leading=8, textColor=c_dark)
    cell_header = ParagraphStyle('CellHeader', fontName='Helvetica-Bold', fontSize=6.6, leading=8.4, textColor=colors.white)

    story = []

    # ==================== PORTADA / ENCABEZADO ====================
    story.append(Paragraph("UNIVERSIDAD NACIONAL DE TRUJILLO", title_style))
    story.append(Paragraph("FACULTAD DE INGENIERÍA — ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=4, spaceBefore=1))
    
    doc_banner_text = (
        "<b>GUÍA OFICIAL INTEGRAL DE CORRECCIONES Y REEMPLAZOS TEXTUALES: ANEXO 8 (CRISP-DM)</b><br/>"
        "<font size=7.5 color='#2B6CB0'><b>Alineamiento Metodológico, Corrección de Fuga de Datos, Ortografía y Estilo Académico</b></font><br/>"
        "<font size=6.8 color='#4A5568'><b>Tesis:</b> 'Sistema de apoyo al tamizaje de diabetes tipo 2 mediante aprendizaje automático en el Centro de Salud Casa Grande, 2025'</font><br/>"
        "<font size=6.2 color='#718096'><b>Autores:</b> Ordoñez Reyes, Abraham Benjamín | Quispe Sánchez, Edward Steven &nbsp;&nbsp;|&nbsp;&nbsp; <b>Fecha:</b> Septiembre 2026</font>"
    )
    banner_p = Paragraph(doc_banner_text, ParagraphStyle('Banner', fontName='Helvetica', fontSize=7.8, leading=10, alignment=1))
    banner_table = Table([[banner_p]], colWidths=[504])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 2))

    # Helper para crear bloques de DIFF
    def make_diff_card(page_ref, phase_name, err_title, err_body, rep_title, rep_body, note_body):
        header_text = f"<b>📍 UBICACIÓN EN TESIS: {page_ref.upper()} &nbsp;&nbsp;|&nbsp;&nbsp; {phase_name.upper()}</b>"
        header_p = Paragraph(header_text, ParagraphStyle('DiffH', fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=c_primary))
        
        col_del = [Paragraph(f"<b>❌ {err_title.upper()}</b>", del_title), Spacer(1, 1), Paragraph(err_body, del_text)]
        col_ins = [Paragraph(f"<b>✅ {rep_title.upper()}</b>", ins_title), Spacer(1, 1), Paragraph(rep_body, ins_text)]
        
        note_p = Paragraph(f"<b>💡 Instrucción / Dictamen Jurado:</b> {note_body}", ParagraphStyle('DiffNote', fontName='Helvetica', fontSize=6.2, leading=8.2, textColor=c_dark))

        t = Table([
            [header_p, ''],
            [col_del, col_ins],
            [note_p, '']
        ], colWidths=[246, 258])
        
        t.setStyle(TableStyle([
            ('SPAN', (0, 0), (1, 0)),
            ('SPAN', (0, 2), (1, 2)),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ('BACKGROUND', (0, 1), (0, 1), c_red_bg),
            ('BACKGROUND', (1, 1), (1, 1), c_green_bg),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#EDF2F7")),
            ('BOX', (0, 0), (-1, -1), 1, c_border),
            ('LINEBELOW', (0, 0), (-1, 0), 1, c_border),
            ('LINEABOVE', (0, 2), (-1, 2), 0.5, c_border),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        return t

    # ==================== PÁGINA 1: FASES 1 Y 2 ====================
    story.append(Paragraph("1. REEMPLAZOS EN FASE 1 Y FASE 2 DE CRISP-DM", sec_title))
    p_diag = (
        "En el <b>Capítulo I (Párrafos 303 al 307)</b>, el Objetivo General y los Objetivos Específicos ya fueron "
        "reformulados correctamente hacia <b>eficacia diagnóstica, tiempo promedio y costo operativo</b>. En el <b>Anexo 8</b> "
        "deben sustituirse las menciones desactualizadas de 'saturación hospitalaria', 'costos familiares' y N=100."
    )
    story.append(Paragraph(p_diag, body_style))
    story.append(Spacer(1, 2))

    # --- ITEM 1: FASE 1 COMPRENSIÓN DEL NEGOCIO ---
    c1 = make_diff_card(
        page_ref="Página 89 — Anexo 8 (Párrafos 1111, 1112 y 1124)",
        phase_name="Fase 1: Comprensión del Negocio (Sprint 0)",
        err_title="Texto Antiguo en Anexo 8 (Desalineado)",
        err_body=(
            "• <i>'Objetivo Principal: Mejorar la detección temprana de la Diabetes Tipo 2 en el Centro de Salud Casa Grande para reducir complicaciones y optimizar recursos.'</i><br/>"
            "• <i>'Objetivos Específicos: Evaluar la precisión del modelo frente a métodos tradicionales, analizar el impacto en la saturación hospitalaria y determinar el efecto en la reducción de costos familiares.'</i><br/>"
            "• <i>'Criterio de Éxito Técnico: Se estableció que el modelo debía alcanzar una precisión (accuracy) superior al 90% y un F1-Score competitivo...'</i>"
        ),
        rep_title="Texto Exacto Oficial (Alineado con Capítulo I)",
        rep_body=(
            "• <b>Objetivo de Negocio Principal (Objetivo General):</b> Determinar en qué medida la implementación de un sistema de apoyo al tamizaje basado en Machine Learning mejora la eficacia diagnóstica, el tiempo del proceso y el costo operativo en el Centro de Salud Casa Grande, 2025.<br/>"
            "• <b>Objetivos Específicos del Proyecto:</b><br/>"
            "  1. Evaluar la eficacia diagnóstica del modelo de Machine Learning en la predicción temprana de Diabetes Tipo 2 frente a métodos tradicionales.<br/>"
            "  2. Evaluar la reducción del tiempo promedio del proceso de atención y tamizaje clínico del paciente mediante el sistema web.<br/>"
            "  3. Evaluar la reducción del costo operativo directo del personal de salud por consulta médica generada mediante el uso del sistema.<br/>"
            "• <b>Criterios de Éxito del Proyecto (CRISP-DM):</b><br/>"
            "  - <i>Criterio Técnico (OE1):</i> Superar la precisión tradicional de línea base (46%), alcanzando en el modelo Random Forest una eficacia diagnóstica superior (86.3% a 90.85% de exactitud y F1 de 0.9126), superando a la Regresión Logística.<br/>"
            "  - <i>Criterio Asistencial (OE2):</i> Reducir significativamente el tiempo promedio de atención frente a los 38.58 minutos tradicionales (Wilcoxon p < 0.05).<br/>"
            "  - <i>Criterio Económico (OE3):</i> Reducir el costo operativo directo del personal de salud frente a la línea base tradicional de S/ 24.17 por consulta."
        ),
        note_body="Unifica literalmente el Objetivo General (P[303]) y los 3 Objetivos Específicos (P[305-307]) con el Anexo 8."
    )
    story.append(c1)
    story.append(Spacer(1, 3))

    # --- ITEM 2: FASE 2 COMPRENSIÓN DE DATOS ---
    c2 = make_diff_card(
        page_ref="Páginas 89-90 — Anexo 8 (Solo Párrafo 1151)",
        phase_name="Fase 2: Comprensión de los Datos (Sprint 1)",
        err_title="Texto Único a Reemplazar (Párrafo 1151)",
        err_body=(
            "• <i>'Se obtuvo un conjunto de datos anonimizados de 100 pacientes del Centro de Salud Casa Grande con los campos clave para poder realizar la predicción como:'</i><br/><br/>"
            "⚠️ <b>Problema observado por el Jurado:</b> Contradecía el tamaño de muestra de 80 y los soportes de cientos/miles de casos en la evaluación del algoritmo."
        ),
        rep_title="Texto Exacto a Pegar (Solo Reemplaza Párrafo 1151)",
        rep_body=(
            "<i>«Para el desarrollo del proyecto se estructuraron dos fuentes de datos complementarias: por un lado, un <b>dataset clínico de entrenamiento y validación</b> compuesto por 10,000 registros estandarizados para el aprendizaje del modelo de Machine Learning; y por otro lado, una <b>muestra asistencial local de 80 pacientes</b> del Centro de Salud Casa Grande para la evaluación preexperimental de tiempos y costos operativos en consulta. En ambos casos, el análisis y la inferencia predictiva se sustentan en los siguientes campos clave:»</i><br/><br/>"
            "<b>📌 INSTRUCCIÓN CLAVE:</b> Las descripciones de las <b>8 variables (Glucosa, Presión, Embarazos, Grosor, Insulina, BMI, Pedigree y Edad) SE MANTIENEN INTACTAS</b> a continuación de este párrafo. Solo se cambia este párrafo introductorio."
        ),
        note_body="Resuelve Observaciones 1, 22, 23, 25 y 26. No borres tus 8 variables, están excelentes."
    )
    story.append(c2)

    story.append(PageBreak())

    # ==================== PÁGINA 2: FASES 3, 4 Y 5 ====================
    story.append(Paragraph("2. REEMPLAZOS EN FASES 3, 4 Y 5 (PREPARACIÓN, MODELADO Y EVALUACIÓN)", sec_title))

    # --- ITEM 3: FASE 3 PREPARACIÓN DE DATOS (DATA LEAKAGE) ---
    c3 = make_diff_card(
        page_ref="Página 90 — Anexo 8 (Solo Párrafo 1170, antes de Tabla X)",
        phase_name="Fase 3: Preparación de Datos (Sprint 2 y 3) — ¡CRÍTICO!",
        err_title="Texto Único a Reemplazar (Párrafo 1170)",
        err_body=(
            "• <i>'Para la transformación de los datos se optó por reemplazar los 0 por su valor media de cada campo, y luego limpiar los datos que no lleguen a estar dentro del rango de ciertos campos como:'</i><br/><br/>"
            "⚠️ <b>Error de Fuga de Datos (Observación 10):</b> Confiesa haber reemplazado ceros por la media antes de entrenar."
        ),
        rep_title="Texto Exacto a Pegar (Reemplaza Párrafo 1170)",
        rep_body=(
            "<i>«En el proceso de limpieza y formateo de datos, se identificaron registros con valores iguales a cero en variables donde fisiológicamente es imposible un valor nulo (como glucemia, presión arterial, insulina o índice de masa corporal), catalogándose como datos faltantes o inconsistencias biológicas. Para evitar la fuga de información (data leakage), no se realizó ninguna imputación global prematura sobre el conjunto total de datos; en su lugar, se establecieron los rangos biológicos admisibles presentados a continuación y se delegó la imputación por mediana al Pipeline de Scikit-Learn ajustado exclusivamente con los datos de entrenamiento:»</i><br/><br/>"
            "<b>📌 INSTRUCCIÓN CLAVE:</b> La <b>Tabla de Rangos Fisiológicos</b> (Glucosa 70-500, Presión 60-180, etc.) <b>SE MANTIENE INTACTA</b> debajo de este párrafo."
        ),
        note_body="Resuelve Observaciones 10, 35 y 36. Elimina la confesión de fuga de datos y preserva la tabla de rangos."
    )
    story.append(c3)
    story.append(Spacer(1, 2))

    # --- ITEM 4: FASE 4 MODELADO, ENTRENAMIENTO Y CÓDIGO ---
    c4 = make_diff_card(
        page_ref="Páginas 91-92 — Anexo 8 (Párrafos 1195, 1202-1205 y Captura)",
        phase_name="Fase 4: Modelado, Hiperparámetros y Entrenamiento (Sprint 4)",
        err_title="Fuga, 100 Árboles y Código Desactualizado",
        err_body=(
            "• <b>P[1195] (Escalamiento):</b> <i>'Antes de dividir los datos se aplicó StandardScaler...'</i> (Fuga de datos).<br/>"
            "• <b>P[1202-1203] (Hiperparámetros):</b> Solo 100 árboles sin optimización.<br/>"
            "• <b>P[1205] (Entrenamiento):</b> <i>'El algoritmo construyó los 100 árboles de decisión...'</i> (Contradice los 200 árboles).<br/>"
            "• <b>Captura de Código:</b> Mostraba <code>RandomForestClassifier(n_estimators=100)</code> sin Pipeline ni parámetros óptimos."
        ),
        rep_title="Reemplazos Exactos Paso a Paso (P[1195], P[1205] y Código)",
        rep_body=(
            "• <b>Reemplazo P[1195] (Escalamiento):</b> <i>'El escalamiento (StandardScaler) se integró en el Pipeline, ajustándose (fit) solo con Train (6,000 casos).'</i><br/>"
            "• <b>Hiperparámetros:</b> <b>n_estimators = 200, max_depth = 10, min_samples_leaf = 8, min_samples_split = 16, criterion = 'entropy', class_weight = 'balanced'</b>.<br/>"
            "• <b>Reemplazo P[1205] (Entrenamiento):</b> <i>«Se ejecutó el método <code>pipeline.fit(X_train, y_train)</code>, proceso en el cual se ajustaron de manera secuencial la imputación, la estandarización y los <b>200 árboles de decisión</b> del ensamble Random Forest, aprendiendo exclusivamente sobre los 6,000 casos de entrenamiento.»</i><br/>"
            "• <b>Captura de Código:</b> Reemplazar la imagen por el bloque con <code>Pipeline([...])</code> y <code>n_estimators=200</code>."
        ),
        note_body="Resuelve la contradicción visual entre los 200 árboles del texto y la captura que mostraba 100 árboles."
    )
    story.append(c4)
    story.append(Spacer(1, 2))

    # --- ITEM 5: FASE 5 EVALUACIÓN Y MÉTRICAS ---
    c5 = make_diff_card(
        page_ref="Páginas 92-93 — Anexo 8 (Párrafos 1214 a 1223)",
        phase_name="Fase 5: Evaluación del Desempeño (Sprint 5)",
        err_title="Contradicciones Numéricas y Costo Familiar",
        err_body=(
            "• P[1214-1216]: Palabras 'pretest' y 'postest' sueltas, y <i>'Figura x. Matriz de confusión'</i> con soportes de 117 y 154 despareados.<br/>"
            "• F1-Score de 61.29% idéntico declarado como 'mejora del 15%'.<br/>"
            "• P[1223]: Afirmaba <i>'reducción de costos familiares'</i> (nunca medido)."
        ),
        rep_title="Vector Completo de Métricas y Cumplimiento de Objetivos",
        rep_body=(
            "• <b>Evaluación en Test Independiente (2,000 casos ciegos):</b><br/>"
            "  - Exactitud (Accuracy): <b>90.85%</b> (IC 95%: 89.50% – 92.10%) | En muestra local: <b>86.3%</b>.<br/>"
            "  - Sensibilidad (Recall): <b>95.50%</b> (detecta a 955 de 1,000 diabéticos).<br/>"
            "  - Especificidad: <b>86.20%</b> | F1-Score: <b>0.9126</b> | ROC-AUC: <b>0.9722</b> | MCC: <b>0.8206</b>.<br/>"
            "• <b>Reemplazo P[1223] (Costos Operativos):</b> <i>'Al automatizar la inferencia, el sistema reduce el tiempo asistencial de 38.58 a 0.35 minutos (Wilcoxon p < 0.001) y disminuye el costo operativo del médico de S/ 24.17 a S/ 0.21 por atención asistencial.'</i>"
        ),
        note_body="Resuelve Observaciones 2, 3, 4, 5, 30, 41, 42, 51, 52 y 55 (Inconsistencias de métricas y costos)."
    )
    story.append(c5)

    story.append(PageBreak())

    # ==================== PÁGINA 3: FASE 6, CÓDIGO NUEVO Y TABLA DE ERRORES ====================
    story.append(Paragraph("3. FASE 6 (DESPLIEGUE), CÓDIGO DEL PIPELINE Y CORRECCIÓN ORTOGRÁFICA", sec_title))

    # --- ITEM 6: FASE 6 DESPLIEGUE ---
    c6 = make_diff_card(
        page_ref="Página 93 — Anexo 8 (Párrafo 1226)",
        phase_name="Fase 6: Despliegue Operativo (Sprint 6) — ¡ESTABA VACÍA!",
        err_title="Sección en Blanco / Truncada en el Informe",
        err_body=(
            "• P[1226]: <i>'Despliegue'</i> (Literalmente figuraba solo el título y pasaba de inmediato al Anexo 9 sin ningún contenido ni explicación de cómo opera el sistema)."
        ),
        rep_title="Redacción Integral de la Arquitectura en Producción",
        rep_body=(
            "• <b>Microservicio REST (Flask Python):</b> Carga en memoria del pipeline inmutable (diabetes_pipeline.pkl), exponiendo el endpoint POST /predict.<br/>"
            "• <b>Validación e Ingesta JSON:</b> Mapeo de parámetros (KEYS_MAPPING), imputación y cálculo de probabilidad continua P(Diabetes=1|X) en menos de 50 ms.<br/>"
            "• <b>Integración Web (Laravel):</b> Consumo asíncrono desde el módulo de triaje, semáforo visual de alerta temprana y persistencia en Supabase.<br/>"
            "• <b>Explicabilidad Asistida (Gemini):</b> Generación de interpretación preventiva contextual transmitiendo únicamente variables numéricas disociadas (cero datos de filiación)."
        ),
        note_body="Resuelve Observaciones 44, 45, 8.5 y 64 (Documentación de despliegue, API, pruebas y seguridad clínica)."
    )
    story.append(c6)
    story.append(Spacer(1, 2))

    # --- NUEVA IMAGEN DE CÓDIGO (REEMPLAZO DE CAPTURA) ---
    story.append(Paragraph("Imagen Actualizada del Código de Entrenamiento (Reemplazo de la captura en Fase 4):", subsec_title))
    if os.path.exists(IMAGE_CODE):
        img_c = Image(IMAGE_CODE, width=440, height=130)
        story.append(img_c)
        story.append(Spacer(1, 2))

    # --- TABLA DE CORRECCIONES MENORES Y ORTOGRAFÍA ---
    story.append(Paragraph("Tabla de Correcciones de Ortografía, Términos Coloquiales y Formato Formal en Anexo 8:", subsec_title))
    t_err_data = [
        [Paragraph("Párrafo", cell_header), Paragraph("Texto Observado (Con Error)", cell_header), Paragraph("Corrección Inmediata a Realizar", cell_header), Paragraph("Tipo de Error", cell_header)],
        [
            Paragraph("<b>1106, 1129, 1147</b>", cell_bold),
            Paragraph("<i>'Compresión del Negocio'</i> / <i>'Compresión de los datos'</i>", cell_style),
            Paragraph("Cambiar por <b>'Comprensión del Negocio'</b> y <b>'Comprensión de los datos'</b> (agregar la letra 'n').", cell_style),
            Paragraph("Ortográfico", cell_bold)
        ],
        [
            Paragraph("<b>1121</b>", cell_bold),
            Paragraph("<i>'La posible falta de acceso... pero nada que impida el desarrollo...'</i>", cell_style),
            Paragraph("Cambiar por: <i>'El tratamiento restringido de datos personales en observancia de la Ley N° 29733 mediante disociación anónima.'</i>", cell_style),
            Paragraph("Estilo informal", cell_bold)
        ],
        [
            Paragraph("<b>1154</b>", cell_bold),
            Paragraph("<i>'Si una mujer ha tenido DMG...'</i>", cell_style),
            Paragraph("Definir la sigla en su primera mención: <b>'Diabetes Mellitus Gestacional (DMG)'</b>.", cell_style),
            Paragraph("Claridad técnica", cell_bold)
        ],
        [
            Paragraph("<b>1161</b>", cell_bold),
            Paragraph("<i>'Se utilizaron librerías de Python... en un entorno de Python'</i>", cell_style),
            Paragraph("Cambiar por: <i>'Se utilizaron librerías especializadas como Pandas, NumPy y Matplotlib en Jupyter Notebook.'</i>", cell_style),
            Paragraph("Redundancia", cell_bold)
        ],
        [
            Paragraph("<b>1185</b>", cell_bold),
            Paragraph("<i>'Tabla X. Rango de valores de las variables clave'</i>", cell_style),
            Paragraph("Cambiar la letra X por el número real (ejemplo: <b>'Tabla 8. Rango de valores...'</b>) y agregar nota al pie.", cell_style),
            Paragraph("Formato APA 7", cell_bold)
        ],
        [
            Paragraph("<b>1186</b>", cell_bold),
            Paragraph("<i>'Y aquí una muestra del código de cómo se aplicó:'</i>", cell_style),
            Paragraph("Cambiar por: <i>'A continuación, se presenta la implementación del preprocesamiento dentro del Pipeline:'</i>", cell_style),
            Paragraph("Estilo coloquial", cell_bold)
        ],
        [
            Paragraph("<b>1189</b>", cell_bold),
            Paragraph("<i>'...se seleccionó, entreno y optimizo el algoritmo...'</i>", cell_style),
            Paragraph("Colocar tildes en pasado: <b>'...se seleccionó, entrenó y optimizó el algoritmo...'</b>.", cell_style),
            Paragraph("Ortográfico", cell_bold)
        ],
        [
            Paragraph("<b>1202</b>", cell_bold),
            Paragraph("<i>'hiper parámetros'</i> (separado)", cell_style),
            Paragraph("Escribir junto: <b>'hiperparámetros'</b>.", cell_style),
            Paragraph("Ortográfico", cell_bold)
        ],
        [
            Paragraph("<b>1216</b>", cell_bold),
            Paragraph("<i>'Figura x. Matriz de confusión'</i> (con letra x minúscula)", cell_style),
            Paragraph("Numerar correlativamente (ejemplo: <b>'Figura 9. Matrices de confusión...'</b>) y agregar nota de fuente.", cell_style),
            Paragraph("Formato APA 7", cell_bold)
        ],
    ]
    t_err = Table(t_err_data, colWidths=[65, 145, 214, 80])
    t_err.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_err)

    story.append(PageBreak())

    # ==================== PÁGINA 4: DIAGRAMA DE FLUJO Y UBICACIÓN EN TESIS ====================
    story.append(Paragraph("4. DIAGRAMA DE FLUJO Y UBICACIÓN EN EL CAPÍTULO II (PÁG. 38)", sec_title))
    p_diag_intro = (
        "El siguiente flujograma modela la inferencia del Pipeline predictivo. Debe insertarse en el "
        "<b>Capítulo II (Subsección 2.3.8: Procedimiento, Pág. 38, Párrafo 508)</b> para resolver la nota pendiente:"
    )
    story.append(Paragraph(p_diag_intro, body_style))
    story.append(Spacer(1, 2))

    if os.path.exists(IMAGE_FLOWCHART):
        img_flow = Image(IMAGE_FLOWCHART, width=504, height=330)
        story.append(img_flow)
        story.append(Spacer(1, 2))
        p_caption = Paragraph(
            "<b>Figura X.</b> <i>Flujograma del proceso de inferencia predictiva y soporte a la decisión clínica mediante el Pipeline de Machine Learning. "
            "Integración del Frontend en Laravel (WebDT2), capa de seguridad/disociación (Ley N° 29733), microservicio en Flask (Scikit-Learn) y persistencia en Supabase.</i>",
            ParagraphStyle('FigCap', fontName='Helvetica', fontSize=6.2, leading=8.2, textColor=c_dark, alignment=1)
        )
        story.append(p_caption)

    story.append(Spacer(1, 3))

    # --- PÁRRAFO EXACTO PARA INSERTAR EN PÁGINA 38 ---
    story.append(Paragraph("Texto Exacto para insertar en el Capítulo II (Pág. 38, Sección 2.3.8 Procedimiento):", subsec_title))
    p_proc_text = (
        "<i>«Para asegurar la reproducibilidad técnica y la robustez del sistema en el entorno asistencial, "
        "la ejecución predictiva se modeló como un flujo de datos desacoplado y blindado contra fuga de información. "
        "Como se ilustra en la <b>Figura X</b>, los datos capturados durante el triaje clínico son disociados en el backend "
        "de la aplicación web (WebDT2) en estricta observancia de la Ley N° 29733 y transmitidos mediante una petición HTTP "
        "REST asíncrona al microservicio en Python. En este entorno, el vector de variables es procesado de manera atómica "
        "por el pipeline serializado (SimpleImputer + StandardScaler + RandomForestClassifier), "
        "el cual fue ajustado exclusivamente con datos de entrenamiento. El modelo calcula la probabilidad continua de riesgo "
        "P(Diabetes=1|X) y emite una clasificación binaria basada en un umbral clínico estandarizado de 0.50, retornando el resultado en "
        "tiempo real como una herramienta de apoyo al tamizaje asistencial del médico tratante (la adaptación detallada de "
        "CRISP-DM con Scrum se documenta exhaustivamente en el <b>Anexo 8</b>).»</i>"
    )
    p_box = Table([[Paragraph(p_proc_text, ParagraphStyle('BoxText', fontName='Helvetica', fontSize=6.6, leading=8.8, textColor=c_primary))]], colWidths=[504])
    p_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(p_box)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF regenerado limpiamente en 4 páginas perfectas: {PDF_OUTPUT_PATH}")

    # Copiar a las otras carpetas
    shutil.copy2(PDF_OUTPUT_PATH, '/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/Guia_Correcciones_Exactas_Anexo8_y_Metodologia.pdf')
    shutil.copy2(PDF_OUTPUT_PATH, '/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/TESIS_FINAL/Guia_Correcciones_Exactas_Anexo8_y_Metodologia.pdf')
    print("✅ Copiado a todas las rutas de tesis.")

if __name__ == '__main__':
    generar_pdf()
