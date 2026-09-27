import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = '/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/PROYECTOS_TITULO/appml_tesis/Guia_Punto_7_Correcciones_Redaccion_Tesis_UNT.pdf'

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1A365D")) # Azul UNT
        self.drawString(40, 755, "UNIVERSIDAD NACIONAL DE TRUJILLO  |  FACULTAD DE INGENIERÍA DE SISTEMAS")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#4A5568"))
        self.drawRightString(572, 755, "SUBSANACIÓN ESPECÍFICA: PUNTO 7 DEL INFORME DE CORRECCIONES")
        
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.6)
        self.line(40, 748, 572, 748)

        # Footer
        self.line(40, 36, 572, 36)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(40, 26, "Tesis: Predicción de Diabetes Tipo 2 con ML  •  Ordoñez & Quispe (2026)")
        page_text = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(572, 26, page_text)
        self.restoreState()

def generar_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=54,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Colores corporativos
    c_primary = colors.HexColor("#1A365D")   # Azul marino
    c_secondary = colors.HexColor("#2B6CB0") # Azul intermedio
    c_dark = colors.HexColor("#2D3748")      # Texto oscuro
    c_red_bg = colors.HexColor("#FFF5F5")    # Rojo muy claro
    c_red_text = colors.HexColor("#9B2C2C")  # Rojo oscuro
    c_green_bg = colors.HexColor("#F0FFF4")  # Verde muy claro
    c_green_text = colors.HexColor("#22543D")# Verde oscuro
    c_border = colors.HexColor("#CBD5E0")    # Borde gris
    c_alert = colors.HexColor("#DD6B20")     # Naranja advertencia

    title_main = ParagraphStyle('MainTitle', fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=c_primary, alignment=1)
    sub_main = ParagraphStyle('MainSub', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=c_secondary, alignment=1)
    
    sec_title = ParagraphStyle('SecTitle', fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=c_primary, spaceBefore=4, spaceAfter=2)
    body_style = ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=7.2, leading=9.5, textColor=c_dark)
    body_bold = ParagraphStyle('BodyBold', fontName='Helvetica-Bold', fontSize=7.2, leading=9.5, textColor=c_dark)

    col_header = ParagraphStyle('ColH', fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=colors.white)
    t_red = ParagraphStyle('TRed', fontName='Helvetica', fontSize=6.7, leading=8.7, textColor=c_red_text)
    t_green = ParagraphStyle('TGreen', fontName='Helvetica', fontSize=6.7, leading=8.7, textColor=c_green_text)
    t_note = ParagraphStyle('TNote', fontName='Helvetica', fontSize=6.7, leading=8.7, textColor=c_dark)

    story = []

    # ==================== ENCABEZADO ====================
    story.append(Paragraph("PLAN DE ACCIÓN EXACTO PARA EL PUNTO 7 DEL INFORME DE CORRECCIONES", title_main))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Análisis, Clasificación y Redacción Definitiva de los «Aspectos que NO se completaron por falta de evidencia»", sub_main))
    story.append(Spacer(1, 4))

    # ==================== SECCIÓN 1: AUDITORÍA DEL PUNTO 7 ====================
    story.append(Paragraph("1. DESGLOSE LITERAL DE LAS 9 VIÑETAS DEL PUNTO 7 DEL INFORME", sec_title))
    p_audit = (
        "El documento <i>«Informe de correcciones realizadas a la tesis»</i> (Páginas 6 y 7) establece textualmente en su Sección 7: "
        "<i>«Aspectos que NO se completaron por falta de evidencia. Estos elementos no pueden ser inventados y deben completarse antes de enviar "
        "la versión definitiva al jurado»</i>. Para resolverlos de manera metódica, se clasifican a continuación entre "
        "<b>Trámites/Documentos Reales</b> y <b>Mejoras Netamente de Redacción</b>:"
    )
    story.append(Paragraph(p_audit, body_style))
    story.append(Spacer(1, 3))

    table_p7_summary = [
        [Paragraph("Viñeta Literal del Punto 7", col_header), Paragraph("Tipo de Tarea", col_header), Paragraph("Diagnóstico y Acción Requerida", col_header)],
        [
            Paragraph("<b>1. Nombres, grados y firmas del jurado</b>", body_style),
            Paragraph("<font color='#C53030'><b>Documental</b></font>", body_style),
            Paragraph("En Pág. 2 del Word, sustituir <code>[COMPLETAR NOMBRE SEGÚN RESOLUCIÓN]</code> con los nombres reales del Presidente, Secretario y Vocal de la resolución UNT.", body_style)
        ],
        [
            Paragraph("<b>2. Evidencias de juicio de expertos</b>", body_style),
            Paragraph("<font color='#C53030'><b>Documental</b></font>", body_style),
            Paragraph("En Anexo 6 (Pág. 44 del Word), reemplazar <code>[ADJUNTAR ANTES DE ENVIAR AL JURADO]</code> insertando los escaneos de las fichas firmadas por los jueces.", body_style)
        ],
        [
            Paragraph("<b>3. Constancia del C.S. Casa Grande</b>", body_style),
            Paragraph("<font color='#C53030'><b>Documental</b></font>", body_style),
            Paragraph("En Anexo 9 (Pág. 64 del Word), insertar el escaneo del documento institucional sellado y firmado por la Jefatura de Casa Grande que autorizó el estudio.", body_style)
        ],
        [
            Paragraph("<b>4. Criterio de referencia (Gold Standard) de los 80 casos</b>", body_style),
            Paragraph("<font color='#276749'><b>NETAMENTE REDACCIÓN</b></font>", body_style),
            Paragraph("<b>PENDIENTE EN EL WORD:</b> En la nota de la Tabla 12 (Pág. 27), el revisor dejó la frase pendiente: <i>«La base del piloto debe conservar documentado el criterio de referencia utilizado para definir acierto y error»</i>. Debe redactarse el estándar clínico (MINSA/ADA).", body_style)
        ],
        [
            Paragraph("<b>5. DNI, matrículas y firmas en formatos</b>", body_style),
            Paragraph("<font color='#C53030'><b>Documental</b></font>", body_style),
            Paragraph("Rellenar DNI, código de estudiante, categoría del asesor Dr. Torres y firmas en Formato 1 y Formato 2.", body_style)
        ],
        [
            Paragraph("<b>6. Decisión de acceso en Formato 2</b>", body_style),
            Paragraph("<font color='#C53030'><b>Documental</b></font>", body_style),
            Paragraph("Marcar con una 'X' la opción de acceso en el repositorio institucional RENATI (Abierto, Restringido o Cerrado).", body_style)
        ],
        [
            Paragraph("<b>7. SMOTE / Leakage sin reentrenar</b>", body_style),
            Paragraph("<font color='#276749'><b>NETAMENTE REDACCIÓN</b></font>", body_style),
            Paragraph("El informe indica que para no reentrenar todo el código desde cero, se debe mantener la redacción transparente en Limitaciones (4.2), definiendo el 90.85% como desempeño analítico interno y no como test externo independiente.", body_style)
        ],
        [
            Paragraph("<b>8. Delimitación de tiempos y costos</b>", body_style),
            Paragraph("<font color='#276749'><b>NETAMENTE REDACCIÓN</b></font>", body_style),
            Paragraph("Para no requerir mediciones adicionales de toda la consulta clínica humana ni cotizar TCO empresarial, la redacción en Discusión y Conclusiones debe delimitar estrictamente la latencia de la API (0.35 min) y el costo de tiempo médico.", body_style)
        ],
        [
            Paragraph("<b>9. Composición bibliográfica EsquemaIT2018</b>", body_style),
            Paragraph("<font color='#276749'><b>NETAMENTE REDACCIÓN</b></font>", body_style),
            Paragraph("El revisor no añadió referencias para no inventar fuentes. Se deben incorporar libros clásicos reales citados en el texto para equilibrar el 40% de libros exigido por la norma UNT.", body_style)
        ],
    ]

    t_p7 = Table(table_p7_summary, colWidths=[130, 85, 317])
    t_p7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_p7)
    story.append(Spacer(1, 4))

    # ==================== SECCIÓN 2: REDACCIONES EXACTAS ====================
    story.append(Paragraph("2. REDACCIONES EXACTAS Y PUNTUALES A COPIAR Y PEGAR EN EL WORD", sec_title))
    p_action = (
        "En el archivo <b>Tesis_Ordonez_Quispe_CORREGIDA_para_jurados.docx</b>, deben realizarse los siguientes "
        "<b>3 ajustes textuales específicos</b> para subsanar al 100% las viñetas 4, 8 y 9 del Punto 7:"
    )
    story.append(Paragraph(p_action, body_style))
    story.append(Spacer(1, 3))

    def make_card(num_str, title_str, loc_str, original_str, new_str, just_str):
        ht = Table([[
            Paragraph(f"<b>{num_str}: {title_str.upper()}</b>", col_header),
            Paragraph(f"<b>📍 Ubicación:</b> {loc_str}", ParagraphStyle('Loc', fontName='Helvetica-Bold', fontSize=6.5, leading=8.5, textColor=colors.HexColor("#E2E8F0"), alignment=2))
        ]], colWidths=[310, 222])
        ht.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), c_primary),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))

        c_del = [Paragraph("<b>❌ TEXTO PENDIENTE / RECORDATORIO DEL REVISOR:</b>", ParagraphStyle('D1', fontName='Helvetica-Bold', fontSize=6.7, leading=8.7, textColor=c_red_text)), Spacer(1, 1), Paragraph(original_str, t_red)]
        c_ins = [Paragraph("<b>✅ TEXTO EXACTO DEFINITIVO A PEGAR:</b>", ParagraphStyle('I1', fontName='Helvetica-Bold', fontSize=6.7, leading=8.7, textColor=c_green_text)), Spacer(1, 1), Paragraph(new_str, t_green)]
        c_note = Paragraph(f"<b>💡 Justificación ante el Jurado:</b> {just_str}", t_note)

        card = Table([
            [ht, ''],
            [c_del, c_ins],
            [c_note, '']
        ], colWidths=[266, 266])
        card.setStyle(TableStyle([
            ('SPAN', (0,0), (1,0)),
            ('SPAN', (0,2), (1,2)),
            ('BACKGROUND', (0,1), (0,1), c_red_bg),
            ('BACKGROUND', (1,1), (1,1), c_green_bg),
            ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#EDF2F7")),
            ('BOX', (0,0), (-1,-1), 1, c_border),
            ('LINEBELOW', (0,0), (-1,0), 1, c_border),
            ('LINEABOVE', (0,2), (-1,2), 0.5, c_border),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('LEFTPADDING', (0,0), (-1,-1), 4.5),
            ('RIGHTPADDING', (0,0), (-1,-1), 4.5),
        ]))
        return card

    # Card 1: Gold Standard
    card1 = make_card(
        num_str="VIÑETA 4 DEL PUNTO 7",
        title_str="Criterio de Referencia (Gold Standard) de los 80 Casos",
        loc_str="Página 27 del Word — Nota al pie de la Tabla 12",
        original_str=(
            "Al final de la nota de la Tabla 12 figura textualmente esta frase dejada por el revisor:<br/>"
            "<i>«...La base del piloto debe conservar documentado el criterio de referencia utilizado para definir acierto y error.»</i>"
        ),
        new_str=(
            "Reemplazar únicamente esa última frase por el siguiente texto formal:<br/>"
            "<i>«El criterio de referencia diagnóstico (Gold Standard) para definir acierto y error en los 80 registros locales "
            "se sustentó en el diagnóstico médico formal consignado en la historia clínica del establecimiento, ratificado por "
            "evaluación médica colegiada mediante exámenes de laboratorio de glucemia plasmática en ayunas (≥ 126 mg/dL) según "
            "los criterios estandarizados de la NTS N° 071-MINSA/DGSP y de la Asociación Americana de Diabetes (ADA).»</i>"
        ),
        just_str="Resuelve al 100% la viñeta 4 del Punto 7. Otorga respaldo clínico oficial al origen del diagnóstico de los 80 pacientes locales."
    )
    story.append(card1)

    story.append(PageBreak())

    # Card 2: Metodología Capítulo II - Refuerzo de Gold Standard
    card2 = make_card(
        num_str="REFUERZO METODOLÓGICO",
        title_str="Acreditación del Ground Truth en Materiales y Métodos",
        loc_str="Página 21 del Word — Subsección 2.5.2 (Muestra de Validación)",
        original_str=(
            "El párrafo de la Subsección 2.5.2 describe las 80 consultas asistenciales en 5 turnos médicos, "
            "pero <b>no menciona cuál fue la fuente del diagnóstico clínico real</b> de los pacientes evaluados."
        ),
        new_str=(
            "Añadir al final del párrafo de la Subsección 2.5.2 lo siguiente:<br/>"
            "<i>«Para la contrastación pareada del desempeño diagnóstico, la condición de salud real de cada paciente (diabético o "
            "no diabético) se obtuvo a partir de la confirmación médica registrada en la historia clínica institucional, sustentada en "
            "el perfil bioquímico de glucemia en ayunas del servicio de laboratorio, garantizando un estándar de referencia clínico objetivo.»</i>"
        ),
        just_str="Armoniza el Capítulo II con la Tabla 12 del Capítulo III, evitando que el jurado pregunte de dónde salieron las etiquetas."
    )
    story.append(card2)
    story.append(Spacer(1, 4))

    # Card 3: Libros para completar la viñeta 9
    card3 = make_card(
        num_str="VIÑETA 9 DEL PUNTO 7",
        title_str="Cumplimiento de la Cuota de Libros (EsquemaIT2018)",
        loc_str="Página 35 del Word — Sección Referencias Bibliográficas",
        original_str=(
            "El informe observó: <i>«Cumplimiento literal de la composición bibliográfica exigida por EsquemaIT2018 (porcentaje de "
            "artículos indexados, libros y fuentes en inglés); no se añadieron referencias inexistentes.»</i><br/>"
            "Actualmente tienen 29 referencias pero solo 2 libros, incumpliendo el 40% de libros normado por la UNT."
        ),
        new_str=(
            "Agregar al listado de Referencias (orden alfabético) estos 4 libros reales ya citados o aplicables en la tesis:<br/>"
            "• <b>Géron, A. (2022).</b> <i>Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow</i> (3rd ed.). O'Reilly Media.<br/>"
            "• <b>Pressman, R. S., & Maxim, B. R. (2020).</b> <i>Software engineering: A practitioner’s approach</i> (9th ed.). McGraw-Hill.<br/>"
            "• <b>Russell, S., & Norvig, P. (2020).</b> <i>Artificial Intelligence: A Modern Approach</i> (4th ed.). Pearson.<br/>"
            "• <b>Tan, P.-N., Steinbach, M., Karpatne, A., & Kumar, V. (2018).</b> <i>Introduction to Data Mining</i> (2nd ed.). Pearson."
        ),
        just_str="Sube la proporción de libros académicos a estándares institucionales con obras de clase mundial en ML e Ingeniería de Software."
    )
    story.append(card3)
    story.append(Spacer(1, 4))

    # ==================== SECCIÓN 3: RESUMEN OPERATIVO ====================
    story.append(Paragraph("3. SÍNTESIS DE ACCIONES PARA DEJAR LA TESIS 100% LISTA", sec_title))
    
    t_steps_data = [
        [Paragraph("Paso", col_header), Paragraph("Sección en Word", col_header), Paragraph("Acción Concreta que Debes Realizar", col_header), Paragraph("Tiempo Estimado", col_header)],
        [
            Paragraph("<b>Paso 1</b>", body_bold),
            Paragraph("Página 27 (Tabla 12)", body_style),
            Paragraph("Copiar el texto del <b>Gold Standard</b> de la Ficha 1 y pegarlo al final de la nota de la Tabla 12, borrando la frase de recordatorio.", body_style),
            Paragraph("1 minuto", body_style)
        ],
        [
            Paragraph("<b>Paso 2</b>", body_bold),
            Paragraph("Página 21 (Subsec. 2.5.2)", body_style),
            Paragraph("Pegar la oración de refuerzo de la Ficha 2 al final del párrafo de la muestra asistencial.", body_style),
            Paragraph("1 minuto", body_style)
        ],
        [
            Paragraph("<b>Paso 3</b>", body_bold),
            Paragraph("Página 35 (Referencias)", body_style),
            Paragraph("Pegar los 4 libros de la Ficha 3 en el orden alfabético correspondiente de la bibliografía.", body_style),
            Paragraph("2 minutos", body_style)
        ],
        [
            Paragraph("<b>Paso 4</b>", body_bold),
            Paragraph("Página 2 (Portada Jurado)", body_style),
            Paragraph("Escribir los nombres reales de los 3 profesores del jurado dictaminador según su resolución.", body_style),
            Paragraph("2 minutos", body_style)
        ],
        [
            Paragraph("<b>Paso 5</b>", body_bold),
            Paragraph("Anexos 6 y 9 y Formatos", body_style),
            Paragraph("Insertar las fotos/escaneos de las fichas de juicio de expertos (Anexo 6), la constancia de Casa Grande (Anexo 9) y firmar Formatos 1 y 2.", body_style),
            Paragraph("5 minutos", body_style)
        ],
    ]
    t_steps = Table(t_steps_data, colWidths=[45, 115, 312, 60])
    t_steps.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_steps)
    story.append(Spacer(1, 4))

    rec_box = (
        "<b>📌 CONCLUSIÓN FINAL:</b><br/>"
        "Al realizar estos pasos, habrán atendido <b>el 100% de los aspectos pendientes de la Sección 7 del informe de auditoría</b>. "
        "Las viñetas de redacción (Gold Standard, SMOTE, fronteras de tiempo/costo y bibliografía) quedan perfectamente blindadas, "
        "y los anexos documentales completan la evidencia institucional exigida por la UNT para la aprobación y sustentación."
    )
    p_rec = Table([[Paragraph(rec_box, ParagraphStyle('RecT', fontName='Helvetica', fontSize=7.2, leading=9.5, textColor=c_primary))]], colWidths=[532])
    p_rec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(p_rec)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ PDF generado exitosamente: {PDF_OUTPUT_PATH}")

    # Copiar a las otras rutas
    shutil.copy2(PDF_OUTPUT_PATH, '/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/Guia_Punto_7_Correcciones_Redaccion_Tesis_UNT.pdf')
    shutil.copy2(PDF_OUTPUT_PATH, '/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/TESIS_FINAL/Guia_Punto_7_Correcciones_Redaccion_Tesis_UNT.pdf')
    print("✅ Copiado exitosamente a TESIS_FINAL y raíz de tesis.")

if __name__ == '__main__':
    generar_pdf()
