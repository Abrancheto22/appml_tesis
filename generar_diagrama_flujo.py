import matplotlib.pyplot as plt
import matplotlib.patches as patches

def crear_diagrama_flujo():
    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Paleta de colores elegante
    c_bg_head = '#1A365D'      # Azul marino
    c_card_bg = '#F8FAFC'      # Gris muy tenue
    c_s1 = '#EBF8FF'           # Azul cielo (Frontend)
    c_s1_b = '#3182CE'
    c_s2 = '#E6FFFA'           # Verde azulado (Backend/Seguridad)
    c_s2_b = '#319795'
    c_s3 = '#FAF5FF'           # Morado (Pipeline ML)
    c_s3_b = '#805AD5'
    c_s4 = '#FFFAF0'           # Naranja/Ámbar (Persistencia/Clínico)
    c_s4_b = '#DD6B20'
    c_text_dark = '#1A202C'

    # Título Principal
    ax.text(50, 97.5, "FLUJOGRAMA DE INFERENCIA PREDICTIVA Y SOPORTE A LA DECISIÓN CLÍNICA",
            ha='center', va='center', fontsize=13, fontweight='bold', color=c_bg_head)
    ax.text(50, 95.2, "Integración del Sistema WebDT2 (Laravel) con el Microservicio de Machine Learning (Python / Scikit-Learn)",
            ha='center', va='center', fontsize=9, fontstyle='italic', color='#4A5568')

    # ==================== SECCIÓN 1: FRONTEND TRIAJE ====================
    rect_s1 = patches.FancyBboxPatch((3, 72), 94, 20.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor=c_s1, edgecolor=c_s1_b, linewidth=1.5)
    ax.add_patch(rect_s1)
    ax.text(5, 90.5, "1. CAPA DE PRESENTACIÓN Y TRIAJE CLÍNICO (WebDT2 - Laravel Frontend)",
            fontsize=9.5, fontweight='bold', color='#2B6CB0')

    # Caja 1.1: Usuario
    b1 = patches.FancyBboxPatch((5, 74), 26, 14.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b1)
    ax.text(18, 86, "Personal de Salud\n(Médico / Enfermera)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text_dark)
    ax.text(18, 78.5, "• Inicio de atención / triaje\n• Acceso por roles y permisos\n• Interfaz Web responsiva",
            ha='center', va='center', fontsize=7.5, color='#4A5568')

    # Flecha 1.1 -> 1.2
    ax.annotate('', xy=(34, 81.25), xytext=(31, 81.25),
                arrowprops=dict(facecolor='#4A5568', edgecolor='none', width=1.5, headwidth=6))

    # Caja 1.2: Parámetros
    b2 = patches.FancyBboxPatch((34, 74), 33, 14.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b2)
    ax.text(50.5, 86.5, "Ingreso de 8 Parámetros Fisiológicos", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text_dark)
    ax.text(50.5, 79, "1. Embarazos    5. Insulina (μU/ml)\n2. Glucosa (mg/dL) 6. IMC (kg/m²)\n3. P. Arterial (mmHg) 7. Función Pedigree\n4. Pliegue Tríceps 8. Edad (años)",
            ha='center', va='center', fontsize=7.2, color='#2D3748', family='monospace')

    # Flecha 1.2 -> 1.3
    ax.annotate('', xy=(70, 81.25), xytext=(67, 81.25),
                arrowprops=dict(facecolor='#4A5568', edgecolor='none', width=1.5, headwidth=6))

    # Caja 1.3: Validación cliente
    b3 = patches.FancyBboxPatch((70, 74), 25, 14.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b3)
    ax.text(82.5, 86, "Validación Sintáctica\n(Cliente / JavaScript)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text_dark)
    ax.text(82.5, 78.5, "• Rangos numéricos válidos\n• Control de campos nulos\n• Alertas visuales inmediatas",
            ha='center', va='center', fontsize=7.5, color='#4A5568')

    # Flecha grande hacia abajo S1 -> S2
    ax.annotate('', xy=(82.5, 69.5), xytext=(82.5, 72),
                arrowprops=dict(facecolor='#2B6CB0', edgecolor='none', width=2, headwidth=7))

    # ==================== SECCIÓN 2: BACKEND LARAVEL & PRIVACIDAD ====================
    rect_s2 = patches.FancyBboxPatch((3, 56), 94, 13, boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor=c_s2, edgecolor=c_s2_b, linewidth=1.5)
    ax.add_patch(rect_s2)
    ax.text(5, 67, "2. BACKEND Y CONTROL DE PRIVACIDAD (Laravel Framework - Ley N° 29733)",
            fontsize=9.5, fontweight='bold', color='#234E52')

    # Caja 2.1: Disociación de datos
    b4 = patches.FancyBboxPatch((5, 57.5), 42, 8.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b4)
    ax.text(26, 63, "Disociación Irreversible de Identidad", ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_text_dark)
    ax.text(26, 59.5, "Se elimina Nombre, DNI y Datos Filiatorios.\nSe genera un identificador anónimo correlativo (PAC-XXX).",
            ha='center', va='center', fontsize=7.2, color='#4A5568')

    # Flecha 2.1 -> 2.2
    ax.annotate('', xy=(50, 61.75), xytext=(47, 61.75),
                arrowprops=dict(facecolor='#4A5568', edgecolor='none', width=1.5, headwidth=6))

    # Caja 2.2: Petición HTTP REST
    b5 = patches.FancyBboxPatch((50, 57.5), 45, 8.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b5)
    ax.text(72.5, 63, "Despacho de Petición HTTP POST Asíncrona", ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_text_dark)
    ax.text(72.5, 59.5, "Endpoint: /predict | Header: Content-Type: application/json\nPayload JSON estructurado con los 8 valores numéricos puros.",
            ha='center', va='center', fontsize=7.2, color='#2B6CB0')

    # Flecha grande hacia abajo S2 -> S3
    ax.annotate('', xy=(50, 53.5), xytext=(50, 56),
                arrowprops=dict(facecolor='#319795', edgecolor='none', width=2, headwidth=7))

    # ==================== SECCIÓN 3: PIPELINE MACHINE LEARNING (PYTHON) ====================
    rect_s3 = patches.FancyBboxPatch((3, 23.5), 94, 29.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor=c_s3, edgecolor=c_s3_b, linewidth=1.5)
    ax.add_patch(rect_s3)
    ax.text(5, 51, "3. MICROSERVICIO DE INFERENCIA PREDICTIVA (Python Flask & Pipeline Scikit-Learn)",
            fontsize=9.5, fontweight='bold', color='#553C9A')

    # Recuadro Interno: Pipeline Serializado (diabetes_pipeline.pkl)
    rect_pipe = patches.FancyBboxPatch((5, 34), 89.5, 15.5, boxstyle="round,pad=0.4,rounding_size=1",
                                      facecolor='#FFFFFF', edgecolor='#9F7AEA', linewidth=1.2, linestyle='--')
    ax.add_patch(rect_pipe)
    ax.text(8, 47.5, "PIPELINE SERIALIZADO EN MEMORIA (diabetes_pipeline.pkl - Joblib)",
            fontsize=8, fontweight='bold', color='#6B46C1')

    # Sub-paso 1: SimpleImputer
    p1 = patches.FancyBboxPatch((7, 36), 27, 10, boxstyle="round,pad=0.2,rounding_size=0.8",
                               facecolor='#FAF5FF', edgecolor='#CBD5E0', linewidth=1)
    ax.add_patch(p1)
    ax.text(20.5, 43, "1. SimpleImputer", ha='center', va='center', fontsize=8, fontweight='bold', color='#44337A')
    ax.text(20.5, 38.5, "Estrategia: Mediana\nImputa ceros biológicamente\nimposibles (sin fuga de datos)",
            ha='center', va='center', fontsize=6.8, color='#4A5568')

    # Flecha p1 -> p2
    ax.annotate('', xy=(37, 41), xytext=(34, 41),
                arrowprops=dict(facecolor='#805AD5', edgecolor='none', width=1.5, headwidth=5))

    # Sub-paso 2: StandardScaler
    p2 = patches.FancyBboxPatch((37, 36), 27, 10, boxstyle="round,pad=0.2,rounding_size=0.8",
                               facecolor='#FAF5FF', edgecolor='#CBD5E0', linewidth=1)
    ax.add_patch(p2)
    ax.text(50.5, 43, "2. StandardScaler", ha='center', va='center', fontsize=8, fontweight='bold', color='#44337A')
    ax.text(50.5, 38.5, "Estandarización Z-Score\nz = (x - μ) / σ\nParámetros de entrenamiento",
            ha='center', va='center', fontsize=6.8, color='#4A5568')

    # Flecha p2 -> p3
    ax.annotate('', xy=(67, 41), xytext=(64, 41),
                arrowprops=dict(facecolor='#805AD5', edgecolor='none', width=1.5, headwidth=5))

    # Sub-paso 3: RandomForestClassifier
    p3 = patches.FancyBboxPatch((67, 36), 25.5, 10, boxstyle="round,pad=0.2,rounding_size=0.8",
                               facecolor='#FAF5FF', edgecolor='#CBD5E0', linewidth=1)
    ax.add_patch(p3)
    ax.text(79.75, 43, "3. RandomForest", ha='center', va='center', fontsize=8, fontweight='bold', color='#44337A')
    ax.text(79.75, 38.5, "Ensamble (100-200 árboles)\nVotación agregada\nHiperparámetros optimizados",
            ha='center', va='center', fontsize=6.8, color='#4A5568')

    # Línea de decisión y umbral
    b_prob = patches.FancyBboxPatch((5, 25), 38, 7.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                   facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b_prob)
    ax.text(24, 29.5, "Cálculo de Probabilidad Posterior", ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_text_dark)
    ax.text(24, 26.5, "predict_proba() -> P(Diabetes = 1 | X)", ha='center', va='center', fontsize=7.2, color='#2B6CB0', family='monospace')

    # Flecha prob -> umbral
    ax.annotate('', xy=(46, 28.75), xytext=(43, 28.75),
                arrowprops=dict(facecolor='#4A5568', edgecolor='none', width=1.5, headwidth=5))

    b_umb = patches.FancyBboxPatch((46, 25), 48.5, 7.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                  facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b_umb)
    ax.text(70.25, 29.5, "Regla de Decisión Clínica por Umbral (τ = 0.50)", ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_text_dark)
    ax.text(70.25, 26.5, "Si P ≥ 0.50 -> Clase 1 (Alto Riesgo) | Si P < 0.50 -> Clase 0 (Bajo Riesgo)\nEmpaquetado JSON {prediction, probability_diabetes, threshold}",
            ha='center', va='center', fontsize=6.8, color='#4A5568')

    # Flecha grande hacia abajo S3 -> S4
    ax.annotate('', xy=(50, 21), xytext=(50, 23.5),
                arrowprops=dict(facecolor='#805AD5', edgecolor='none', width=2, headwidth=7))

    # ==================== SECCIÓN 4: PERSISTENCIA Y SOPORTE CLÍNICO ====================
    rect_s4 = patches.FancyBboxPatch((3, 3), 94, 17.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor=c_s4, edgecolor=c_s4_b, linewidth=1.5)
    ax.add_patch(rect_s4)
    ax.text(5, 18.5, "4. INTEGRACIÓN, PERSISTENCIA Y SOPORTE A LA DECISIÓN CLÍNICA (CDSS)",
            fontsize=9.5, fontweight='bold', color='#C05621')

    # Caja 4.1: Supabase
    b6 = patches.FancyBboxPatch((5, 4.5), 26, 12.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b6)
    ax.text(18, 14, "Persistencia y Auditoría\n(Supabase / PostgreSQL)", ha='center', va='center', fontsize=8, fontweight='bold', color=c_text_dark)
    ax.text(18, 8, "• Registro inmutable de triaje\n• Timestamp y trazabilidad\n• Almacenamiento disociado",
            ha='center', va='center', fontsize=7, color='#4A5568')

    # Caja 4.2: Explicabilidad Gemini
    b7 = patches.FancyBboxPatch((34, 4.5), 29, 12.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b7)
    ax.text(48.5, 14, "Explicabilidad Asistida\n(Google Gemini API)", ha='center', va='center', fontsize=8, fontweight='bold', color=c_text_dark)
    ax.text(48.5, 8, "• Síntesis en lenguaje claro\n• Factores de riesgo clave\n• No prescriptiva (solo apoyo)\n• Cero datos identificables",
            ha='center', va='center', fontsize=7, color='#4A5568')

    # Caja 4.3: Panel Médico
    b8 = patches.FancyBboxPatch((66, 4.5), 29, 12.5, boxstyle="round,pad=0.3,rounding_size=1",
                               facecolor='#FFFFFF', edgecolor='#A0AEC0', linewidth=1)
    ax.add_patch(b8)
    ax.text(80.5, 14, "Panel de Apoyo Médico\n(Interfaz de Consulta)", ha='center', va='center', fontsize=8, fontweight='bold', color='#C53030')
    ax.text(80.5, 8, "• Semáforo visual de alerta\n• Porcentaje de riesgo estimado\n• DESCARGO: Decisión final\n  a cargo del médico tratante",
            ha='center', va='center', fontsize=7, color='#2D3748')

    plt.tight_layout()
    output_path = "/Users/edwardstevenquispesanchez/Documents/TESIS - TÍTULO/PROYECTOS_TITULO/appml_tesis/diagrama_flujo_prediccion_pipeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Diagrama guardado en: {output_path}")

if __name__ == '__main__':
    crear_diagrama_flujo()
