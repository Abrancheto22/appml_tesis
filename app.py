"""
API REST - Motor Predictivo de Diabetes Tipo 2
Sistema WebPrediccionDT2 | Universidad Nacional de Trujillo - 2026
Autores: Br. Abraham Ordoñez Reyes & Br. Edward Quispe Sanchez
Asesor: Ms. Marcelino Torres Villanueva

Modelo: Random Forest Classifier (Pipeline Scikit-learn)
Dataset: Pima Indians Diabetes Database / NIDDK (N=10,000 balanceado)
Partición declarada: 60% Train / 20% Val / 20% Test.
Nota: el conjunto de 10,000 filas fue balanceado con SMOTE antes de particionarlo;
las métricas reportadas no deben tratarse como una evaluación independiente.
"""

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import hmac
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuración de la aplicación
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
CORS(app, resources={r"/*": {"origins": allowed_origins}})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas de archivos del modelo
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'modelo', 'diabetes_pipeline.pkl')

# ---------------------------------------------------------------------------
# Constantes clínicas del sistema
# ---------------------------------------------------------------------------
# Umbral declarado por el proyecto; necesita validación clínica independiente.
CLASSIFICATION_THRESHOLD = 0.5
TRAINING_AGE_MIN = 21
TRAINING_AGE_MAX = 81
MODEL_AGE_MIN = int(os.getenv("ML_AGE_MIN", str(TRAINING_AGE_MIN)))
MODEL_AGE_MAX = int(os.getenv("ML_AGE_MAX", str(TRAINING_AGE_MAX)))
if not TRAINING_AGE_MIN <= MODEL_AGE_MIN <= MODEL_AGE_MAX <= TRAINING_AGE_MAX:
    raise ValueError(
        "ML_AGE_MIN/ML_AGE_MAX deben estar dentro del rango de edades del modelo actual "
        f"({TRAINING_AGE_MIN}–{TRAINING_AGE_MAX}). Para ampliarlo, reentrene y valide otro modelo."
    )

# Hiperparámetros del modelo (solo documentación; ya están dentro del .pkl)
MODEL_INFO = {
    "algorithm":       "RandomForestClassifier",
    "n_estimators":    200,
    "max_depth":       10,
    "criterion":       "entropy",
    "class_weight":    "balanced",
    "min_samples_leaf": 8,
    "min_samples_split": 16,
    "dataset":         "768 registros Pima, balanceados con SMOTE hasta 10,000 filas",
    "particion":       "60/20/20 estratificada después del balanceo con SMOTE",
    "metricas_test":   {
        "N_test":      2000,
        "Accuracy":    "90.85%",
        "Precision_PPV": "87.37%",
        "Recall_Sens": "95.50%",
        "Specificity": "86.20%",
        "F1_Score":    0.9126,
        "ROC_AUC":     0.9722,
        "MCC":         0.8206,
        "Threshold":   CLASSIFICATION_THRESHOLD
    }
}

# Mapeo español -> inglés para las 8 variables del dataset PIMA/NIDDK
KEYS_MAPPING = {
    'embarazos':       'Pregnancies',
    'glucosa':         'Glucose',
    'presion_sanguinea': 'BloodPressure',
    'grosor_piel':     'SkinThickness',
    'insulina':        'Insulin',
    'bmi':             'BMI',
    'pedigree':        'DiabetesPedigreeFunction',
    'edad':            'Age'
}

# Orden exacto requerido por el Pipeline serializado
EXPECTED_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]

# Variable global del pipeline
pipeline = None

# ---------------------------------------------------------------------------
# Carga del Pipeline
# ---------------------------------------------------------------------------
def load_pipeline():
    """
    Carga el Pipeline serializado (SimpleImputer + StandardScaler + RF).
    El script que generó los datos aplica SMOTE antes de la partición; por
    tanto, las métricas de esa partición requieren una evaluación independiente.
    """
    global pipeline
    try:
        if not os.path.exists(PIPELINE_PATH):
            logger.error(f"Pipeline no encontrado en: {PIPELINE_PATH}")
            return

        pipeline = joblib.load(PIPELINE_PATH)

        # Verificar que es un Pipeline de Scikit-learn válido
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"El archivo cargado no es un Pipeline de Scikit-learn (tipo: {type(pipeline)})")

        # Verificar que el Pipeline tiene los pasos esperados
        step_names = [name for name, _ in pipeline.steps]
        expected_steps = ['imputer', 'scaler', 'classifier']
        if not all(s in step_names for s in expected_steps):
            raise ValueError(f"Pipeline incompleto. Pasos encontrados: {step_names}")

        rf = pipeline.named_steps['classifier']
        scaler = pipeline.named_steps['scaler']

        logger.info("=" * 60)
        logger.info("Pipeline cargado correctamente.")
        logger.info(f"  n_estimators : {rf.n_estimators}")
        logger.info(f"  max_depth    : {rf.max_depth}")
        logger.info(f"  criterion    : {rf.criterion}")
        logger.info(f"  class_weight : {rf.class_weight}")
        logger.info(f"  n_features   : {rf.n_features_in_}")
        logger.info(f"  Scaler mean  : {np.round(scaler.mean_, 3)}")
        logger.info(f"  Threshold    : {CLASSIFICATION_THRESHOLD}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error al cargar el Pipeline: {e}")
        pipeline = None


load_pipeline()


# ---------------------------------------------------------------------------
# Endpoint raíz — Health Check
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    """Health check y estado del sistema."""
    return jsonify({
        "status":          "API Online",
        "sistema":         "WebPrediccionDT2 - Motor Predictivo de Diabetes Tipo 2",
        "institucion":     "Universidad Nacional de Trujillo - Ing. de Sistemas",
        "pipeline_loaded": pipeline is not None,
        "threshold":       CLASSIFICATION_THRESHOLD,
        "model_age_range": {"min": MODEL_AGE_MIN, "max": MODEL_AGE_MAX},
        "nota":            (
            "Sistema de Soporte a la Decisión Clínica (CDSS). "
            "La predicción es una herramienta de apoyo al tamizaje preventivo. "
            "La decisión diagnóstica y terapéutica definitiva corresponde "
            "exclusivamente al médico colegiado tratante (Ley N° 26842)."
        )
    })


# ---------------------------------------------------------------------------
# Endpoint /predict — Predicción principal
# ---------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe los 8 biomarcadores fisiológicos del paciente y devuelve
    la probabilidad de riesgo de Diabetes Tipo 2 estimada por el modelo RF.

    Se exigen las 8 variables. El Pipeline solo imputa valores NaN que hayan
    llegado al modelo; los ceros se interpretan como valores numéricos, igual
    que en el dataset usado por entrenamiento_mejorado.py.

    Body JSON esperado (nombres en español o inglés). Todas las variables
    deben incluirse y cumplir los rangos configurados; no se imputan campos
    omitidos ni se reinterpretan ceros como datos faltantes.
    {
        "embarazos": 2,   "glucosa": 120,  "presion_sanguinea": 70,
        "grosor_piel": 25, "insulina": 100, "bmi": 28.5,
        "pedigree": 0.45,  "edad": 35
    }
    """
    if pipeline is None:
        logger.error("Intento de predicción con pipeline no cargado.")
        return jsonify({
            "error": "El modelo predictivo no está disponible en el servidor.",
            "detalle": f"Verifique que exista el archivo: {PIPELINE_PATH}"
        }), 503

    api_key = os.getenv("ML_API_KEY", "").strip()
    if not api_key:
        logger.error("ML_API_KEY no está configurada; se rechaza la inferencia.")
        return jsonify({"error": "El servicio de predicción no está configurado."}), 503

    authorization = request.headers.get("Authorization", "")
    supplied_key = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(supplied_key, api_key):
        return jsonify({"error": "No autorizado."}), 401

    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "No se recibió un cuerpo JSON válido."}), 400

        # ----------------------------------------------------------------
        # Extracción y validación de las 8 variables clínicas
        # ----------------------------------------------------------------
        values_list      = []
        allowed_ranges = {
            'Pregnancies': (0, 20),
            'Glucose': (0, 500),
            'BloodPressure': (0, 180),
            'SkinThickness': (0, 100),
            'Insulin': (0, 1000),
            'BMI': (0, 100),
            'DiabetesPedigreeFunction': (0, 5),
            'Age': (MODEL_AGE_MIN, MODEL_AGE_MAX),
        }

        for feature in EXPECTED_FEATURES:
            # Buscar la clave en español correspondiente
            esp_key = next((k for k, v in KEYS_MAPPING.items() if v == feature), None)

            # Buscar en inglés primero, luego en español
            raw_value = data.get(feature)
            if raw_value is None and esp_key:
                raw_value = data.get(esp_key)

            if raw_value is None:
                return jsonify({"error": f"Falta el campo requerido: {feature}."}), 400

            try:
                raw_value = float(raw_value)
            except (ValueError, TypeError):
                return jsonify({"error": f"El campo '{feature}' debe ser numérico."}), 400

            minimum, maximum = allowed_ranges[feature]
            if not np.isfinite(raw_value) or not minimum <= raw_value <= maximum:
                if feature == 'Age':
                    return jsonify({
                        "error": (
                            f"Los datos de origen del modelo incluyen edades entre "
                            f"{MODEL_AGE_MIN} y {MODEL_AGE_MAX} años. "
                            "Quedar fuera de este rango no descarta diabetes; "
                            "se requiere evaluación clínica."
                        )
                    }), 422
                return jsonify({"error": f"El campo '{feature}' está fuera del rango permitido."}), 422

            values_list.append(raw_value)

        # ----------------------------------------------------------------
        # Construcción del DataFrame con el orden exacto del Pipeline
        # ----------------------------------------------------------------
        input_df = pd.DataFrame([values_list], columns=EXPECTED_FEATURES)

        # ----------------------------------------------------------------
        # Inferencia con el Pipeline (imputa + escala + predice internamente)
        # ----------------------------------------------------------------
        prediction_prob  = pipeline.predict_proba(input_df)[0]
        prob_diabetes    = float(prediction_prob[1])
        prob_no_diabetes = float(prediction_prob[0])

        # Aplicar el umbral de clasificación configurado
        result_class = 1 if prob_diabetes >= CLASSIFICATION_THRESHOLD else 0
        diagnosis    = "Diabetes" if result_class == 1 else "No Diabetes"

        # Estratificación de nivel de riesgo para Gemini / vista clínica
        if prob_diabetes < 0.30:
            risk_level, risk_desc = "BAJO",    "Riesgo bajo de Diabetes Tipo 2."
        elif prob_diabetes < 0.50:
            risk_level, risk_desc = "MODERADO", "Riesgo moderado. Recomendable seguimiento preventivo."
        elif prob_diabetes < 0.70:
            risk_level, risk_desc = "ALTO",     "Riesgo alto. Evaluación clínica prioritaria."
        else:
            risk_level, risk_desc = "MUY ALTO", "Riesgo muy alto. Atención clínica inmediata recomendada."

        logger.info(
            f"Predicción: prob={prob_diabetes:.4f} | clase={result_class} | "
            f"riesgo={risk_level}"
        )

        return jsonify({
            # Campos principales consumidos por Laravel
            "prediction":          result_class,
            "resultado":           prob_diabetes,
            "probability_diabetes":    prob_diabetes,
            "probability_no_diabetes": prob_no_diabetes,
            "diagnosis":           diagnosis,
            "threshold_used":      CLASSIFICATION_THRESHOLD,
            # Nivel de riesgo para la vista clínica y Gemini
            "risk_level":          risk_level,
            "risk_description":    risk_desc,
            "message":             "Predicción procesada exitosamente.",
            # Advertencia clínica obligatoria (Ley N° 26842)
            "aviso_legal":         (
                "Este resultado es una estimación de apoyo al tamizaje clínico "
                "generada por un Sistema de Soporte a la Decisión (CDSS). "
                "No reemplaza el juicio ni la prescripción del médico tratante."
            ),
        }), 200

    except Exception as e:
        logger.error("Error inesperado al ejecutar la predicción.", exc_info=True)
        return jsonify({"error": "No se pudo procesar la predicción."}), 500


# ---------------------------------------------------------------------------
# Endpoint /metrics — Información del modelo
# ---------------------------------------------------------------------------
@app.route('/metrics', methods=['GET'])
def metrics_info():
    """Devuelve la ficha técnica del modelo (Model Card)."""
    if pipeline is None:
        return jsonify({"error": "Pipeline no cargado"}), 503

    rf     = pipeline.named_steps['classifier']
    scaler = pipeline.named_steps['scaler']

    return jsonify({
        "model_card": {
            "nombre":      "Random Forest Classifier para Tamizaje de Diabetes Tipo 2",
            "version":     "2.0 - Pipeline Scikit-learn",
            "dataset":     MODEL_INFO["dataset"],
            "particion":   MODEL_INFO["particion"],
            "hiperparametros": {
                "n_estimators":      rf.n_estimators,
                "max_depth":         rf.max_depth,
                "criterion":         rf.criterion,
                "class_weight":      rf.class_weight,
                "min_samples_leaf":  rf.min_samples_leaf,
                "min_samples_split": rf.min_samples_split,
                "max_features":      rf.max_features,
            },
            "metricas_test_reportadas": MODEL_INFO["metricas_test"],
            "variables_entrada":   EXPECTED_FEATURES,
            "rango_edad_modelo": {"min": MODEL_AGE_MIN, "max": MODEL_AGE_MAX},
            "limitaciones_validacion": (
                "El conjunto se balanceó con SMOTE antes de particionarlo; las métricas reportadas "
                "no constituyen una validación independiente. El dataset de origen no incluye "
                "edades fuera del rango indicado."
            ),
            "pipeline_pasos": [
                "1. SimpleImputer(strategy='median') — imputa valores ausentes; los ceros se conservan",
                "2. StandardScaler() — estandariza con media y desv. del entrenamiento",
                "3. RandomForestClassifier — ensemble de 200 árboles de decisión"
            ],
            "uso_permitido":  "Apoyo preventivo al tamizaje clínico (CDSS).",
            "uso_prohibido":  [
                "Diagnóstico autónomo sin supervisión médica.",
                "Prescripción terapéutica automatizada.",
                "Interpretar una estimación fuera del rango etario configurado como validada.",
                "Transmisión de identificadores personales a servicios externos."
            ],
            "marco_legal": {
                "confidencialidad": "Ley N° 29733 — Protección de Datos Personales del Perú",
                "acto_medico":      "Ley N° 26842 — Ley General de Salud del Perú"
            }
        }
    })


# ---------------------------------------------------------------------------
# Endpoint /health — Verificación detallada del Pipeline
# ---------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """Verificación detallada del estado del Pipeline cargado."""
    if pipeline is None:
        return jsonify({"status": "ERROR", "detail": "Pipeline no cargado"}), 503

    rf     = pipeline.named_steps['classifier']
    scaler = pipeline.named_steps['scaler']

    return jsonify({
        "status":           "OK",
        "pipeline_steps":   [name for name, _ in pipeline.steps],
        "n_estimators":     rf.n_estimators,
        "n_features_in":    rf.n_features_in_,
        "scaler_mean":      list(np.round(scaler.mean_, 4)),
        "scaler_scale":     list(np.round(scaler.scale_, 4)),
        "threshold":        CLASSIFICATION_THRESHOLD,
        "model_age_range":  {"min": MODEL_AGE_MIN, "max": MODEL_AGE_MAX},
        "pipeline_path":    PIPELINE_PATH
    })


# ---------------------------------------------------------------------------
# Inicio del servidor
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
