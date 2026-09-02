import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'modelo/diabetes_pipeline.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'modelo/random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'modelo/scaler.pkl')

pipeline = None
model = None
scaler = None

CLASSIFICATION_THRESHOLD = 0.5

KEYS_MAPPING = {
    'embarazos': 'Pregnancies',
    'glucosa': 'Glucose',
    'presion_sanguinea': 'BloodPressure',
    'grosor_piel': 'SkinThickness',
    'insulina': 'Insulin',
    'bmi': 'BMI',
    'pedigree': 'DiabetesPedigreeFunction',
    'edad': 'Age'
}

EXPECTED_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]

def load_models():
    global pipeline, model, scaler
    try:
        if os.path.exists(PIPELINE_PATH):
            pipeline = joblib.load(PIPELINE_PATH)
            logger.info("Pipeline cargado correctamente.")
        else:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info("Modelo y scaler cargados por separado (compatibilidad).")
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")

load_models()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "API Online",
        "pipeline_loaded": pipeline is not None,
        "model_loaded": model is not None,
        "threshold": CLASSIFICATION_THRESHOLD
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline and not model:
        return jsonify({"error": "Modelo no cargado en el servidor"}), 500

    try:
        data = request.get_json(force=True)

        values_list = []
        debug_received = data
        debug_interpreted = {}

        for feature in EXPECTED_FEATURES:
            esp_key = None
            for k, v in KEYS_MAPPING.items():
                if v == feature:
                    esp_key = k
                    break

            value = data.get(feature) or data.get(esp_key)
            if value is None:
                logger.warning(f"Feature {feature} faltante, usando 0.0")
                value = 0.0

            values_list.append(float(value))
            debug_interpreted[feature] = float(value)

        try:
            import pandas as pd
            input_data = pd.DataFrame([values_list], columns=EXPECTED_FEATURES)
        except ImportError:
            input_data = np.array([values_list])

        if pipeline:
            prediction_class = pipeline.predict(input_data)[0]
            prediction_prob = pipeline.predict_proba(input_data)[0]
        else:
            if model is None or scaler is None:
                return jsonify({"error": "Modelo o scaler no cargados"}), 500
            scaled = scaler.transform(input_data)
            prediction_class = model.predict(scaled)[0]
            prediction_prob = model.predict_proba(scaled)[0]

        prob_diabetes = float(prediction_prob[1])
        prob_no_diabetes = float(prediction_prob[0])
        result_class = 1 if prob_diabetes >= CLASSIFICATION_THRESHOLD else 0
        diagnosis = "Diabetes" if result_class == 1 else "No Diabetes"

        return jsonify({
            "prediction": result_class,
            "resultado": prob_diabetes,
            "probability_diabetes": prob_diabetes,
            "probability_no_diabetes": prob_no_diabetes,
            "diagnosis": diagnosis,
            "threshold_used": CLASSIFICATION_THRESHOLD,
            "message": "Predicción exitosa",
            "debug_received": debug_received,
            "debug_interpreted": debug_interpreted,
        }), 200

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route('/metrics', methods=['GET'])
def metrics_info():
    return jsonify({
        "model_type": "RandomForestClassifier",
        "metrics_required": [
            "Accuracy", "Sensitivity/Recall", "Specificity",
            "PPV/Precision", "NPV", "F1-Score", "ROC-AUC",
            "Brier Score", "Confusion Matrix"
        ],
        "note": "Las metricas se calculan en el script de evaluacion, no en la API."
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
