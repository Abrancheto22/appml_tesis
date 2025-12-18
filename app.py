import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# Configuración CORS
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Cargamos solo el pipeline (que ya incluye Scaler + Imputer + Modelo)
MODEL_PATH = os.path.join(BASE_DIR, 'modelo/random_forest_model.pkl')

pipeline = None

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

def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("Pipeline cargado correctamente.")
    except Exception as e:
        print(f"Error cargando modelo: {e}")

load_model()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API Online"})

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({"error": "Modelo no cargado"}), 500

    try:
        data = request.get_json(force=True)
        
        # 1. TRADUCIR CLAVES (Español -> Inglés)
        data_translated = {}
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        for key, value in data.items():
            english_key = KEYS_MAPPING.get(key, key)
            data_translated[english_key] = float(value)

        # 2. VALIDAR CAMPOS FALTANTES
        for feature in expected_features:
            if feature not in data_translated:
                data_translated[feature] = np.nan 

        # 3. CREAR DATAFRAME
        input_df = pd.DataFrame([data_translated], columns=expected_features)
        
        # --- MEJORA CRÍTICA BASADA EN TU ARCHIVO ANTIGUO ---
        # Estas son las columnas donde un 0 es biológicamente imposible.
        # En tu archivo viejo, las identificabas pero usabas 'pass'.
        # AQUÍ las convertimos a NaN para que el Pipeline las arregle matemáticamente.
        cols_bad_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in cols_bad_zeros:
            val = input_df[col].iloc[0]
            if val == 0:
                # Transformamos 0 -> NaN. El Pipeline (SimpleImputer) verá el NaN 
                # y le pondrá el valor promedio automáticamente.
                input_df[col].iloc[0] = np.nan

        # NOTA: 'Pregnancies' NO está en la lista. Si es 0, se queda en 0 (es válido).

        # 4. PREDECIR
        prediction_class = pipeline.predict(input_df)[0]
        prediction_prob = pipeline.predict_proba(input_df)[0]

        result = {
            "prediction": int(prediction_class),
            "resultado": float(prediction_prob[1]), # Probabilidad de Diabetes
            "message": "Predicción exitosa"
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)