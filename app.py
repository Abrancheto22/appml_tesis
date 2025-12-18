import joblib
# import pandas as pd  <--- ELIMINADO PARA AHORRAR ESPACIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo/random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'modelo/scaler.pkl')

model = None
scaler = None

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

def load_model_and_scaler():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Modelos cargados correctamente.")
    except Exception as e:
        print(f"Error cargando modelos: {e}")

load_model_and_scaler()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API Online"})

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Modelo no cargado."}), 500

    try:
        data = request.get_json(force=True)
        
        # 1. TRADUCIR CLAVES (Español -> Inglés)
        data_translated = {}
        for key, value in data.items():
            if key in KEYS_MAPPING:
                data_translated[KEYS_MAPPING[key]] = value
            else:
                data_translated[key] = value

        # 2. DEFINIR EL ORDEN EXACTO (Crucial para el modelo)
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Validar
        if not all(feature in data_translated for feature in expected_features):
             return jsonify({"error": f"Faltan datos. Se esperaban: {expected_features}"}), 400

        # 3. CREAR ARRAY (En lugar de DataFrame usamos una lista de listas)
        # Esto hace lo mismo que Pandas pero sin ocupar 100MB de espacio
        input_data = [[ data_translated[feature] for feature in expected_features ]]

        # 4. Escalar y Predecir
        scaled_input = scaler.transform(input_data)
        
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        result = {
            "prediction": int(prediction),
            "probability": float(prediction_proba[1]),
            "message": "Predicción exitosa"
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)