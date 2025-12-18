import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# Configuración CORS que permite todo
CORS(app, resources={r"/*": {"origins": "*"}})

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo/random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'modelo/scaler.pkl')

model = None
scaler = None

# Mapeo de Español (React) -> Inglés (Modelo)
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
    return jsonify({"status": "API Online", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Modelo no cargado en el servidor"}), 500

    try:
        data = request.get_json(force=True)
        
        # --- 1. TRADUCCIÓN Y EXTRACCIÓN (DEBUG) ---
        data_processed = {}
        missing_keys = []
        
        # Lista exacta que espera el modelo (en orden)
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Traducimos lo que llega
        for key, value in data.items():
            english_key = KEYS_MAPPING.get(key, key) # Si existe en el mapa lo traduce, sino usa la llave original
            data_processed[english_key] = float(value) # Aseguramos que sea número

        # Verificamos si falta algo
        values_list = []
        for feature in expected_features:
            if feature not in data_processed:
                missing_keys.append(feature)
                values_list.append(0.0) # Rellenar con 0 para que no explote
            else:
                values_list.append(data_processed[feature])

        # --- 2. PREDICCIÓN ---
        # Convertimos a formato lista de listas (2D array)
        input_data = [values_list] 
        
        # Escalamos
        scaled_input = scaler.transform(input_data)
        
        # Predecimos
        prediction_class = model.predict(scaled_input)[0]       # Clase: 0 o 1
        prediction_prob = model.predict_proba(scaled_input)[0]  # Probabilidad: [0.2, 0.8]

        # Tomamos la probabilidad de que sea POSITIVO (Diabetes)
        probabilidad_diabetes = float(prediction_prob[1])

        result = {
            "prediction": int(prediction_class),      # 0 o 1
            "resultado": probabilidad_diabetes,       # 0.85 (Esto es lo que quiere React)
            "message": "Predicción exitosa",
            
            # DATOS DE DEBUG (Para ver qué está llegando)
            "debug_received": data,                   # Lo que envió React
            "debug_interpreted": data_processed,      # Lo que entendió Python
            "debug_missing": missing_keys             # Lo que faltó
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)