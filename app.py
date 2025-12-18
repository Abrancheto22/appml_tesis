import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Inicializar Flask
app = Flask(__name__)
# Permitir CORS para cualquier origen (Solución a tu error de React)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURACIÓN DE RUTAS ---
# Vercel necesita rutas absolutas para encontrar los archivos .pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo/random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'modelo/scaler.pkl')

model = None
scaler = None

# --- DICCIONARIO DE TRADUCCIÓN ---
# Esto conecta tu Frontend (Español) con tu Modelo (Inglés)
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
        print(f"Error cargando modelos en: {MODEL_PATH}")
        print(f"Detalle: {e}")
        model = None

# Cargar al inicio
load_model_and_scaler()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API Online", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "El modelo no se cargó correctamente en el servidor."}), 500

    try:
        # 1. Obtener datos
        data = request.get_json(force=True)
        
        # 2. TRADUCIR LAS CLAVES (De Español a Inglés para el modelo)
        # Si llegan en español, las renombramos. Si llegan en inglés, las dejamos.
        data_translated = {}
        for key, value in data.items():
            if key in KEYS_MAPPING:
                data_translated[KEYS_MAPPING[key]] = value
            else:
                data_translated[key] = value

        # 3. Crear DataFrame con las columnas esperadas
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Verificar que existan todos los datos necesarios
        if not all(feature in data_translated for feature in expected_features):
             return jsonify({"error": f"Faltan datos. Se esperaban: {expected_features}"}), 400

        # Crear DataFrame en el orden correcto
        input_df = pd.DataFrame([data_translated], columns=expected_features)

        # 4. Escalar
        scaled_input = scaler.transform(input_df)

        # 5. Predecir
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        # 6. Respuesta
        result = {
            "prediction": int(prediction), # 0 o 1
            "probability": float(prediction_proba[1]), # Probabilidad de tener diabetes
            "message": "Predicción exitosa"
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)