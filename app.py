import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# CONFIGURACIÓN CORS (Permisos totales)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# --- RUTAS ---
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
        # Cargamos AMBOS archivos (Esto es lo que falló antes al intentar cargar solo uno)
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Modelo y Escalador cargados correctamente.")
    except Exception as e:
        print(f"❌ Error fatal cargando archivos: {e}")

load_model_and_scaler()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API Online - Modo Recuperación"})

@app.route('/predict', methods=['POST'])
def predict():
    # Chequeo de seguridad
    if not model or not scaler:
        return jsonify({"error": "El servidor no pudo cargar los modelos .pkl"}), 500

    try:
        data = request.get_json(force=True)
        
        # 1. TRADUCIR Y PREPARAR
        data_translated = {}
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        for key, value in data.items():
            english_key = KEYS_MAPPING.get(key, key)
            data_translated[english_key] = float(value)

        # 2. VALIDAR FALTANTES
        for feature in expected_features:
            if feature not in data_translated:
                data_translated[feature] = 0  # Valor por defecto seguro

        # 3. CREAR DATAFRAME
        input_df = pd.DataFrame([data_translated], columns=expected_features)

        # --- TRUCO CLÍNICO (Imputación Manual) ---
        # Como no pudimos reentrenar, hacemos esto para mejorar la precisión AHORA MISMO.
        # Si un valor es 0 (y no debería serlo), le ponemos el promedio de una persona promedio.
        
        # Glucosa 0 -> 120 (Promedio riesgo)
        if input_df['Glucose'].iloc[0] == 0: input_df['Glucose'].iloc[0] = 120.0
        
        # Presión 0 -> 72 (Normal)
        if input_df['BloodPressure'].iloc[0] == 0: input_df['BloodPressure'].iloc[0] = 72.0
        
        # Piel 0 -> 29 (Mediana)
        if input_df['SkinThickness'].iloc[0] == 0: input_df['SkinThickness'].iloc[0] = 29.0
        
        # BMI 0 -> 32 (Promedio de sobrepeso del dataset)
        if input_df['BMI'].iloc[0] == 0: input_df['BMI'].iloc[0] = 32.0
        
        # Insulina: Si es 0 es difícil saber, lo dejamos o ponemos un valor bajo-medio
        if input_df['Insulin'].iloc[0] == 0: input_df['Insulin'].iloc[0] = 80.0 # Opcional

        # 4. ESCALAR (Usando el scaler viejo que sí tienes)
        scaled_input = scaler.transform(input_df)

        # 5. PREDECIR
        prediction_class = model.predict(scaled_input)[0]
        prediction_prob = model.predict_proba(scaled_input)[0] # [Prob_Sano, Prob_Enfermo]

        result = {
            "prediction": int(prediction_class),
            "resultado": float(prediction_prob[1]), # Probabilidad de Diabetes
            "message": "Predicción exitosa"
        }

        return jsonify(result), 200

    except Exception as e:
        print(f"Error interno: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)