import joblib
import pandas as pd
import json
from flask import Flask, request, jsonify
import os

# --- Rutas de Archivos ---
MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'scaler.pkl'
IMPUTATION_PATH = 'imputation_values.json' # Archivo con las medias

# Inicializar la aplicación Flask
app = Flask(__name__)

# --- Variables Globales ---
model = None
scaler = None
imputation_values = None
expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'] # Orden de Pima

def load_artifacts():
    """Carga todos los artefactos necesarios al iniciar la aplicación."""
    global model, scaler, imputation_values
    
    print("Iniciando carga de artefactos...")
    try:
        # Cargar modelo
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
        # Cargar escalador
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Archivo de escalador no encontrado: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        
        # Cargar valores de imputación
        if not os.path.exists(IMPUTATION_PATH):
            raise FileNotFoundError(f"Archivo de imputación no encontrado: {IMPUTATION_PATH}")
        with open(IMPUTATION_PATH, 'r') as f:
            imputation_values = json.load(f)
            
        print("Modelo, escalador y valores de imputación cargados exitosamente.")
        
    except Exception as e:
        print(f"Error crítico al cargar artefactos: {e}")
        # En una app real, podrías querer que la app falle si no puede cargar
        # o manejar este estado en el endpoint de predicción.
        model, scaler, imputation_values = None, None, None


# Cargar artefactos al iniciar el contexto de la aplicación
with app.app_context():
    load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si los artefactos están cargados
    if not all([model, scaler, imputation_values]):
        print("Error: Petición de predicción recibida, pero los artefactos no están cargados.")
        return jsonify({"error": "El servidor no está listo. Los artefactos del modelo no se han cargado."}), 503 # Service Unavailable

    try:
        # 1. Obtener los datos del request JSON
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON válidos."}), 400

        # Convertir los datos de entrada a un DataFrame
        input_df = pd.DataFrame([data])

        # --- 2. Preprocesar los datos (¡EL PASO CRÍTICO!) ---

        # 2a. Imputar 0s usando los valores guardados del entrenamiento
        for col, mean_val in imputation_values.items():
            if col in input_df.columns:
                # Reemplaza el 0 con la media de entrenamiento guardada
                input_df[col] = input_df[col].replace(0, mean_val) 
            else:
                print(f"Advertencia: Columna de imputación '{col}' no encontrada en el JSON de entrada.")

        # 2b. Asegurar que todas las características esperadas estén presentes y en el orden correcto
        if not all(feature in input_df.columns for feature in expected_features):
            missing = [f for f in expected_features if f not in input_df.columns]
            return jsonify({"error": f"Faltan características en el JSON: {', '.join(missing)}"}), 400
        
        # Reordenar las columnas para que coincidan con el entrenamiento
        input_df = input_df[expected_features]

        # 2c. Escalar los datos de entrada (YA IMPUTADOS)
        scaled_input = scaler.transform(input_df)

        # 3. Realizar la predicción
        prediction_proba = model.predict_proba(scaled_input)[0] # Probabilidades [prob_0, prob_1]
        # prediction = model.predict(scaled_input)[0]       # <-- LÍNEA ORIGINAL ELIMINADA

        # --- NUEVO: Ajuste de Umbral de Decisión para AumentAR RECALL ---
        # Queremos ser más sensibles a la Clase 1 (Diabetes).
        # Bajamos el umbral del 50% (default) a un valor menor.
        CUSTOM_THRESHOLD = 0.35  # Puedes "jugar" con este valor (0.40, 0.30, etc.)

        if prediction_proba[1] >= CUSTOM_THRESHOLD:
            prediction = 1 # Positivo (Diabetes)
        else:
            prediction = 0 # Negativo (No Diabetes)
        # --- FIN DEL NUEVO BLOQUE ---

        # 4. Devolver la predicción
        result = {
            "prediction": int(prediction), # 0 o 1
            "diagnosis": "Posiblemente tenga diabetes." if prediction == 1 else "Probablemente no tenga diabetes.",
            "probability_no_diabetes": float(prediction_proba[0]),
            "probability_diabetes": float(prediction_proba[1]),
            "message": "Predicción realizada exitosamente."
        }

        return jsonify(result), 200

    except KeyError as ke:
        return jsonify({"error": f"Clave faltante en el JSON de entrada: {ke}. Se esperan: {expected_features}"}), 400
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": f"Ocurrió un error al procesar la solicitud: {e}"}), 500

# Esto solo se ejecuta cuando se corre directamente 'python app.py'
if __name__ == '__main__':
    print("Iniciando servidor Flask en http://127.0.0.1:5000/...")
    # host='0.0.0.0' permite conexiones desde fuera del contenedor/máquina
    # debug=True es solo para desarrollo, cámbialo a False en producción
    app.run(debug=True, host='0.0.0.0', port=5000)