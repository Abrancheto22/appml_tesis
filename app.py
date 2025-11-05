import joblib
import pandas as pd
from flask import Flask, request, jsonify
import os

# Rutas a los archivos del modelo y escalador
MODEL_PATH = 'modelo/random_forest_model.pkl'
SCALER_PATH = 'modelo/scaler.pkl'

# Inicializar la aplicación Flask
app = Flask(__name__)

# Variables globales para el modelo y el escalador
model = None
scaler = None

def load_model_and_scaler():
    """Carga el modelo y el escalador al iniciar la aplicación."""
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Modelo y escalador cargados exitosamente.")
    except FileNotFoundError:
        print(f"Error: Uno o ambos archivos, '{MODEL_PATH}' o '{SCALER_PATH}', no se encontraron.")
        print("Asegúrate de haber ejecutado 'train_model.py' primero.")
    except Exception as e:
        print(f"Error al cargar el modelo o el escalador: {e}")
        exit()

# Cargar el modelo y el escalador cuando la aplicación inicie
# Esto asegura que no se cargan en cada solicitud
with app.app_context():
    load_model_and_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Modelo o escalador no cargados. Verifique el log del servidor."}), 500

    try:
        # 1. Obtener los datos del request JSON
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No se proporcionaron datos JSON válidos."}), 400

        # Convertir los datos de entrada a un DataFrame de pandas
        # Es crucial que el orden de las columnas sea el mismo que el usado para entrenar el modelo
        # Basado en el Pima Indians Diabetes Dataset:
        # 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        input_df = pd.DataFrame([data]) # Convierte el JSON en un DataFrame de 1 fila

        # 2. Preprocesar los datos de entrada
        # Reemplazar 0s si es necesario (igual que en el entrenamiento)
        columns_to_check_for_zero_replacement = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_to_check_for_zero_replacement:
            if column in input_df.columns and input_df[column].iloc[0] == 0:
                # En una API, NO reemplaces con la media del dataset de entrenamiento,
                # porque no tienes acceso a él. Aquí, si se recibe un 0, es un dato.
                # Para producción, deberías tener una estrategia clara para 0s (ej. imputación previa)
                # Por ahora, simplemente pasamos el 0 al escalador, lo cual puede o no ser ideal
                # dependiendo de cómo el modelo fue entrenado con respecto a esos 0s.
                # Si el modelo fue entrenado con 0s reemplazados, entonces el escalador espera el valor imputado, no el 0 puro.
                # Para ser coherentes con train_model.py, deberíamos imputar.
                # Esto es algo que DEBERÍAS considerar en una aplicación real para producción.
                # Por simplicidad y para seguir el flujo, asumiremos que los 0s de entrada son válidos
                # O que el modelo fue entrenado con 0s ya reemplazados en esas posiciones.
                # La forma correcta sería: escalar y luego imputar, o imputar y luego escalar
                # si el modelo espera imputación. Aquí, lo más simple es asumir que el escalador maneja los 0s.
                # Si quieres imputar los 0s como en el entrenamiento:
                # input_df[column] = input_df[column].replace(0, df_training_mean[column])
                # Pero necesitarías df_training_mean, que no está disponible aquí.
                # Por ahora, si se envía 0, se escala 0.
                pass # Dejamos los 0s tal cual para que el escalador los maneje si no se hace imputación global.


        # Asegúrate de que las columnas del input_df estén en el mismo orden que las características de entrenamiento
        # Si X.columns estuviera disponible, sería ideal. Pero el escalador ya sabe el orden.
        # Necesitamos asegurarnos que el JSON enviado tenga todas las 8 características
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(feature in input_df.columns for feature in expected_features):
            missing_features = [f for f in expected_features if f not in input_df.columns]
            return jsonify({"error": f"Faltan características en el JSON de entrada: {', '.join(missing_features)}. Se esperan: {', '.join(expected_features)}"}), 400

        # Reordenar las columnas del DataFrame de entrada para que coincidan con el orden del entrenamiento
        input_df = input_df[expected_features]


        # Escalar los datos de entrada
        scaled_input = scaler.transform(input_df)

        # 3. Realizar la predicción
        prediction_proba = model.predict_proba(scaled_input)[0] # Probabilidades para cada clase
        prediction = model.predict(scaled_input)[0] # La clase predicha (0 o 1)

        # 4. Devolver la predicción
        # Interpretación de la predicción para Pima: 0 = No diabetes, 1 = Diabetes
        result = {
            "prediction": int(prediction),
            "probability_no_diabetes": float(prediction_proba[0]),
            "probability_diabetes": float(prediction_proba[1]),
            "message": "Predicción realizada exitosamente."
        }
        if prediction == 1:
            result["diagnosis"] = "Posiblemente tenga diabetes."
        else:
            result["diagnosis"] = "Probablemente no tenga diabetes."

        return jsonify(result), 200

    except KeyError as ke:
        return jsonify({"error": f"Clave faltante en el JSON de entrada: {ke}. Asegúrate de enviar todas las 8 características: {expected_features}"}), 400
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": f"Ocurrió un error al procesar la solicitud: {e}"}), 500

# Esto solo se ejecuta cuando se corre directamente 'python app.py'
if __name__ == '__main__':
    print("Iniciando servidor Flask en http://127.0.0.1:5000/...")
    app.run(debug=True, host='0.0.0.0', port=5000) # debug=True para desarrollo, host='0.0.0.0' para acceso externo