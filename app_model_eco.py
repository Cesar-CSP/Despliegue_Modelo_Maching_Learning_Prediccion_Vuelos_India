from flask import Flask, jsonify, request, render_template
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

app = Flask(__name__)

# Cargar el modelo
model_eco = joblib.load("xgb_eco.joblib")

# 1. Página de documentación (Home)
@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

# 2. Herramienta visual de predicción
@app.route('/predict', methods=['GET'])
def predict_visual():
    return render_template("landing_page.html")

# 3. La lógica de la API se mantiene igual
@app.route('/api/v1/predict', methods=['GET'])
def predict():
    fields = { 'airline': request.args.get('airline'),
              'from': request.args.get('from'),
              'to': request.args.get('to'),
              'duration(h)': request.args.get('duration(h)'),
              'dep_time_cat': request.args.get('dep_time_cat'),
              'arr_time_cat': request.args.get('arr_time_cat'),
              'stop_num': request.args.get('stop_num'),
              'days_left': request.args.get('days_left') }

    # Reemplazar la parte de la imputación manual por esta:
    missing = []
    
    for key, val in fields.items():
        if val is None or val == "":
            missing.append(key)
            # En lugar de -1, asignamos None para que el Imputer del Pipeline trabaje
            fields[key] = None 
    
    # Convertir numéricos (usando pd.to_numeric para manejar los None correctamente)
    fields['duration(h)'] = pd.to_numeric(fields['duration(h)'], errors='coerce')
    fields['stop_num'] = pd.to_numeric(fields['stop_num'], errors='coerce')
    fields['days_left'] = pd.to_numeric(fields['days_left'], errors='coerce')

    # Crear DataFrame
    input_data = pd.DataFrame([fields])

    # Predecir
    prediction_eco = model_eco.predict(input_data)

    response_eco = {'predictions': float(prediction_eco[0])}
    if missing:
        response_eco['warning'] = f"Missing values replaced for: {', '.join(missing)}"

    return jsonify(response_eco)
    

if __name__ == '__main__':
    app.run()
