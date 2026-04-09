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

# 2. Herramienta visual de predicción (el HTML que ya tenías)
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


# Enruta la funcion al endpoint /api/v1/retrain
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    global model_eco
    if os.path.exists("data/business.csv"):
        data = pd.read_csv('data/business.csv')
        data.columns = [col.lower() for col in data.columns]

        X_eco = data.drop(columns=['price'])
        y_eco = data['price']

        X_eco_train, X_eco_test, y_eco_train, y_eco_test = train_test_split(X_eco, y_eco, test_size=0.20, random_state=42)

        model_eco.fit(X_eco_train, y_eco_train)
        rmse = root_mean_squared_error(y_eco_test, model_eco.predict(X_eco_test))
        mae = mean_absolute_error(y_eco_test, model_eco.predict(X_eco_test))
        r2 = r2_score(y_eco_test, model_eco.predict(X_eco_test))
        model_eco.fit(X_eco, y_eco)

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAE: {str(mae)}, R2: {str(r2)}"
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"
    

if __name__ == '__main__':
    app.run()
