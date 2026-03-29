
# LIBRERÍAS NECESARIAS

# Python estándar
import re
import numpy as np
import pandas as pd

# Visualización 
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn: Transformadores y Preprocesado 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

# Scikit-Learn: Modelos-Regresión
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Scikit-Learn: Modelos-Clasificación 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Scikit-Learn: Métricas Regresión
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Scikit-Learn: Métricas Clasificación
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
)

# Scikit-Learn: Validación 
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    cross_val_score,
    KFold
)

# Scikit-Learn: Selección de features 
from sklearn.feature_selection import mutual_info_regression

# Otros 
from scipy import stats
import joblib
import os
import sys
sys.path.append(os.path.abspath("./src/utils"))

pd.options.mode.copy_on_write = True
pd.set_option("future.no_silent_downcasting", True)


os.chdir(os.path.dirname(__file__))

df_eco = pd.read_csv("data/economy.csv")
df_bus = pd.read_csv("data/business.csv")

# Transformer 1: Convertir duración
class DurationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        def convert(x):
            h = re.search(r'(\d+)h', x)
            m = re.search(r'(\d+)m', x)
            hours = int(h.group(1)) if h else 0
            minutes = int(m.group(1)) if m else 0
            return hours + minutes/60
        X["duration(h)"] = X["time_taken"].apply(convert)
        return X

# Transformer 2: Categorizar horas
class TimeCategoryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["dep_time"] = pd.to_datetime(X["dep_time"], format="%H:%M")
        X["arr_time"] = pd.to_datetime(X["arr_time"], format="%H:%M")

        def categorize(h):
            if 0 <= h < 4: return "Late Night"
            if 4 <= h < 8: return "Early Morning"
            if 8 <= h < 12: return "Morning"
            if 12 <= h < 16: return "Afternoon"
            if 16 <= h < 20: return "Evening"
            return "Night"

        X["dep_time_cat"] = X["dep_time"].dt.hour.apply(categorize)
        X["arr_time_cat"] = X["arr_time"].dt.hour.apply(categorize)
        return X

# Transformer 3: Limpiar stops
class StopCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        def clean(x):
            x = x.lower().strip()
            if "non" in x: return "non-stop"
            m = re.search(r'(\d+)', x)
            if m: return f"{m.group(1)}-stop"
            return "unknown"
        X["stop"] = X["stop"].apply(clean)
        X["stop_num"] = X["stop"].replace({
            "non-stop": 0,
            "1-stop": 1,
            "2-stop": 2
        }).astype(int)
        return X

# Transformer 4: Calcular days_left
class DaysLeftTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scraping_date="10-02-2022"):
        self.scraping_date = pd.to_datetime(scraping_date, format="%d-%m-%Y")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["date"] = pd.to_datetime(X["date"], format="%d-%m-%Y", errors="coerce")
        X["days_left"] = (X["date"] - self.scraping_date).dt.days
        return X

# Transformer 5: Eliminar columnas irrelevantes
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=[c for c in self.cols_to_drop if c in X.columns])

# Transformer 6: Convertir price a float
class PriceToFloat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        # convertir a float
        X["price"] = (
            X["price"]
            .astype(str)
            .str.replace(",", "")
            .astype(float)
        )

        # renombrar con unidades
        X = X.rename(columns={"price": "price (INR)"})

        return X

# Tranformer 7: Agrupar las aerolíneas
class AirlineGrouper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rare_airlines = ["SpiceJet", "StarAir", "Trujet", "AirAsia"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["airline"] = X["airline"].replace(self.rare_airlines, "Other")
        return X

# Tranformer 8: Eliminar los duplicados
class DropDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop_duplicates().reset_index(drop=True)
    
# Pipeline ETL completo dentro de sklearn
cols_to_drop = [
    "dep_time", "arr_time", "date", "time_taken",
    "ch_code", "num_code", "dep_hour", "arr_hour",
    "flight", "stop"
]

etl_pipeline = Pipeline([
    ("price_float", PriceToFloat()),
    ("duration", DurationTransformer()),
    ("time_cat", TimeCategoryTransformer()),
    ("stop_clean", StopCleaner()),
    ("days_left", DaysLeftTransformer()),
    ("drop_cols", ColumnDropper(cols_to_drop)),
    ("airline_group", AirlineGrouper()),
    ("drop_dupes", DropDuplicates()),
])

df_eco_clean = etl_pipeline.fit_transform(df_eco)
df_bus_clean = etl_pipeline.fit_transform(df_bus)

# Split Train y Test
# Economy
df_eco_ml = df_eco_clean.copy()
# Business
df_bus_ml = df_bus_clean.copy()

df_eco_ml["price_bin"] = pd.qcut(df_eco_ml["price (INR)"], q=10, duplicates="drop")
X_eco = df_eco_ml.drop(columns=["price_bin", "price (INR)"])
y_eco = df_eco_ml["price (INR)"]

df_bus_ml["price_bin"] = pd.qcut(df_bus_ml["price (INR)"], q=10, duplicates="drop")
X_bus = df_bus_ml.drop(columns=["price_bin", "price (INR)"])
y_bus = df_bus_ml["price (INR)"]

X_eco_train, X_eco_test, y_eco_train, y_eco_test = train_test_split(
    X_eco, y_eco, test_size=0.2, random_state=42, stratify = df_eco_ml["price_bin"]
)

X_bus_train, X_bus_test, y_bus_train, y_bus_test = train_test_split(
    X_bus, y_bus, test_size=0.2, random_state=42, stratify = df_bus_ml["price_bin"]
)

features_num_reg_eco = ["duration(h)", "days_left", "stop_num"]
features_cat_reg_eco = [col for col in X_eco_train.columns if col not in features_num_reg_eco]
features_num_reg_bus = ["duration(h)", "days_left", "stop_num"]
features_cat_reg_bus = [col for col in X_bus_train.columns if col not in features_num_reg_bus]


# Tratamiento de features
preprocessor_trees_eco = ColumnTransformer([
    ("procesar_cat_OH", OneHotEncoder(handle_unknown = "ignore"), features_cat_reg_eco)
], remainder = "passthrough")

preprocessor_trees_bus = ColumnTransformer([
    ("procesar_cat_OH", OneHotEncoder(handle_unknown = "ignore"), features_cat_reg_bus)
], remainder = "passthrough")

# Modelo
xgb_eco = Pipeline([
    ("prep", preprocessor_trees_eco),
    ("model", XGBRegressor(
        max_depth = 5,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_bus = Pipeline([
    ("prep", preprocessor_trees_bus),
    ("model", XGBRegressor(
        max_depth = 5,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_eco.fit(X_eco_train, y_eco_train)
xgb_bus.fit(X_bus_train, y_bus_train)

print("Economy")
print("RMSE Test: ", root_mean_squared_error(y_eco_test, xgb_eco.predict(X_eco_test)))
print("MAE Test: ", mean_absolute_error(y_eco_test, xgb_eco.predict(X_eco_test)))
print("R2 Test: ", r2_score(y_eco_test, xgb_eco.predict(X_eco_test)))
print("Business")
print("RMSE Test: ", root_mean_squared_error(y_bus_test, xgb_bus.predict(X_bus_test)))
print("MAE Test: ", mean_absolute_error(y_bus_test, xgb_bus.predict(X_bus_test)))
print("R2 Test: ", r2_score(y_bus_test, xgb_bus.predict(X_bus_test)))

xgb_eco.fit(X_eco, y_eco)
xgb_bus.fit(X_bus, y_bus)

joblib.dump(xgb_eco, "xgb_eco.joblib")
joblib.dump(xgb_bus, "xgb_bus.joblib")