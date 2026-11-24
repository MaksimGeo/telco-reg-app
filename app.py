import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Cargar el modelo y el DictVectorizer
with open('churn-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Churn de Clientes")

# Formulario para introducir datos del cliente
st.header("Introduce los datos del cliente:")

contract = st.selectbox("Tipo de contrato", ["month-to-month", "one_year", "two_year"])
dependents = st.selectbox("Dependientes", ["no", "yes"])
device_protection = st.selectbox("Protección de dispositivos", ["no", "yes", "no_internet_service"])
internet_service = st.selectbox("Tipo de Internet", ["dsl", "fiber_optic", "no"])
online_backup = st.selectbox("Copia de seguridad online", ["no", "yes", "no_internet_service"])
online_security = st.selectbox("Seguridad online", ["no", "yes", "no_internet_service"])
paperless_billing = st.selectbox("Facturación sin papel", ["no", "yes"])
payment_method = st.selectbox("Método de pago", ["bank_transfer_(automatic)", "credit_card_(automatic)", "electronic_check", "mailed_check"])
streaming_movies = st.selectbox("Películas en streaming", ["no", "yes", "no_internet_service"])
streaming_tv = st.selectbox("Televisión en streaming", ["no", "yes", "no_internet_service"])
tech_support = st.selectbox("Soporte técnico", ["no", "yes", "no_internet_service"])

# Dejamos solo los 11 más importantes 

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario SOLO con los 11 datos seleccionados
    client_data = {
        "contract": contract,
        "dependents": dependents,
        "deviceprotection": device_protection,
        "internetservice": internet_service,
        "onlinebackup": online_backup,
        "onlinesecurity": online_security,
        "paperlessbilling": paperless_billing,
        "paymentmethod": payment_method,
        "streamingmovies": streaming_movies,
        "streamingtv": streaming_tv,
        "techsupport": tech_support
    }

    # Transformar los datos del cliente
    X_client = dv.transform([client_data])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_client)[0][1]  # Probabilidad de churn

    # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}")