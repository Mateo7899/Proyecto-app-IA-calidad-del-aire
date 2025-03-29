import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo y el scaler
scaler = joblib.load("scaler.bin")
model = joblib.load("knn_model.bin")

# Título y subtítulo
st.title("Predicción de calidad del aire")
st.subheader("Realizado por Mateo Redondo")

# Mostrar imagen
st.image("https://www.poligonosindustrialesasturias.com/udecontrol_datos/objetos/2283.jpg", use_container_width=True)

# Capturar variables
st.sidebar.header("Ingrese los valores del AQI")
PM25 = st.sidebar.slider("PM2.5 AQI", -1.0, 1.0, 0.0)
CO = st.sidebar.slider("CO AQI", -1.0, 1.0, 0.0)
NO2 = st.sidebar.slider("NO2 AQI", -1.0, 1.0, 0.0)

# Crear DataFrame con los datos ingresados
data = pd.DataFrame([[PM25, CO, NO2]], columns=["PM2.5 AQI", "CO AQI", "NO2 AQI"])

# Escalar los datos
data_scaled = scaler.transform(data)

# Predicción
prediction = model.predict(data_scaled)[0]

# Mostrar resultados
if prediction == 0:
    st.markdown("<h2 style='color: green;'>El aire no tiene riesgo</h2>", unsafe_allow_html=True)
    st.markdown("✅")
else:
    st.markdown("<h2 style='color: pink;'>Alta probabilidad de aire nocivo</h2>", unsafe_allow_html=True)
    st.markdown("⚠️")

# Dibujar línea
st.markdown("---")

# Footer
st.markdown("© UNAB 2025")

