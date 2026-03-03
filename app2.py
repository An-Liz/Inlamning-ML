import streamlit as st
import joblib

st.title("✍️ MNIST – Sifferigenkänning")

# Ladda modell och scaler
model = joblib.load("mnist_model.pkl")
scaler = joblib.load("mnist_scaler.pkl")

st.success("Modell och scaler laddade!")