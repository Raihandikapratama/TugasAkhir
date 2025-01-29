import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model terbaik yang telah disimpan
model_filename = 'best_diabetes_model.sav'
try:
    model = pickle.load(open(model_filename, 'rb'))
except FileNotFoundError:
    st.error("Model tidak ditemukan! Pastikan file 'best_diabetes_model.sav' ada di folder yang sama.")
    st.stop()

# Judul Aplikasi
st.title("Prediksi Diabetes Sederhana")
st.write("Masukkan data di bawah untuk mengetahui apakah pasien berisiko terkena diabetes atau tidak.")

# Input Data dari User
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input("Glukosa", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Ketebalan Kulit", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Usia", min_value=1, max_value=120, value=30)

# Tombol Prediksi
if st.button("Prediksi"):
    # Konversi input ke numpy array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Prediksi dengan model
    prediction = model.predict(input_data)
    
    # Hasil Prediksi
    if prediction[0] == 1:
        st.error("Pasien berisiko terkena diabetes!")
    else:
        st.success("Pasien tidak berisiko terkena diabetes")