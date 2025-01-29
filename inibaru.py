import pickle
import streamlit as st
import numpy as np

# Load model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Load scaler jika model menggunakan normalisasi
try:
    scaler = pickle.load(open('scaler.sav', 'rb'))
except:
    scaler = None  # Jika tidak ada scaler, gunakan data mentah

# Judul aplikasi
st.title('Prediksi Diabetes')

# Input nilai dari user
Pregnancies = st.number_input('Input nilai Pregnancies', min_value=0, step=1)
Glucose = st.number_input('Input Glucose', min_value=0)
BloodPressure = st.number_input('Input nilai Blood Pressure', min_value=0)
SkinThickness = st.number_input('Input Nilai Skin Thickness', min_value=0)
Insulin = st.number_input('Input nilai Insulin', min_value=0)
BMI = st.number_input('Input nilai BMI', min_value=0.1, format="%.1f")
DiabetesPedigreeFunction = st.number_input('Input nilai Diabetes Pedigree Function', min_value=0.0, format="%.3f")
Age = st.number_input('Input nilai Age', min_value=0, step=1)

# Prediksi
diab_diagnosis = ''

if st.button('Test Prediksi Diabetes'):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Normalisasi jika scaler tersedia
    if scaler:
        input_data = scaler.transform(input_data)

    # Prediksi probabilitas
    diab_prediction = diabetes_model.predict(input_data)
    proba = diabetes_model.predict_proba(input_data)

    # Output hasil prediksi
    if diab_prediction[0] == 1:
        diab_diagnosis = f'Pasien Terkena Diabetes (Probabilitas: {proba[0][1]:.2f})'
    else:
        diab_diagnosis = f'Pasien Tidak Terkena Diabetes (Probabilitas: {proba[0][0]:.2f})'

    st.success(diab_diagnosis)
