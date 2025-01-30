import streamlit as st
import numpy as np
import pickle

# Load model
model_filename = 'best_diabetes_model.sav'
scaler_filename = 'scaler.sav'

try:
    model = pickle.load(open(model_filename, 'rb'))
    scaler = pickle.load(open(scaler_filename, 'rb'))
except Exception:
    st.error("❌ Gagal memuat model atau scaler. Pastikan file tersedia dan tidak rusak.")
    st.stop()

# Judul
st.title("Prediksi Diabetes")
st.write("Masukkan data untuk mengetahui apakah berisiko terkena diabetes atau tidak.")

# Input Data dari User (default kosong, BMI & DPF dengan contoh format)
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=None, step=1)
glucose = st.number_input("Glukosa", min_value=0, max_value=200, value=None, step=1)
blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=150, value=None, step=1)
skin_thickness = st.number_input("Ketebalan Kulit", min_value=0, max_value=100, value=None, step=1)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=None, step=1)
bmi = st.number_input("BMI (Contoh: 23.2)", min_value=0.0, max_value=100.0, value=None, step=0.1, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function (Contoh: 0.487)", min_value=0.0, max_value=3.0, value=None, step=0.001, format="%.3f")
age = st.number_input("Usia", min_value=1, max_value=120, value=None, step=1)

# Tombol Prediksi
if st.button("Prediksi"):
    # Tangani input kosong
    if None in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]:
        st.error("⚠️ Harap isi semua kolom sebelum melakukan prediksi!")
    else:
        # Format input ke numpy array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

        # Standardisasi data sebelum prediksi
        input_data_scaled = scaler.transform(input_data)

        # Prediksi dengan model yang sudah distandardisasi
        prediction = model.predict(input_data_scaled)

        # Menampilkan hasil akhir
        if prediction[0] == 1:
            st.error("⚠️ Resiko Terkena Diabetes!")
        else:
            st.success("✅ Tidak Beresiko Terkena Diabetes")
