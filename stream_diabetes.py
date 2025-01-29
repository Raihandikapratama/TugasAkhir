import pickle
import streamlit as st 

# load model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

#judul web
st.title('Prediksi Diabetes')

Pregnancies = st.text_input('Input nilai Pregnancies')

Glucose = st.text_input('Input Glucose')

BloodPressure = st.text_input ('Input nilai Blood Pressure')

SkinThickness = st.text_input ('Input Nilai Skin Thickness')

Insulin = st.text_input ('Input nilai Insulin')

BMI = st.text_input ('Input nilai BMI')

DiabetesPedigreeFunction = st.text_input ('Input nilai Diabetes Pedigree Function ')

Age = st.text_input ('Input nilai age')

# prediksi
diab_diagnosis = ''

# button prediksi
if st.button('test Prediksi Diabetes'):
    diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    
    if(diab_prediction[0]== 1):
        diab_diagnosis = 'Pasien Terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien Tidak Terkena Diabetes'
    
    st.success(diab_diagnosis)