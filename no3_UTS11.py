#!/usr/bin/env python
# coding: utf-8

# In[19]:


import streamlit as st
import joblib
import numpy as np

# Harus paling atas!
st.set_page_config(page_title="Loan Status Prediction", layout="centered")

# Informasi Mahasiswa
st.markdown("**Nama: Naira Faizanoor**  \n**NIM: 2702241465**")

@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('xgboost_model.pkl')
        label_encoder_gender = joblib.load('label_encoder_gender.pkl')
        label_encoder_defaults = joblib.load('label_encoder_defaults.pkl')
        label_encoder_home = joblib.load('label_encoder_home.pkl')
        label_encoder_loan_intent = joblib.load('label_encoder_loan_intent.pkl')
        ordinal_encoder_education = joblib.load('ordinal_encoder_education.pkl')
        return model, label_encoder_gender, label_encoder_defaults, label_encoder_home, label_encoder_loan_intent, ordinal_encoder_education
    except Exception as e:
        st.error(f"❌ Gagal memuat model/encoder: {e}")
        return None, None, None, None, None, None

def main():
    st.title("📊 Loan Booking Status Prediction")

    model, le_gender, le_defaults, le_home, le_intent, ord_edu = load_model_and_encoders()
    if model is None:
        return

    if "inputs" not in st.session_state:
        st.session_state.inputs = {
            "age": 30,
            "gender": "Male",
            "education": "Bachelor",
            "income": 50000,
            "exp": 5,
            "home": "RENT",
            "amount": 10000,
            "intent": "PERSONAL",
            "rate": 12.0,
            "percent_income": 0.2,
            "history": 5,
            "credit": 650,
            "defaults": "No"
        }

    st.sidebar.markdown("### 🧪 Jalankan Test Case")
    
    if st.sidebar.button("Test Case 1"):
        st.session_state.inputs = {
            "age": 22,
            "gender": "Female",
            "education": "High School",
            "income": 25000,
            "exp": 1,
            "home": "RENT",
            "amount": 30000,
            "intent": "EDUCATION",
            "rate": 23.5,
            "percent_income": 1.0,
            "history": 3,
            "credit": 522,
            "defaults": "Yes"
        }

    if st.sidebar.button("Test Case 2"):
        st.session_state.inputs = {
            "age": 22,
            "gender": "Female",
            "education": "High School",
            "income": 25000,
            "exp": 1,
            "home": "RENT",
            "amount": 30000,
            "intent": "VENTURE",
            "rate": 23.5,
            "percent_income": 0.9,
            "history": 2,
            "credit": 510,
            "defaults": "Yes"
        }

    if st.sidebar.button("Lihat Variabel Test Case 1"):
        st.write("Variabel untuk Test Case 1:")
        st.write("Age: 22")
        st.write("Gender: Female")
        st.write("Education: High School")
        st.write("Income: 25000")
        st.write("Experience: 1 year")
        st.write("Home Ownership: RENT")
        st.write("Loan Amount: 30000")
        st.write("Loan Intent: EDUCATION")
        st.write("Interest Rate: 23.5%")
        st.write("Loan % of Income: 1.0")
        st.write("Credit History Length: 3")
        st.write("Credit Score: 522")
        st.write("Previous Loan Defaults: Yes")

    if st.sidebar.button("Lihat Variabel Test Case 2"):
        st.write("Variabel untuk Test Case 2:")
        st.write("Age: 22")
        st.write("Gender: Female")
        st.write("Education: High School")
        st.write("Income: 25000")
        st.write("Experience: 1 year")
        st.write("Home Ownership: RENT")
        st.write("Loan Amount: 30000")
        st.write("Loan Intent: VENTURE")
        st.write("Interest Rate: 23.5%")
        st.write("Loan % of Income: 0.9")
        st.write("Credit History Length: 2")
        st.write("Credit Score: 510")
        st.write("Previous Loan Defaults: Yes")

    st.header("📝 Masukkan Data Peminjam")
    age = st.number_input("Person Age", 18, 100, st.session_state.inputs["age"])
    gender = st.selectbox("Person Gender", ['Male', 'Female'], index=['Male', 'Female'].index(st.session_state.inputs["gender"]))
    education = st.selectbox("Person Education", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], index=['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'].index(st.session_state.inputs["education"]))
    income = st.number_input("Person Income", 0, 1000000, st.session_state.inputs["income"])
    exp = st.number_input("Employment Experience (years)", 0, 50, st.session_state.inputs["exp"])
    home = st.selectbox("Home Ownership", ['OWN', 'RENT', 'MORTGAGE', 'OTHER'], index=['OWN', 'RENT', 'MORTGAGE', 'OTHER'].index(st.session_state.inputs["home"]))
    amount = st.number_input("Loan Amount", 1000, 35000, st.session_state.inputs["amount"])
    intent = st.selectbox("Loan Intent", ['EDUCATION', 'DEBTCONSOLIDATION', 'VENTURE', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT'], index=['EDUCATION', 'DEBTCONSOLIDATION', 'VENTURE', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT'].index(st.session_state.inputs["intent"]))
    rate = st.number_input("Interest Rate (%)", 0.0, 30.0, st.session_state.inputs["rate"])
    percent_income = st.number_input("Loan % of Income", 0.0, 1.0, st.session_state.inputs["percent_income"])
    history = st.number_input("Credit History Length", 0, 30, st.session_state.inputs["history"])
    credit = st.number_input("Credit Score", 400, 850, st.session_state.inputs["credit"])
    defaults = st.selectbox("Previous Loan Defaults", ['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.inputs["defaults"]))

    if st.button("🔍 Prediksi Status"):
        try:
            gender_enc = le_gender.transform([gender])[0]
            edu_enc = ord_edu.transform([[education]])[0][0]
            home_enc = le_home.transform([home])[0]
            intent_enc = le_intent.transform([intent])[0]
            default_enc = le_defaults.transform([defaults])[0]

            features = np.array([
                age, gender_enc, edu_enc, income, exp,
                home_enc, amount, intent_enc, rate,
                percent_income, history, credit, default_enc
            ]).reshape(1, -1)

            pred = model.predict(features)
            result = "Approved" if pred[0] == 1 else "Rejected"
            st.success(f"Hasil Prediksi: {result}")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




