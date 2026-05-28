import streamlit as st
import requests

# 🌟 මෙතනට ඔයාගේ Ngrok ලින්ක් එක දෙන්න
BACKEND_URL = st.sidebar.text_input("Backend URL", value="https://seventh-boney-subject.ngrok-free.dev")

st.title("🚀 Loan Approval MLOps Dashboard")
st.write("FastAPI Real-Time Prediction App ")
 
st.sidebar.header("User Input Features")
age = st.sidebar.slider("Age", 18, 75, 35)
income = st.sidebar.number_input("Annual Income ($)", value=50000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 700)
loan_amount = st.sidebar.number_input("Loan Amount ($)", value=15000)
emp_years = st.sidebar.slider("Employment Years", 0, 40, 5)
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 1.0, 0.3, step=0.01)

if st.button("Predict Loan Status"):
    payload = {
        "age": age,
        "annual_income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": emp_years,
        "debt_ratio": debt_ratio
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload)
        if response.status_code == 200:
            res_data = response.json()
            st.success(f"Result: {res_data['decision']}")
            st.metric("Confidence", f"{res_data['confidence'] * 100:.2f}%")
            st.json(res_data)
        else:
            st.error(f"Backend Error: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to FastAPI Backend. Check Ngrok URL! Error: {e}")