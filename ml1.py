import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load ML models
diabetes_model = pickle.load(open("C:/Users/HP/Desktop/disease detection/diabetes_model.sav", "rb"))
heart_model = pickle.load(open("C:/Users/HP/Desktop/disease detection/heart_model.sav", "rb"))
parkinson_model = pickle.load(open("C:/Users/HP/Desktop/disease detection/parksinson_model.sav", "rb"))

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "AI-Powered Health Diagnosis",
        ["Diabetes Analysis", "Heart Health Check", "Parkinson's Assessment"],
        menu_icon="stethoscope",
        icons=["activity", "heart-pulse", "person"],
        default_index=0,
    )

st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Diabetes Prediction Page
if selected == "Diabetes Analysis":
    st.title("🔬 Predict Your Diabetes Risk")
    st.write("Fill in the details to check your likelihood of having diabetes.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, format="%.2f")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    
    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0.0, format="%.2f")
        Insulin = st.number_input("Insulin Level", min_value=0.0, format="%.2f")
        Age = st.number_input("Age", min_value=1, step=1)
    
    with col3:
        BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, format="%.2f")
        BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f")
    
    if st.button("🔍 Get Diabetes Risk Assessment"):
        result = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        st.success("Diabetic" if result[0] == 1 else "Not Diabetic")

# Heart Disease Prediction Page
elif selected == "Heart Health Check":
    st.title("❤️ Heart Disease Risk Analysis")
    st.write("Assess your heart condition with AI-powered insights.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, step=1)
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=0.0, format="%.2f")
        restecg = st.number_input("Resting Electrocardiographic Results", min_value=0, step=1)
    
    with col2:
        sex = st.radio("Sex", ("Male", "Female"))
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0.0, format="%.2f")
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, step=1)
    
    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
        exang = st.radio("Exercise Induced Angina", ("Yes", "No"))
    
    if st.button("🔍 Check Heart Health"):
        input_data = [
            age, 1 if sex == "Male" else 0, cp, trestbps, chol,
            1 if fbs == "Yes" else 0, restecg, thalach, 1 if exang == "Yes" else 0
        ]
        prediction = heart_model.predict([input_data])
        st.success("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected")

# Parkinson's Prediction Page
elif selected == "Parkinson's Assessment":
    st.title("🧠 Parkinson's Disease Risk Evaluation")
    st.write("Analyze vocal biomarkers to detect Parkinson's disease early.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.3f")
        Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.3f")
        RAP = st.number_input("MDVP:RAP", min_value=0.0, format="%.3f")
        APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, format="%.3f")
    
    with col2:
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, format="%.3f")
        Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, format="%.3f")
        PPQ = st.number_input("MDVP:PPQ", min_value=0.0, format="%.3f")
        APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, format="%.3f")
    
    with col3:
        flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, format="%.3f")
        DDP = st.number_input("Jitter:DDP", min_value=0.0, format="%.3f")
        Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, format="%.3f")
        NHR = st.number_input("NHR", min_value=0.0, format="%.3f")
    
    if st.button("🔍 Assess Parkinson's Risk"):
        user_input = [
            fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,
            DDP, Shimmer, APQ3, APQ5, NHR
        ]
        prediction = parkinson_model.predict([user_input])
        st.success("Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's Detected")
