import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load ML models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models using relative paths
try:
    diabetes_model = pickle.load(open(os.path.join(BASE_DIR, "diabetes_model.sav"), "rb"))
    heart_model = pickle.load(open(os.path.join(BASE_DIR, "heart_model.sav"), "rb"))
    parkinson_model = pickle.load(open(os.path.join(BASE_DIR, "parksinson_model.sav"), "rb"))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

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
    st.write("Check your likelihood of having diabetes.")
    
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
        try:
            result = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            st.success("Diabetic" if result[0] == 1 else "Not Diabetic")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Heart Disease Prediction Page
elif selected == "Heart Health Check":
    st.title("❤️ Heart Disease Risk Analysis")
    st.write("Assess your heart condition with AI-powered insights.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, step=1)
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=0.0, format="%.2f")
        restecg = st.number_input("Resting Electrocardiographic Results", min_value=0, step=1)
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, format="%.2f")
    
    with col2:
        sex = st.radio("Sex", ("Male", "Female"))
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0.0, format="%.2f")
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, step=1)
        slope = st.number_input("Slope of the Peak Exercise ST Segment", min_value=0, step=1)
    
    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
        exang = st.radio("Exercise Induced Angina", ("Yes", "No"))
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, step=1)
        thal = st.number_input("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", min_value=0, step=1)
    
    if st.button("🔍 Check Heart Health"):
        input_data = [
            age, 1 if sex == "Male" else 0, cp, trestbps, chol,
            1 if fbs == "Yes" else 0, restecg, thalach, 1 if exang == "Yes" else 0,
            oldpeak, slope, ca, thal
        ]
        try:
            prediction = heart_model.predict([input_data])
            st.success("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Parkinson's Prediction Page
elif selected == "Parkinson's Assessment":
    st.title("🧠 Parkinson's Disease Risk Evaluation")
    st.write("Analyze biomarkers to detect Parkinson's disease early.")
    
    inputs = []
    cols = st.columns(3)
    feature_names = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
                     "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
                     "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", min_value=-100.0, "spread2",
                     "D2", "PPE"]
    
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            inputs.append(st.number_input(feature, min_value=0.0, format="%.3f"))
    
    if st.button("🔍 Assess Parkinson's Risk"):
        try:
            prediction = parkinson_model.predict([inputs])
            st.success("Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's Detected")
        except Exception as e:
            st.error(f"Prediction error: {e}")
