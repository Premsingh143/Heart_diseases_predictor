import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sqlite3
from datetime import datetime
import io
import plotly.express as px
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import red, green

# Optional: SHAP for model explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not installed. Install it with `pip install shap` for feature explanations.")

# Page configuration
st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide", initial_sidebar_state="expanded")

# ------------------- Custom CSS for Modern UI -------------------
st.markdown("""
<style>
    /* Main container padding */
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    /* Slider track */
    .stSlider>div>div>div {
        background-color: #4CAF50;
    }
    /* Selectbox */
    .stSelectbox>div>div>div {
        border-radius: 8px;
    }
    /* Radio buttons styled as pills */
    .stRadio>div>div {
        gap: 1rem;
    }
    .stRadio label {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
    }
    .stRadio label:hover {
        background: #e0e2e6;
    }
    .stRadio [data-baseweb="radio"] {
        display: none;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.2rem;
        font-weight: bold;
    }
    /* Table styling */
    .dataframe {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Database Setup -------------------
conn = sqlite3.connect("patients.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    chest_pain TEXT,
    exercise_angina TEXT,
    max_hr INTEGER,
    oldpeak REAL,
    st_slope TEXT,
    fasting_bs TEXT,
    result TEXT,
    date TEXT
)
""")
conn.commit()

# ------------------- Helper Functions -------------------
def get_next_patient_id():
    row = cursor.execute("SELECT patient_id FROM patients ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return "PAT-0001"
    last_id = int(row[0].split("-")[1])
    return f"PAT-{last_id+1:04d}"

@st.cache_resource
def load_model():
    try:
        with open("heart_disease_model.pkl", "rb") as f:
            obj = pickle.load(f)
        return obj["model"], obj.get("scaler"), obj.get("features")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None

def predict_risk(age, sex, chest_pain, blood_sugar, max_hr, exercise_angina, oldpeak, st_slope):
    user_input = {
        'age': age,
        'sex': sex,
        'chest pain type': chest_pain,
        'fasting blood sugar': blood_sugar,
        'max heart rate': max_hr,
        'exercise angina': exercise_angina,
        'oldpeak': oldpeak,
        'ST slope': st_slope
    }
    ordered = [user_input[f] for f in feature_order]
    features = np.array([ordered]).astype(float)
    if scaler is not None:
        features = scaler.transform(features)
    pred = model.predict(features)[0]
    try:
        prob = model.predict_proba(features)[0]
    except:
        prob = None
    return pred, prob

def create_pdf(data, result, recommendations, prob):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = 760

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(300, y, "HEART DISEASE MEDICAL REPORT")
    y -= 40
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Report Generated: {datetime.now().strftime('%d %b %Y')}")
    y -= 30

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Patient Details")
    y -= 20
    c.setFont("Helvetica", 12)
    for k in ["Patient ID", "Name", "Age", "Gender"]:
        c.drawString(60, y, f"{k}: {data[k]}")
        y -= 18
    y -= 10

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Clinical Measurements")
    y -= 20
    c.setFont("Helvetica", 12)
    inputs = ["Chest Pain Type", "Exercise Angina", "Max Heart Rate", "ST Depression", "ST Slope", "Fasting Blood Sugar"]
    for i in inputs:
        c.drawString(60, y, f"{i}: {data[i]}")
        y -= 18
    y -= 20

    c.setFont("Helvetica-Bold", 16)
    if result == "High Risk":
        c.setFillColor(red)
    else:
        c.setFillColor(green)
    c.drawString(40, y, f"Prediction Result: {result}")
    c.setFillColor("black")
    y -= 20

    if prob is not None:
        c.setFont("Helvetica", 12)
        c.drawString(60, y, f"Disease Probability: {round(prob[1] * 100, 2)} %")
        y -= 18
        c.drawString(60, y, f"No Disease Probability: {round(prob[0] * 100, 2)} %")
        y -= 25

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Recommendations")
    y -= 20
    c.setFont("Helvetica", 11)
    for r in recommendations:
        c.drawString(60, y, f"• {r}")
        y -= 16
    y -= 30

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, y, "Disclaimer: This AI prediction is for educational purposes only.")
    c.drawString(40, y - 12, "Consult a qualified cardiologist for medical advice.")
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def show_risk_gauge(probability):
    risk_level = "Low" if probability < 0.3 else "Medium" if probability < 0.7 else "High"
    color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
    st.markdown(f"""
    <div style="background-color: #f0f2f6; border-radius: 10px; padding: 10px; margin: 10px 0;">
        <div style="background-color: {color}; width: {probability*100}%; height: 20px; border-radius: 10px;"></div>
        <p style="margin: 5px 0 0 0; font-weight: bold;">Risk Level: {risk_level} ({probability*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- Load Model -------------------
model, scaler, feature_order = load_model()
if model is None:
    st.error("Model could not be loaded. Please check your pickle file.")
    st.stop()

# ------------------- Sidebar Menu -------------------
menu = st.sidebar.selectbox("Menu", ["Prediction", "Admin Dashboard"], key="menu_select")

# ------------------- Prediction Page -------------------
if menu == "Prediction":
    st.title(" Heart Disease Risk Assessment")

    patient_id = get_next_patient_id()

    #Patient Details 
    with st.expander(" Patient Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", placeholder="Enter patient's full name", key="patient_name")
        with col2:
            st.text_input("Patient ID", value=patient_id, disabled=True, key="patient_id_display")

    # Medical Measurements
    with st.expander("Clinical Measurements", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=40, step=1,
                                   help="Age in years", key="age_input")
            sex = st.radio("Gender", ["Male", "Female"], horizontal=True,
                            help="Select patient's gender", key="gender_radio")
            chest = st.selectbox("Chest Pain Type",
                                 ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                                 help="Typical Angina: substernal chest pain provoked by exertion",
                                 key="chest_type")
            exercise_angina = st.radio("Exercise Induced Angina", ["No", "Yes"], horizontal=True,
                                        help="Angina induced by exercise", key="exercise_angina_radio")
        with col2:
            max_hr = st.slider("Maximum Heart Rate", 60, 220, 150,
                               help="Maximum heart rate achieved during exercise", key="max_hr_slider")
            oldpeak = st.slider("ST Depression (mm)", -2.0, 6.0, 1.0, step=0.1,
                                help="ST depression induced by exercise relative to rest", key="st_depression_slider")
            slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"],
                                 help="Slope of the peak exercise ST segment", key="st_slope_select")
            fasting = st.radio("Fasting Blood Sugar", ["Normal", "High"], horizontal=True,
                               help="Fasting blood sugar > 120 mg/dl", key="fasting_sugar_radio")

    # Prediction
    if st.button("🩺 Assess Heart Disease Risk", type="primary", key="predict_button"):
        with st.spinner("Analyzing data..."):
            # Convert inputs
            sex_val = 1 if sex == "Male" else 0
            chest_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest)
            angina_val = 1 if exercise_angina == "Yes" else 0
            slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
            fasting_val = 0 if fasting == "Normal" else 1

            pred, prob = predict_risk(
                age, sex_val, chest_val, fasting_val,
                max_hr, angina_val, oldpeak, slope_val
            )

            result = "High Risk" if pred == 1 else "Low Risk"
            prob_risk = prob[1] if prob is not None else None

        # Display result
        if result == "High Risk":
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

        if prob_risk is not None:
            show_risk_gauge(prob_risk)

        # Smart recommendations
        recommendations = []
        if age > 50:
            recommendations.append("Consider regular cardiovascular screening due to age.")
        if fasting_val == 1:
            recommendations.append("Manage blood sugar levels through diet and exercise.")
        if max_hr < 100:
            recommendations.append("Low maximum heart rate may indicate reduced cardiovascular fitness.")
        if oldpeak > 2.0:
            recommendations.append("ST depression suggests possible ischemia; follow up with a stress test.")
        if result == "High Risk":
            recommendations.append("Consult a cardiologist within the next 2 weeks.")
        if not recommendations:
            recommendations.append("Maintain a healthy lifestyle and regular checkups.")

        with st.expander("Recommendations", expanded=True):
            for rec in recommendations:
                st.write(f"• {rec}")

        # Save to database
        cursor.execute("""
            INSERT INTO patients
            (patient_id, name, age, gender, chest_pain, exercise_angina, max_hr, oldpeak, st_slope, fasting_bs, result, date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            patient_id, name, age, sex,
            chest, exercise_angina, max_hr, oldpeak,
            slope, fasting, result, str(datetime.now())
        ))
        conn.commit()
        st.toast("Patient record saved successfully")

        # PDF report data
        report_data = {
            "Patient ID": patient_id,
            "Name": name,
            "Age": age,
            "Gender": sex,
            "Chest Pain Type": chest,
            "Exercise Angina": exercise_angina,
            "Max Heart Rate": max_hr,
            "ST Depression": oldpeak,
            "ST Slope": slope,
            "Fasting Blood Sugar": fasting
        }
        pdf_bytes = create_pdf(report_data, result, recommendations, prob)
        st.download_button(
            " Download Medical Report",
            data=pdf_bytes,
            file_name=f"{patient_id}_report.pdf",
            mime="application/pdf",
            key="download_pdf"
        )

        # Optional SHAP explanation
        if SHAP_AVAILABLE and st.checkbox("Explain this prediction (SHAP)", key="shap_checkbox"):
            with st.spinner("Computing SHAP values..."):
                # Create a background dataset (use a sample of training data if available)
                # Here we'll use the same input as background (not ideal, but for demo)
                # In production, load a background dataset from the training set.
                st.warning("SHAP explanation requires a background dataset. Using current input only for demonstration.")
                explainer = shap.TreeExplainer(model)
                # Ensure features is defined (it is inside the with block, but we need it outside)
                # We'll recreate features here for clarity
                user_input = {
                    'age': age,
                    'sex': sex_val,
                    'chest pain type': chest_val,
                    'fasting blood sugar': fasting_val,
                    'max heart rate': max_hr,
                    'exercise angina': angina_val,
                    'oldpeak': oldpeak,
                    'ST slope': slope_val
                }
                ordered = [user_input[f] for f in feature_order]
                features_arr = np.array([ordered]).astype(float)
                if scaler is not None:
                    features_arr = scaler.transform(features_arr)
                shap_values = explainer.shap_values(features_arr)
                # For binary classification, shap_values is a list of two arrays. Use the positive class.
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    shap_vals = shap_values[0]
                # Create DataFrame for display
                shap_df = pd.DataFrame({'Feature': feature_order, 'SHAP Value': shap_vals})
                fig, ax = plt.subplots()
                shap.summary_plot(shap_vals, features_arr, feature_names=feature_order, show=False)
                st.pyplot(fig)

# ---- Admin Dashboard -------
elif menu == "Admin Dashboard":
    st.title(" Admin Dashboard")

    # Load all records
    data = cursor.execute("SELECT * FROM patients").fetchall()
    columns = [
        "DB_ID", "Patient ID", "Name", "Age", "Gender",
        "Chest Pain", "Exercise Angina", "Max HR",
        "ST Depression", "ST Slope", "Fasting BS",
        "Result", "Date"
    ]
    df = pd.DataFrame(data, columns=columns)

    st.metric("Total Patients", len(df))

    # Search/filter
    search = st.text_input(" Search by Name or Patient ID", placeholder="Enter name or ID", key="admin_search")
    if search:
        filtered = df[df["Name"].str.contains(search, case=False) | df["Patient ID"].str.contains(search, case=False)]
    else:
        filtered = df

    # Display table
    st.dataframe(filtered.drop(columns=["DB_ID"]), use_container_width=True)

    # Export CSV
    if st.button(" Export All Data as CSV", key="export_csv"):
        csv = df.drop(columns=["DB_ID"]).to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="heart_disease_patients.csv", mime="text/csv", key="download_csv")

    # Visualization
    if len(df) > 0:
        st.subheader("Risk Distribution")
        fig = px.histogram(df, x="Result", title="Risk Distribution", color="Result", barmode="group")
        st.plotly_chart(fig)

        st.subheader("Risk Over Time")
        df_date = df.copy()
        df_date["Date"] = pd.to_datetime(df_date["Date"])
        df_date = df_date.sort_values("Date")
        fig_line = px.line(df_date, x="Date", y="Age", color="Result", title="Age Distribution Over Time")
        st.plotly_chart(fig_line)