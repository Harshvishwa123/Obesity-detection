import streamlit as st
import pandas as pd
import joblib

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Obesity AI Dashboard",
    page_icon="🧬",
    layout="wide"
)

# ==========================
# CUSTOM CSS (Premium UI)
# ==========================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-size: 22px !important;
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    color: white;
}

/* Hide Streamlit menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Title */
.title-text {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    background: -webkit-linear-gradient(#00f5d4, #00bbf9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.subtitle-text {
    font-size: 22px;
    text-align: center;
    color: #cbd5e1;
    margin-top: -10px;
}

/* Card design */
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 0px 18px rgba(0,0,0,0.5);
}

/* Result box */
.result-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    margin-top: 20px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #00f5d4, #00bbf9);
    color: black;
    font-size: 26px !important;
    font-weight: 900;
    padding: 15px 20px;
    border-radius: 16px;
    width: 100%;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(0,245,212,0.6);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    border-right: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)


# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model_path = r"obesity_outputs\kaggle\working\obesity_outputs\best_obesity_model.pkl"
    encoder_path = r"obesity_outputs\kaggle\working\obesity_outputs\label_encoder.pkl"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    return model, encoder


model, label_encoder = load_model()


# ==========================
# HEADER
# ==========================
st.markdown('<div class="title-text">🧬 Obesity AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Predict obesity category using lifestyle & body parameters (ML Powered)</div>', unsafe_allow_html=True)
st.write("")
st.write("")


# ==========================
# SIDEBAR INPUTS
# ==========================
st.sidebar.markdown("## 🧾 Patient Input Panel")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age (Years)", min_value=1, max_value=120, value=21)

height = st.sidebar.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
weight = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)

family_history = st.sidebar.selectbox("Family History with Overweight?", ["yes", "no"])
favc = st.sidebar.selectbox("FAVC (High calorie food consumption)", ["yes", "no"])
fcvc = st.sidebar.slider("FCVC (Vegetable Consumption Level)", 1, 3, 2)
ncp = st.sidebar.slider("NCP (Meals per Day)", 1, 4, 3)

caec = st.sidebar.selectbox("CAEC (Eating between meals)", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.sidebar.selectbox("SMOKE", ["yes", "no"])
ch2o = st.sidebar.slider("CH2O (Water Intake)", 1, 3, 2)
scc = st.sidebar.selectbox("SCC (Calories monitoring)", ["yes", "no"])

faf = st.sidebar.slider("FAF (Physical Activity Frequency)", 0, 3, 1)
tue = st.sidebar.slider("TUE (Tech device usage)", 0, 2, 1)

calc = st.sidebar.selectbox("CALC (Alcohol Consumption)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.sidebar.selectbox("MTRANS (Transportation mode)",
                              ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])


# ==========================
# BMI CALCULATION
# ==========================
bmi = weight / (height ** 2)

if bmi < 18.5:
    bmi_status = "Underweight"
elif bmi < 25:
    bmi_status = "Normal"
elif bmi < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"


# ==========================
# DASHBOARD METRICS
# ==========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <h2 style="color:#00f5d4;">📌 BMI Score</h2>
        <h1 style="font-size:50px;">{bmi:.2f}</h1>
        <p style="font-size:22px; color:#cbd5e1;">Status: <b>{bmi_status}</b></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h2 style="color:#00bbf9;">👤 Patient Age</h2>
        <h1 style="font-size:50px;">{age}</h1>
        <p style="font-size:22px; color:#cbd5e1;">Gender: <b>{gender}</b></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <h2 style="color:#facc15;">⚡ Lifestyle</h2>
        <p style="font-size:20px; color:#cbd5e1;">
        Meals/Day: <b>{ncp}</b><br>
        Water Intake: <b>{ch2o}</b><br>
        Physical Activity: <b>{faf}</b><br>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")


# ==========================
# PREDICTION
# ==========================
st.markdown("## 🔮 Prediction Engine")

if st.button("🚀 Predict Obesity Category"):

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }])

    prediction = model.predict(input_data)[0]
    predicted_class = label_encoder.inverse_transform([prediction])[0]

    # Color mapping for result
    color_map = {
        "Insufficient_Weight": "#22c55e",
        "Normal_Weight": "#00f5d4",
        "Overweight_Level_I": "#facc15",
        "Overweight_Level_II": "#fb923c",
        "Obesity_Type_I": "#f97316",
        "Obesity_Type_II": "#ef4444",
        "Obesity_Type_III": "#dc2626"
    }

    result_color = color_map.get(predicted_class, "#00bbf9")

    st.markdown(f"""
        <div class="result-box" style="background: {result_color}; color: black;">
        ✅ Prediction Result <br><br>
        {predicted_class}
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### 📋 Input Summary")
    st.dataframe(input_data, use_container_width=True)


# ==========================
# FOOTER
# ==========================
st.write("")
st.markdown("""
<hr style="border: 1px solid rgba(255,255,255,0.1);">

<div style="text-align:center; font-size:18px; color: #94a3b8;">
Developed with ❤️ using <b>Streamlit</b> | Obesity Detection AI System <br>
Deployment Ready (Streamlit Cloud / Render / AWS)
</div>
""", unsafe_allow_html=True)
