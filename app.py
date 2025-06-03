import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load model, scaler, and column names
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
X_columns = pickle.load(open("columns.pkl", "rb"))

# Title
st.title("🩺 AI for Diabetes Risk Prediction")
st.markdown("Enter patient details to predict diabetes risk.")

# Input form
with st.form("prediction_form"):
    highbp = st.selectbox("High Blood Pressure?", [0, 1])
    highchol = st.selectbox("High Cholesterol?", [0, 1])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    genhlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3)
    age = st.slider("Age Group (1 to 13)", 1, 13, 5)

    # Submit button
    submit = st.form_submit_button("Predict")

# On submission
if submit:
    # Step 1: Prepare input
    input_data = pd.DataFrame([[highbp, highchol, bmi, genhlth, age]], columns=X_columns)
    input_scaled = scaler.transform(input_data)

    # Step 2: Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Step 3: Show prediction
    st.subheader("🧠 Prediction Result:")
    st.success("Positive for Diabetes 🚨" if prediction == 1 else "No Diabetes ✅")
    st.info(f"Risk Probability: {round(probability * 100, 2)}%")

    # ✅ Step 4: SHAP Explanation (must be indented into this block)
    @st.cache_resource
    def get_shap_explainer():
        background = pd.DataFrame(scaler.transform([[0, 0, 25, 3, 5]]), columns=X_columns)
        return shap.Explainer(model, background)

    explainer = get_shap_explainer()
    shap_input_df = pd.DataFrame(input_scaled, columns=X_columns)
    shap_values = explainer(shap_input_df)

    st.subheader("📊 Feature Impact (SHAP Explanation):")
    import shap
    import matplotlib.pyplot as plt

# Safer rendering: smaller figure + fewer features
    shap.plots.bar(shap_values[0], max_display=5, show=False)
    st.pyplot(plt.gcf())
    plt.clf()


