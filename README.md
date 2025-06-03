# 🩺 AI for Diabetes Risk Prediction

An interactive healthcare AI app that predicts the likelihood of diabetes based on patient health indicators. Built with **Streamlit**, **scikit-learn**, and **SHAP** for explainable AI.

## 🚀 Live App

👉 [Click to Try the App](https://ai-diabetes-app-app-kxqzpxpayw85pla6sqrfhe.streamlit.app/)

## 📊 Features

- Real-time diabetes risk prediction
- SHAP visualizations to explain predictions
- Clean, responsive Streamlit UI
- Deployed on Streamlit Cloud

## 🧠 Model Info

- **Algorithm**: Logistic Regression
- **Features Used**: HighBP, HighChol, BMI, GenHlth, Age
- **Explainability**: SHAP waterfall plots (individual impact)

## 📁 Project Structure

ai-diabetes-streamlit-app/
├── app.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md