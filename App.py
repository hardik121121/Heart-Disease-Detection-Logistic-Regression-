import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# App Configuration
st.set_page_config(
    page_title="Heart Disease Detection ❤️",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and Description
st.title("🌟 Heart Disease Detection System ❤️")
st.markdown("""
### 💡 Welcome to the Heart Disease Detection App!
This tool uses a Logistic Regression model to predict the likelihood of having heart disease based on your health parameters.

🔍 **Features**:
- 🖥️ **Interactive UI**: Input your health data below to check your heart's health status.
- 📊 **Data Insights**: Explore the dataset and the model's accuracy.
- 🎉 **Audience Engagement**: Share your results and let others try it out!

**⚠️ Note:** This app is for educational purposes and not intended for medical diagnosis.
""")

# Sidebar - Input Parameters
st.sidebar.header("📝 Input Your Health Details")
def user_input_features():
    age = st.sidebar.slider("👵 Age", 20, 80, 40)
    sex = st.sidebar.selectbox("⚤ Sex", options=[1, 0], format_func=lambda x: "Male ♂️" if x == 1 else "Female ♀️")
    cp = st.sidebar.selectbox("💓 Chest Pain Type (CP)", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
    trestbps = st.sidebar.slider("💉 Resting Blood Pressure (Trestbps)", 80, 200, 120)
    chol = st.sidebar.slider("🍔 Cholesterol Level (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
    restecg = st.sidebar.selectbox("🫀 Resting ECG Results", options=[0, 1, 2])
    thalach = st.sidebar.slider("🏃 Max Heart Rate Achieved (Thalach)", 60, 220, 150)
    exang = st.sidebar.selectbox("🏋️ Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes ⚠️" if x == 1 else "No ✅")
    oldpeak = st.sidebar.slider("📉 ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("📐 Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.sidebar.slider("🔬 Number of Major Vessels Colored by Fluoroscopy (CA)", 0, 4, 0)
    thal = st.sidebar.selectbox("🧬 Thalassemia", options=[0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return pd.DataFrame(data, index=[0])

# Load and Process Dataset
df = pd.read_csv("heart_disease_data.csv")
X = df.drop(columns="target", axis=1)
Y = df["target"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train Logistic Regression Model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression())
])
pipeline.fit(X_train, Y_train)

# Model Accuracy
Y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Display Dataset Insights
if st.checkbox("📊 Show Dataset Insights"):
    st.subheader("📂 Heart Disease Dataset")
    st.write(df.head())
    st.write("🔢 Shape of the dataset:", df.shape)
    st.write("📈 Target Value Distribution:")
    st.bar_chart(df["target"].value_counts())
    st.metric(label="🧮 Model Accuracy", value=f"{accuracy:.2%}")

# User Input Features
input_df = user_input_features()

# Prediction
if st.button("🩺 Predict"):
    prediction = pipeline.predict(input_df)
    result = "🎉 **Your Heart is Perfectly Healthy! ❤️**" if prediction[0] == 0 else "⚠️ **You must consult a Doctor! 🚨**"
    st.subheader("🔍 Prediction Result")
    st.success(result)

    # Visualize User Input
    st.subheader("🧾 Your Input Features")
    st.write(input_df)

# Footer
st.markdown("""
---
👨‍💻 Developed with ❤️ by **Hardik Arora**. 🌟
""")
