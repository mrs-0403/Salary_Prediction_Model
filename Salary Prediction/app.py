import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load model and scaler
# -----------------------------

model = joblib.load("salary_model.pkl")
scaler = joblib.load("salary_scaler.pkl")


# -----------------------------
# Page title
# -----------------------------

st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰"
)


st.title("💰 Salary Prediction App")

st.write("Enter the employee details below to predict the salary.")


# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

experience = st.number_input(
    "Experience_year",
    min_value=0,
    max_value=50,
    value=2
)

education = st.selectbox(
    "Education_Level",
    [
        "High School",
        "Bachelor",
        "Master",
        "PhD"
    ]
)


# -----------------------------
# Predict button
# -----------------------------

if st.button("Predict Salary"):

    # Scale Age and Experience
    scaled_values = scaler.transform(
        [[age, experience]]
    )

    age_scaled = scaled_values[0][0]
    experience_scaled = scaled_values[0][1]


    # Education dummy variables
    highschool = 0
    master = 0
    phd = 0

    if education == "High School":
        highschool = 1

    elif education == "Master":
        master = 1

    elif education == "PhD":
        phd = 1


    # Create input dataframe
    input_data = pd.DataFrame(
        [[
            age_scaled,
            experience_scaled,
            highschool,
            master,
            phd
        ]],
        columns=[
            "Age",
            "Experience_year",
            "Education_level_HighSchool",
            "Education_level_Master",
            "Education_level_PhD"
        ]
    )


    # Make prediction
    prediction = model.predict(input_data)


    # Display result
    st.success(
        f"Predicted Salary: ₹{prediction[0]:,.2f}"
    )