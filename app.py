import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.metrics import r2_score

# Load trained models
with open("temperature_prediction_model.pkl", "rb") as file:
    temp_model = pickle.load(file)

with open("co2_prediction_model.pkl", "rb") as file:
    co2_model = pickle.load(file)

# Streamlit App Title
st.title("🌍 Climate Change Prediction: Temperature & CO₂ Emissions")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Visualize Data"])

# Home Page
if page == "Home":
    st.write("""
    ## Climate Change Prediction Web App
    This application predicts **global temperature rise** and **CO₂ emissions** using Machine Learning models.
    
    ### Features:
    - Upload a dataset or enter values manually
    - Predict **future temperature** and **CO₂ emissions**
    - Visualize **trends and model performance**
    
    *Developed using **Streamlit**, **SVR**, and **Random Forest Regressor**.*
    """)

# Prediction Page
elif page == "Predict":
    st.subheader("📊 Predict Temperature & CO₂ Emissions")

    # User input
    st.write("### Enter Features for Prediction")

    # Take both inputs together
    co2_input = st.number_input("Enter CO₂ Emissions (normalized)", min_value=0.0, max_value=1.0, step=0.01)
    temp_input = st.number_input("Enter Land Average Temperature (normalized)", min_value=0.0, max_value=1.0, step=0.01)

    # Combine both features correctly
    input_features = np.array([[co2_input, temp_input]])  # Corrected to match model input shape

    # Prediction logic
    if st.button("Predict"):
        co2_poly_features = poly.fit_transform(np.array([[co2_input]]))  # Transform input correctly
        temp_pred = temp_model.predict(co2_poly_features)[0]  # Predict temperature  # Fixed input shape
        co2_pred = co2_model.predict(np.array([[temp_input]]))[0]  # Predict CO₂ emissions


        st.success(f"🌡️ Predicted Temperature: {temp_pred:.4f} (normalized)")
        st.success(f"💨 Predicted CO₂ Emissions: {co2_pred:.4f} (normalized)")

# Visualization Page
elif page == "Visualize Data":
    st.subheader("📈 Model Performance & Data Insights")

    # Load dataset
    st.write("### Upload Dataset for Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  
        st.success("✅ Uploaded dataset loaded successfully!")
    else:
        st.warning("⚠ No file uploaded! Using `climate_data.csv` for visualization.")
        
        # Load climate_df from saved CSV
        df = pd.read_csv("climate_data.csv")

    # Display Data Preview
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Correlation Heatmap
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Scatter Plot: Actual vs Predicted CO₂ Emissions
    st.write("### 📊 CO₂ Emissions vs Predicted Values")
    fig, ax = plt.subplots()

    
    try:
        # **Temperature Prediction (Using Polynomial CO₂ Features)**
        co2_features = df[["CO2_Emissions"]]  # Use CO₂ Emissions as input
        co2_features_poly = poly.fit_transform(co2_features)  # Apply transformation
        predicted_temp = temp_model.predict(co2_features_poly)  # Make predictions of temperature

        # **CO₂ Prediction (Using Temperature)**
        temp_features = df[["LandAverageTemperature"]]
        predicted_co2 = co2_model.predict(temp_features)  # Predict CO₂ emissions


        ax.scatter(df["CO2_Emissions"], predicted_co2, label="Predicted CO₂ Emissions", color="red", alpha=0.6)
        ax.scatter(df["CO2_Emissions"], df["CO2_Emissions"], label="Actual CO₂ Emissions", color="blue", alpha=0.5)

        plt.xlabel("Actual CO₂ Emissions")
        plt.ylabel("Predicted CO₂ Emissions")
        plt.title("CO₂ Emissions vs Predicted Values")
        plt.legend()
        plt.grid(True)

        st.pyplot(fig)
        r2_co2 = r2_score(df["CO2_Emissions"], predicted_co2)
        st.write(f"**📊 R² Score for CO₂ Emission Prediction:** {r2_co2:.4f}")

        # Scatter Plot: Actual vs Predicted Global Temperature
        st.write("### 🌡️ Global Temperature vs Predicted Values")
        fig2, ax2 = plt.subplots()

        ax2.scatter(df["LandAverageTemperature"], predicted_temp, label="Predicted Temperature", color="red", alpha=0.6)
        ax2.scatter(df["LandAverageTemperature"], df["LandAverageTemperature"], label="Actual Temperature", color="blue", alpha=0.5)

        plt.xlabel("Actual LandAverageTemperature")
        plt.ylabel("Predicted LandAverageTemperature")
        plt.title("Global Temperature vs Predicted Values")
        plt.legend()
        plt.grid(True)

        st.pyplot(fig2)
        r2_temp = r2_score(df["LandAverageTemperature"], predicted_temp)
        st.write(f"**📊 R² Score for Temperature Prediction:** {r2_temp:.4f}")

    except Exception as e:
        st.error(f"⚠ Error in generating predictions: {e}")


st.sidebar.info("Select a page to proceed ⬆️")

# Footer
st.markdown("---")
st.markdown("🔍 **Developed by Unnati Malik** | 📊 Powered by **Machine Learning & Streamlit**")