import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

##Load the encoder and model
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
##Load the trained model
with open('best_random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

##Streamlit App
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: brown'>🌳Forest Cover Type Prediction🌳</h1>", unsafe_allow_html=True)
st.write("Enter the features to predict the forest cover type.")
background_color = "#EB7B7B"
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
    }}
    </style>
    """, unsafe_allow_html=True)

##Function to get user input
def get_user_input():
    Elevation = st.number_input("Elevation", min_value=0, max_value=5000, value=2000) 
    Horizontal_Distance_To_Roadways = st.number_input("Horizontal Distance To Roadways", min_value=0, max_value=10000, value=500)
    Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal Distance To Fire Points", min_value=0, max_value=10000, value=600)
    Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance To Hydrology", min_value=0, max_value=10000, value=300)
    Wilderness_Area_1 = st.selectbox("Wilderness Area 1", [0, 1])
    Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance To Hydrology", min_value=-1000, max_value=1000, value=50)
    Wilderness_Area_4 = st.selectbox("Wilderness Area 4", [0, 1])
    Hillshade_9am = st.slider("Hillshade 9am", min_value=0, max_value=255, value=150)
    Aspect = st.slider("Aspect", min_value=0, max_value=360, value=90)
    Hillshade_3pm = st.slider("Hillshade 3pm", min_value=0, max_value=255, value=100)
    Hillshade_Noon = st.slider("Hillshade Noon", min_value=0, max_value=255, value=200)
    Slope = st.slider("Slope", min_value=0, max_value=90, value=10)
    Wilderness_Area_3 = st.selectbox("Wilderness Area 3", [0, 1])
    Soil_Type_10 = st.selectbox("Soil Type 10", [0, 1])
    Soil_Type_3 = st.selectbox("Soil Type 3", [0, 1])

##For Prediction
    if button := st.button("Predict"):
        input_data = np.array([[Elevation, Horizontal_Distance_To_Roadways, Horizontal_Distance_To_Fire_Points,
                                Horizontal_Distance_To_Hydrology, Wilderness_Area_1, Vertical_Distance_To_Hydrology,
                                Wilderness_Area_4, Hillshade_9am, Aspect, Hillshade_3pm, Hillshade_Noon,
                                Slope, Wilderness_Area_3, Soil_Type_10, Soil_Type_3]])
        prediction = model.predict(input_data)
        predicted_label = label_encoder.inverse_transform(prediction)
        st.markdown(f"<h2 style='color: green; font-weight: bold;'>🏞️ Predicted Forest Cover Type: {predicted_label[0]}</h2>", unsafe_allow_html=True)
    else:
        st.write("Click the button to make a prediction.")
    return None
##Get user input and make prediction
get_user_input()
