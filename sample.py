import pickle
import streamlit as st
import numpy as np

# Load data
with open("house.pkl", "rb") as obj1:
    data = pickle.load(obj1)

# Page config
st.set_page_config(page_title="House Price Prediction", layout="centered")

# Custom CSS for luxury look + text visibility
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(-45deg, #d4fc79, #96e6a1, #84fab0, #8fd3f4);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff7eb3, #ff758c);
        color: white !important;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #ff7eb3;
        transform: scale(1.05);
    }

    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #e6f7ff;  /* Light pastel blue */
        border-radius: 8px;
        border: 1px solid #ccc;
        color: black !important;
    }
    .stSelectbox div[data-baseweb="popover"] {
        background-color: #ffffff;
        color: black !important;
    }

    /* Number input styling */
    .stNumberInput input {
        background-color: #fffbe6; /* Light pastel yellow */
        color: black !important;
        border-radius: 8px;
        border: 1px solid #ccc;
    }

    /* Labels */
    label, .stSelectbox label, .stNumberInput label {
        color: black !important;
        font-weight: bold !important;
    }

    /* Result box */
    .result-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        color: #004d00;
        animation: fadeIn 1s ease-in-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

# Hero banner
st.image("https://user-images.githubusercontent.com/26305084/117038955-35c4c980-acd6-11eb-9a5e-4e98d4d4b764.gif")

# Title
st.markdown("<h1 style='text-align:center; color:#000;'>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#222;'>Enter property details below and get an instant price estimate.</p>", unsafe_allow_html=True)

# Dictionaries
house_rating = {"Excellent": 5, "Good": 4, "Average": 3, "Bad": 2}
grade_dict = {"Below Average": 6, "Average": 7, "Good": 8, "Very Good": 9}

# Two column layout
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0)
    bathroom = st.number_input("Bathrooms", min_value=0)
    sqft_live = st.number_input("Living Area (sqft)", min_value=0)
    floors = st.number_input("Number of Floors", min_value=0)

with col2:
    condition = st.selectbox("Condition", list(house_rating.keys()))
    grade = st.selectbox("Grade", list(grade_dict.keys()))
    zipcode = st.selectbox("Zipcode", data['zipcodes'])
    age = st.number_input("House Age (Years)", min_value=0)

# Convert selected values
condition_val = house_rating[condition]
grade_val = grade_dict[grade]
zipcode_val = data['onehot'].transform([[zipcode]])

# Predict button
if st.button("🔍 Predict Price"):
    a = np.array([[bedrooms, bathroom, sqft_live, floors, condition_val, grade_val, age]])
    b = np.hstack([a, zipcode_val])  # shape: (1, features)
    test = data['scaler'].transform(b)
    res = data['model'].predict(test)[0]
    st.markdown(f"<div class='result-box'>💰 Predicted Price: ₹{round(res):,}</div>", unsafe_allow_html=True)



