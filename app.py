import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")
#Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('saved_model/model.pkl')

model = load_model()


st.title("House Price Predictor")
st.markdown("use the sidebar to input house features and predict its **median price**.")

#input from user
st.sidebar.header("Input House Details")

longitude = st.number_input("Longitude",value=-122.0)
latitude = st.number_input("Latitude",value=37.0)
housing_median_age = st.slider("Housing Median Age",1, 100, 30)
total_rooms = st.number_input("Toatal Rooms",value=2000)
total_bedrooms= st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income(in $10,000s)", value=3.0)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN","INLAND", "ISLAND", "NEWAR BAY", "NEAR OCEAN"])

#Prepare the input as dataframe
input_data = pd.DataFrame({
    "longitude":[longitude],
    "latitude":[latitude],
    "housing_median_age":[housing_median_age],
    "total_rooms":[total_rooms],
    "total_bedrooms":[total_bedrooms],
    "population":[population],
    "households":[households],
    "median_income":[median_income],
    "ocean_proximity":[ocean_proximity]
})

#One-hot encode 'ocean_proximity
input_data = pd.get_dummies(input_data)

#Align with taraining featueres
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_features]  # reordering columns

#predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"EStimated Median House Value:**${int(prediction):,}**")
