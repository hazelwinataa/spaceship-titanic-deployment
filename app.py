import pickle
import pandas as pd
import streamlit as st

from preprocess import transform_data


# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load trained model
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("ASG 04 MD - Hazel - Spaceship Titanic Model Deployment")
st.write("Fill in the 13 passenger features below to predict whether the passenger was Transported.")


# 13 input features
PassengerId = st.text_input("PassengerId", value="0001_01")

HomePlanet = st.selectbox(
    "HomePlanet",
    options=["Earth", "Europa", "Mars"]
)

CryoSleep = st.selectbox(
    "CryoSleep",
    options=[True, False]
)

Cabin = st.text_input("Cabin", value="B/0/P")

Destination = st.selectbox(
    "Destination",
    options=["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"]
)

Age = st.number_input("Age", min_value=0.0, value=27.0)

VIP = st.selectbox(
    "VIP",
    options=[False, True]
)

RoomService = st.number_input("RoomService", min_value=0.0, value=0.0)
FoodCourt = st.number_input("FoodCourt", min_value=0.0, value=0.0)
ShoppingMall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
Spa = st.number_input("Spa", min_value=0.0, value=0.0)
VRDeck = st.number_input("VRDeck", min_value=0.0, value=0.0)

Name = st.text_input("Name", value="Hazel Novero")


if st.button("Predict"):
    input_df = pd.DataFrame(
        {
            "PassengerId": [PassengerId],
            "HomePlanet": [HomePlanet],
            "CryoSleep": [CryoSleep],
            "Cabin": [Cabin],
            "Destination": [Destination],
            "Age": [Age],
            "VIP": [VIP],
            "RoomService": [RoomService],
            "FoodCourt": [FoodCourt],
            "ShoppingMall": [ShoppingMall],
            "Spa": [Spa],
            "VRDeck": [VRDeck],
            "Name": [Name],
        }
    )

    X_input = transform_data(input_df, preprocessor, is_train=False)
    prediction = model.predict(X_input)[0]

    if prediction == 1:
        st.success("Prediction: Transported = True")
    else:

        st.error("Prediction: Transported = False")

