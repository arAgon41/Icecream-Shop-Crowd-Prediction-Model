import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

@st.cache_resource
def LoadModel(modelPath : str):
    """
    Loads trained model from path 
    """
    basePath = os.path.dirname(__file__)
    modelPath = os.path.join(basePath, modelPath)
    return joblib.load(modelPath)

model = LoadModel("../models/CrowdLevelPredictionModel.pkl")

def DecodeWeather(code: int) -> str:
    """
    Decodes the weather code into a human readable string.
    code: int 
    Returns:
    str: human readable weather

    1. innput the weather code
    2. go through condition statements and return the weather
    """
    if code == 0:
        return "Clear"
    elif code == 1:
        return "Mostly Clear"
    elif code == 2:
        return "Partly Cloudy"
    elif code == 3:
        return "Overcast"
    elif code in [45, 48]:
        return "Foggy"
    elif code in [51, 53, 55]:
        return "Drizzle"
    elif code in [61, 63, 65]:
        return "Rain"
    elif code in [71, 73, 75]:
        return "Snow"
    elif code in [80, 81, 82]:
        return "Rain Showers"
    elif code in [95, 96, 99]:
        return "Thunderstorm"
    return "Unknown"


def FetchWeather(selectedDate):
    """
    1. retrieving future weather data from openmeteo
    2. creating a human readable weather column
    3. creating ML features

    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=40.4406&longitude=-79.9959&hourly=temperature_2m,weathercode"
        f"&start_date={selectedDate}&end_date={selectedDate}"
    )

    raw = requests.get(url).json()
    hourly = raw["hourly"]

    df = pd.DataFrame({
        "time": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "weathercode": hourly["weathercode"]
    })

    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["month"] = selectedDate.month
    df["date"] = selectedDate

    # Human readable weather
    df["weather"] = df["weathercode"].apply(DecodeWeather)

    # weekend flag
    dayName = selectedDate.strftime("%A")
    df["weekend"] = int(dayName in ["Saturday", "Sunday"])

    # sunny flag
    df["sunny"] = (df["weathercode"] == 0).astype(int)

    return df


def BuildModelFeatures(df):
    """
    Creating seperate columns for each weather type. (One Hot Encoding)
    Returns:
    dataframe: df that is machine read
    """
    dfML = pd.get_dummies(df, columns=["weather"], drop_first=False)

    expectedCols = [
        "temperature",
        "month",
        "hour",
        "weekend",
        "sunny",
        "weather_Clear",
        "weather_Drizzle",
        "weather_Mostly Clear",
        "weather_Overcast",
        "weather_Partly Cloudy",
        "weather_Rain",
        "weather_Snow"
    ]

    for col in expectedCols:
        if col not in dfML.columns:
            dfML[col] = 0

    return dfML[expectedCols]


def BuildUserTable(df : pd.DataFrame, preds):
    """
    input: dataframe with all data used
    preds: predictions from the model

    returns: dataframe with all the data combined
    """
    return pd.DataFrame({
        "Date": df["date"],
        "Temperature": df["temperature"],
        "Hour": df["hour"],
        "Weather": df["weather"],
        "Prediction": preds
    })


def PlotPredictionChart(resultDF: pd.DataFrame): #returns a plot 
    """
    Input: 
    resultDF: dataFrame with all data seen by user
    Output:
    plot: chart of predictions hour vs. prediction

    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(resultDF["Hour"], resultDF["Prediction"], color="red", linewidth=2)
    ax.set_xlabel("Hour", fontsize=12)
    ax.set_ylabel("Crowd Level", fontsize=12)
    ax.set_title("Crowd Prediction", fontsize=14)
    ax.set_ylim(0, 1) 
    ax.grid(True, alpha=0.5)
    return fig


SUMMER_MONTHS = {6, 7, 8}

def get_valid_hours(month : int) -> list:
    """
    Input:Month:int
    Returns: a list of times that the shop will be open.

    Summer months: June, July, August : 10am to 9pm
    All other months: 11am to 5pm
    """
    if month in SUMMER_MONTHS:
        return list(range(10, 22))
    return list(range(11, 18))

# UI
"""
Stream lit UI created
features a chart and table with all data needed to make predictions
"""
st.set_page_config(layout="wide")

st.title("🍦 Ice Cream Shop Crowd Predictor")

today = datetime.today().date()
maxDay = today + timedelta(days=10)
# Data selection
center = st.columns([3, 2, 3])
with center[1]:
    selectedDate = st.date_input("Select a Date", min_value=today, max_value=maxDay, value=today)
# Button actions
if st.button("Predict Crowd Level"):

    rawDF= FetchWeather(selectedDate)

    month = selectedDate.month
    validHours = get_valid_hours(month)
    rawDF = rawDF[rawDF["hour"].isin(validHours)]

    if rawDF.empty:
        st.error("No valid prediction hours for this date.")
        st.stop()

    modelDF = BuildModelFeatures(rawDF)
    preds = model.predict(modelDF)
    userDF = BuildUserTable(rawDF, preds)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Crowd Table")
        st.dataframe(userDF, use_container_width=True)

    with col2:
        st.subheader("Crowd Chart")
        fig = PlotPredictionChart(userDF)
        st.pyplot(fig)

    st.caption("⚠️ Prediction are based on weather forecasts and will change as the weather changes! Please check back for updates!!!")