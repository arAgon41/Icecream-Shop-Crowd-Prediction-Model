# Riswanth Haris Sundaresh Babu
import requests
import pandas as pd
from datetime import datetime
import numpy as np

def fetch_weather_data(start_date, end_date, lat=39.0438, lon=-77.4874)-> pd.DataFrame:
    """
    Gather weather data from the OpenMeteo API for a specified time range and location based on the latitude and logitudes values.
    Inputs:
    Start_date: str
    End_date: str
    Latitude: float
    Longitude: float
    Returns:
    A data fratme with all weather data inputted.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,cloudcover,weathercode",
        "timezone": "America/New_York"
    }

    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "cloudcover": data["hourly"]["cloudcover"],
        "weathercode": data["hourly"]["weathercode"]
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"]    = df["datetime"].dt.month
    df["day"]      = df["datetime"].dt.day
    df["hour"]     = df["datetime"].dt.hour
    df["weekday"]  = df["datetime"].dt.weekday
    df["weekend"]  = (df["weekday"] >= 5).astype(int)
    df["sunny"]    = (df["weathercode"] == 0).astype(int)

    return df

def clean_hours(df):
    """
    Cleaning the data by removing rows that fall outside of the specified hours based on the month.
    Inputs:
    df: pd.DataFrame
    Returns:
    A cleaned data frame with rows outside of the specified hours removed. 

    IF its summer the closing hour is 9pm .
    IF its winter the closing hour is 5pm
    for fall/spring months we close at 7pm
    

    """
    cleaned_rows = []
    for i in range(len(df)):
        month = df["month"][i]
        hour  = df["hour"][i]

        if month in [6, 7, 8]:
            open_hour = 10
            close_hour = 21
        elif month in [12, 1, 2]:
            open_hour = 11
            close_hour = 17
        else:
            open_hour = 11
            close_hour = 19

        if hour < open_hour or hour >= close_hour:
            continue

        cleaned_rows.append(df.iloc[i])

    return pd.DataFrame(cleaned_rows)

def decode_weathercode(code):
    """
    Decoding the codes to weather condition
    Weather codes: https://open-meteo.com/en/docs
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
    else:
        return "Unknown"

# Constants for crowd level generation
# Base crowd level from 0 to 1 for each month on any given day. 
MONTH_BASE = {
    1: 0.05, 2: 0.05, 3: 0.10,
    4: 0.20, 5: 0.35, 6: 0.55,
    7: 0.65, 8: 0.55, 9: 0.35,
    10: 0.20, 11: 0.10, 12: 0.05
}

# Crowd level multiplier based on the weather 
WEATHER_MULTIPLIERS = {
    "Thunderstorm": 0.08,
    "Snow": 0.05,
    "Rain": 0.20,
    "Rain Showers": 0.25,
    "Drizzle": 0.35,
    "Foggy": 0.40,
    "Overcast": 0.60,
    "Partly Cloudy": 0.80,
    "Mostly Clear": 0.90,
    "Clear": 1.00
}


def get_temperature_modifier(temperature):
    """
    Returns a score modifier based on temperature in Celsius.
    Sweet spot is 22-28C (72-82F), peak ice cream weather.
    Cold temperatures below 10 degrees Celcius = negative modifiers
    Warm temperatures = low positive integers
    Hot outside? = HIGH POSITIVE MODIFIERS
    """
    if temperature >= 28:
        return 0.15
    elif temperature >= 22:
        return 0.10
    elif temperature >= 17:
        return 0.03
    elif temperature >= 10:
        return -0.03
    elif temperature >= 5:
        return -0.07
    else:
        return -0.10


def get_hour_modifier(hour, month):
    """
    Since the shops normally have different open and close times throughout the year we will be taking into account the month. 
    If it is summer the shop opens earlier than normal. 
    Depending on this, we will then be able to calculate the modifier and reutrn it.
    
    Based on a normal average pattern.
    Morning = low time
    Hour after Open = Crowd Starts increasing
    Noon = average
    afternoon = Peak time
    evening = Decreasing numbers but still high
    Night = Lowers

    Input: Hour and month
    Output: modifier 
    """
    if month in [6, 7, 8]:
        opening_hour = 10
    else:
        opening_hour = 11

    if hour >= 15 and hour <= 18:
        return 0.12
    elif hour >= 13 and hour < 15:
        return 0.08
    elif hour >= 19 and hour < 21:
        return 0.05
    elif hour == opening_hour:
        return -0.05
    elif hour == opening_hour + 1:
        return 0.0
    else:
        return 0.02


def generate_crowd_level(df):
    """
    Generates a crowd level score between 0 and 1 for each row using:
    - A base score from the month (season)
    - Modifiers added for temperature, hour of day, and weekend
    - A multiplier based on weather condition, applied to the WHOLE score
      so severe weather suppresses everything, not just the seasonal base.
      We are able to make sure a thunderstorm in summer will cause crowd levels to be lower than usual, 
      whereas a sunny day in winter may have higher crowd levels than normal.
    - A small amount of random noise to simulate natural variation (random between 0 and 0.05)

    Input:
    df: pd.DataFrame with columns month, weather, temperature, hour, weekend

    Output:
    df: same dataframe with a new crowd_level column added
    """
    crowd_scores = []
    
    for i in range(len(df)):
        month = df["month"][i]
        weather = df["weather"][i]
        temperature = df["temperature"][i]
        hour = df["hour"][i]
        weekend = df["weekend"][i]

        # --- Base score from month ---
        score = MONTH_BASE[month]
        # --- Temperature modifier ---
        score += get_temperature_modifier(temperature)
        # --- Hour modifier ---
        score += get_hour_modifier(hour,month)
        # --- Weekend modifier ---
        if weekend == 1:
            score += 0.10
        else:
            score -= 0.03

        # Ensure score is between 0 and 1 
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        # --- Weather multiplier applied to the WHOLE score ---
        # This is the key fix: severe weather suppresses everything,
        # not just the seasonal base. A thunderstorm during peak rush hour
        # in July should still end up very low, not just "less high".
        multiplier = WEATHER_MULTIPLIERS.get(weather, 0.70)
        score = score * multiplier

        # --- Random noise (0 to 0.05) to simulate natural variation ---
        noise = np.random.uniform(0, 0.05)
        score += noise

        # --- Final clip between 0 and 1 ---
        if score > 1.0:
            score = 1.0
        if score < 0.0:
            score = 0.0

        crowd_scores.append(round(score, 2))

    df["crowd_level"] = crowd_scores
    return df

# Fetch data
df = fetch_weather_data("2024-06-01", "2026-06-18")
weather_labels = []

for i in range(len(df)):
    code = df["weathercode"][i]
    label = decode_weathercode(code)
    weather_labels.append(label)

df["weather"] = weather_labels

# Save to CSV
df.to_csv("weather_dataset.csv", index=False)

print("Dataset created successfully!")


