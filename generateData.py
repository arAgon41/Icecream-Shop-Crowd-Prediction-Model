# Riswanth Haris Sundaresh Babu
import requests
import pandas as pd
from datetime import datetime

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
            close_hour = 21
        elif month in [12, 1, 2]:
            close_hour = 17
        else:
            close_hour = 19

        if hour < 11 or hour >= close_hour:
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
# Fetch data
df = fetch_weather_data("2024-06-01", "2026-06-18")

# Save to CSV
df.to_csv("weather_dataset.csv", index=False)

print("Dataset created successfully!")
