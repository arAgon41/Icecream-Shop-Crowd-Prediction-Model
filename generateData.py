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
    Lat: float
    Lon: float
    Returns:
    A data frame with all weather data inputted.
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

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    # Sunny or not (weathercode 0 = clear)
    df["sunny"] = df["weathercode"].apply(lambda x: 1 if x == 0 else 0)

    return df

# Fetch data
df = fetch_weather_data("2024-06-01", "2026-06-18")

# Save to CSV
df.to_csv("weather_dataset.csv", index=False)

print("Dataset created successfully!")
