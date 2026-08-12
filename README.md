# Icecream-Shop-Crowd-Prediction-Model
Machine learning model created to predict the crowd levels at an average ice cream shop given features like time, day, month, temperature, etc. 
The goal is to create a dashboard that can be used to help small ice cream shops predict crowd levels better so that they have a sufficient amount of supplies on any given day.

# Demo

# Overview
Due to privacy reasons, foot traffic data for local shops aren't public. To solve this problem I simulated crowd levels based on weather conditions(weather,temperature), hour, month, and is_weekend and trained a model to learn the relationship and predict crowd levels in the future. People will be able to interact with the model using a basic UI.

# Project Structure
```
Icecream-Shop-Crowd-Prediction-Model/
├── dashboard/
│   └── display.py                    # UI for live predictions
├── data/
│   ├── model_ready_pittsburgh.csv
│   └── PittsburghWeather.csv
├── models/
│   └── CrowdLevelPredictionModel.pkl  # Saved Serialized model
├── src/
│   ├── CleanData.py                   # Preprocessing
│   ├── generateData.py                # Simulation logic
│   └── model.py                        # Training + evaluation
├── ShopDataSetSampleAnalysis.ipynb    # Dataset Analysis notebook
└── README.md
```

# Dataset
- Source: Weather Data -> (https://open-meteo.com/)
- Features: Temperature, month, hour, weather, is_weekened, is_sunny
- Target: Crowdlevel(0-1)
- Simulation Logic: Refer to generateData.py for full logic.

# Approach
1. Data collection — pulled historical weather data via OpenMeteo
2. Simulation — generated synthetic crowd-level labels using a rule-based formula tied to weather + calendar features
3. Preprocessing — converted weathercode to real weather, encoded categorical weather conditions, engineered features like is_weekend
4. Modeling — Used Random Train/Test split for 80|20 sets. Trained a Random Forest Regressor
6. UI — built a simple interface where a user inputs weather conditions and gets a predicted crowd level

# Tech Stack
Python  
Scikit-learn  
Pandas  
NumPy  
Streamlit


# Further improvements
- Training model based on user given dataset to predict for other locations.
- Training model using real foot traffic data.
- Adding features like is_holiday, precipitation, weatheralertlevel to increase accuracies on real datasets.

# Contact
Riswanth Haris Sundaresh Babu - 
