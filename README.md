# Icecream-Shop-Crowd-Prediction-Model 🍦
Machine learning model created to predict the crowd levels at an average ice cream shop given features like time, day, month, temperature, etc. This model is trained on synthetic data so we can't predict real world traffic with this. 
# Demo 🌟
<img width="1512" height="782" alt="Screenshot 2026-08-13 at 1 34 29 PM" src="https://github.com/user-attachments/assets/6a2639b0-3c5e-4067-be05-5ac0fcc67967" />

# Overview 🗒️
Due to privacy reasons, foot traffic data for local shops aren't public. To solve this problem I simulated crowd levels based on weather conditions(weather,temperature), hour, month, and is_weekend and trained a model to learn the relationship and predict crowd levels in the future. People will be able to interact with the model using a basic UI.
# Project Structure 📂
```
Icecream-Shop-Crowd-Prediction-Model/
├── dashboard/
│   └── display.py                     # UI for live predictions
├── data/
│   ├── model_ready_pittsburgh.csv.    # model readable csv
│   └── PittsburghWeather.csv          # human readable csv
├── models/
│   └── CrowdLevelPredictionModel.pkl  # Saved Serialized model
├── src/
│   ├── CleanData.py                   # Preprocessing
│   ├── generateData.py                # Simulation logic
│   └── model.py                       # Training + evaluation
├── ShopDataSetSampleAnalysis.ipynb    # Dataset Analysis notebook
└── README.md                          # project details
└── requirements.txt                   # necessary downloads used during setup
```

# Setup 🖥️
Follow these steps to run this project on your computer!
```
git clone https://github.com/arAgon41/Icecream-Shop-Crowd-Prediction-Model
cd Icecream-Shop-Crowd-Prediction-Model
```
Installing libraries
```
pip install -r requirements.txt
```
Running UI
```
streamlit run dashboard/display.py
```
If you want to regenerate the dataset
```
python src/generateData.py
```
If you want to retrain the model
```
python src/model.py
```

# Dataset 📊
- Source: Weather Data -> (https://open-meteo.com/)
- Features: Temperature, month, hour, weather, is_weekened, is_sunny
- Target: Crowdlevel(0-1)
- Simulation Logic: Refer to generateData.py for full logic.
# Approach 🧠
1. Data collection — pulled historical weather data via OpenMeteo
2. Simulation — generated synthetic crowd-level labels using a rule-based formula tied to weather + calendar features
3. Preprocessing — converted weathercode to real weather, encoded categorical weather conditions, engineered features like is_weekend
4. Modeling — Used Random Train/Test split for 80|20 sets. Trained a Random Forest Regressor
6. UI — built a simple interface where a user inputs weather conditions and gets a predicted crowd level

## Data set Creation
## Synthetic Data Generation Overview
This project uses rule based logic to generate realistic crowd levels for an ice cream shop.  
Below is a chart that summarizes th logic used. Details are in ```generateData.py```

| **Feature** | **Crowd Level Calculation** |
|-------------|--------------------------|
| **Weather Condition** | Multiplies the base score (Clear = high demand, Storm = low demand). |
| **Temperature** | Warm days increase demand. Cold days reduce it. |
| **Hour of Day** | Afternoon is peak time; opening hour is low. |
| **Month** | Summer months have higher base scores; winter months have lower ones. |
| **Weekend** | Adds a small positive boost to the crowd level. |
| **Noise** | Adds slight randomness (0–0.05) to make the data more realistic. |
| **Shop Hours Filtering** | Only hours when the shop is open are kept (Summer: 10–9, Winter: 11–5, Spring/Fall: 11–7). |

# Tech Stack 💻
Python  
Scikit-learn  
Pandas  
NumPy  
Streamlit
Pickle 

# Model Evaluation
Model was trained on Synthetic Data, hence model was highly accurate in predicting traffic. 
| **Metric** | **Value** |
|------------|-----------|
| **Mean Absolute Error** | 0.01962 |
| **Root Mean Squared Error** | 0.02817 |
| **R² Score** | 0.98780 |

Real world data would have random spikes that would be harder to predict.

# Further improvements 🚀
- Training model based on user given dataset to predict for other locations.
- Training model using real foot traffic data.
- Adding features like is_holiday, precipitation, weatheralertlevel to increase accuracies on real datasets.
# Contact
Riswanth Haris Sundaresh Babu -
