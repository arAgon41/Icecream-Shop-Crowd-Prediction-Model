# Icecream-Shop-Crowd-Prediction-Model
Machine learning model created to predict the crowd levels at an average ice cream shop given features like time, day, month, temperature, etc. 
The goal is to create a dashboard that can be used to help small ice cream shops predict crowd levels better so that they have a sufficient amount of supplies on any given day.

# Steps I will be taking
I will be using features like, weather, month, day, hour, temperature, cloudcover to predict the amount of crowd in a typical ice cream shop.
To do this we will need to:
1. Input past weather details into a dataset
2. Columns = Month, Day, Hour, Sunny or Not(0 or 1), Cloudcover, temperature, weekend(0 or 1).
3. Using observations, we will make guesses on what the crowd will be like at the ice cream shop given the features.
4. Next we will standardize the data to ensure that the model doesn't give more importance to 1 feature over the other.
5. We will then train the model, and predict the crowd on a day that was not tested.
6. Finally we will deploy the model on a streamlit based application for easier interactibility.

# Limitations
Due to not having access to real foot traffic data of any ice cream shop, we are creating a simulation dataset based on assumptions, and few visual observations. We are using python to extrapolate the observations onto the whole dataset while also adding a little bit of variation between the days.

# Dataset
Weather data was inputted using API
Crowd levels were calculated using visual observations, and assumption based point system to calculate foot traffic. 

# Files
icecreamCrowd.csv - created data set consisting of all features  
Display.py - consists of code to run the UI of the model, and generates charts to predict the crowd throughout the day.  
IcecreamCrowdGen - Generates the sample dataset for ice cream shop crowds  
IceCreamShopCrowdPrediction.py - Standardizes the data, and trains the model to predict values  

# Tech Stack
Python  
Scikit-learn  
Pandas  
NumPy  
Streamlit

# Further improvements
Since the main goal of this project is to predict the foot traffic at a location, we are able to do this for any locations as long as data is inputted for the particular location. To increase accuracy even more, we can add even more features like: nearbyevents, holiday_or_not, etc.
Using real world data would allow us to predict foot traffic for a specific location much more accurately rather than using a base average for all locations.


We can use real datasets to train a model to be even more accurate based on the specific location. This repo could be used to predict the crowd levels of any location provided data is given to train models.

# Contact
Riswanth Haris Sundaresh Babu - 
