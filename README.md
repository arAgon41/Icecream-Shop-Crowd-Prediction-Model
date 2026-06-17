# Icecream-Shop-Crowd-Prediction-Model
Machine learning model created to predict the crowd levels at an average ice cream shop given features like time, day, month, temperature, etc. 
The goal is to create a dashboard that can be used to help small ice cream shops predict crowd levels better so that they have a sufficient amount of supplies on any given day.

We will be using features like, weather, month, day, hour, temperature, cloudcover to predict the amount of crowd in a typical ice cream shop.
To do this we will need to:
1. Input past weather details into a dataset
2. Columns = Month, Day, Hour, Sunny or Not(0 or 1), Cloudcover, temperature, weekend(0 or 1).
3. Using observations, we will make guesses on what the crowd will be like at the ice cream shop given the features.
4. Next we will standardize the data to ensure that the model doesn't give more importance to 1 feature over the other.
5. We will then train the model, and predict the crowd on a day that was not tested.
6. Finally we will deploy the model on a streamlit based application for easier interactibility.
