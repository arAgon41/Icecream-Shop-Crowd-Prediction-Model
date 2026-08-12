import pandas as pd
"""
Create seperate columns for each of the weather conditions to use One Hot Encoding so model can use categorical data.
"""

def CreateWeatherColumns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a seperate column for each of the weather conditions. 
    Goal: Allowing the model to understand categorical variables by encoding them using binary classification.
    1. Create columns for the weather conditions
    2. Add columns to the data frame
    3. Remove the unnecessary columns
    4. save to a csv file 
    """
    weatherDummies = pd.get_dummies(df["weather"], prefix="weather")
    df = pd.concat([df, weatherDummies], axis=1)
    return df

def RemoveColumns(columnsToDrop: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns from the dataframe.
    """
    return df.drop(columnsToDrop, axis=1)

"""
1. Read CSV and convert to dataframe
2. Use function to create seperate weather columns
3. Drop columns that we dont need
4. Save dataframe as a new CSV file
"""
df = pd.read_csv("PittsburghWeather.csv")
df = CreateWeatherColumns(df)
columnsToDrop = ["weather", "weathercode", "cloudcover"]
df = RemoveColumns(columnsToDrop, df)
df.to_csv("model_ready_pittsburgh.csv", index=False)
print("data cleaned and ready")