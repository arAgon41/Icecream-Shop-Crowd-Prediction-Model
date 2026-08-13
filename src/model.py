import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("../data/model_ready_pittsburgh.csv")

target_col = "crowd_level"
X = df.drop(columns=[target_col])
y = df[target_col]

# 80 - 20 train - test split using randomized splitting 
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train the model using the random forest regressor
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(Xtrain, ytrain)

y_pred = rf.predict(Xtest)
# calculate the average error
mae = mean_absolute_error(ytest, y_pred)
# calculate how off the model is on its worst predictions
rmse = np.sqrt(mean_squared_error(ytest, y_pred))
# calculate overall accuracy of the model
r2 = r2_score(ytest, y_pred)

print("Model Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)
print("R^2:", r2)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances")
print(importances)

residuals = ytest - rf.predict(Xtest)

plt.scatter(rf.predict(Xtest), residuals)
plt.axhline(0, color='red')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

# saving model to pickle file
with open("Ice-Cream-Shop-Crowd-Prediction-Model/data/CrowdLevelPredictionModel.pkl", "wb") as f:
    pickle.dump(rf, f)
print("\nModel saved")
