random_forest_regressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

tennis = pd.read_csv('tennis_stats.csv')
tennis.drop(columns=['Player'], inplace=True)

X = tennis.drop(columns=['Wins'])   
y = tennis['Wins']

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and fit the model
rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)

# 5. Predict on test set
y_pred = rf.predict(X_test)

# 6. Evaluate using RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
mean_y = y_test.mean()
rmse_pct = rmse / mean_y

print(f"RMSE: {rmse}")
print(f"RMSE as % of mean: {rmse_pct:.2%}")