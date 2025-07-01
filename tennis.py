
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
tennis = pd.read_csv('tennis_stats.csv')
print(tennis)

# Select the features (X) and target (y) variables
features2 = tennis[['Aces', 'Wins']]
outcomes2 = tennis['Winnings']  # No need to wrap this in double brackets

# Split the data into training and testing sets
features_train2, features_test2, outcomes_train2, outcomes_test2 = train_test_split(features2, outcomes2, test_size=0.2)

# No need to reshape the features and outcomes here

# Initialize the linear regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(features_train2, outcomes_train2)

# Calculate the R-squared score on the testing data
score2 = model.score(features_test2, outcomes_test2)
print("R-squared score:", score2)

# Make predictions on the testing data
y_pred2 = model.predict(features_test2)

# Scatter plot of actual vs. predicted values
plt.scatter(outcomes_test2, y_pred2)
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.title("Actual vs. Predicted Winnings")
plt.savefig("actual vs predecited winnings.png")
plt.show()
plt.close()




mse = mean_squared_error(outcomes_test2, y_pred2)
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)
print("RMSE as a percentage of mean outcomes:", rmse / outcomes_test2.mean())

#as the RMSE 41.6% as a percentage of the mean outcomes, it is not a good metric to use here.
