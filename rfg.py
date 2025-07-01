import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Load data
tennis = pd.read_csv('tennis_stats.csv')
tennis.drop(columns=['Player'], inplace=True)
tennis_numeric = tennis.select_dtypes(include='number')

# Define features and target
X = tennis_numeric.drop(columns=['Winnings'])   
y = tennis_numeric['Wins']

# Drop features weakly correlated with Wins
def drop_low_correlated_columns(df, target_series, threshold):
    df_with_target = df.copy()
    df_with_target['Winnings'] = target_series
    corr_matrix = df_with_target.corr()
    low_corr_cols = corr_matrix['Winnings'][abs(corr_matrix['Winnings']) < threshold].index
    df_reduced = df.drop(columns=low_corr_cols, errors='ignore')
    return df_reduced

X_filtered = drop_low_correlated_columns(X, y, threshold=0.2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

print("Columns used for training:", X_filtered.columns.tolist())

# Train model
rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate with RMSE
rmse = mean_squared_error(y_test, y_pred)
mean_y = y_test.mean()
rmse_pct = rmse / mean_y

print(f"RMSE: {rmse:.2f}")
print(f"RMSE as % of mean: {rmse_pct:.2%}")

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(tennis_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix of Tennis Stats")
plt.show()

sns.histplot(y, kde=True)
plt.title("Wins distribution")
plt.show()


importances = rf.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importances (Random Forest)")
plt.show()
plt.clf()
#testing cross-validation
scores = cross_val_score(rf, X_filtered, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_cv = -scores

print(f"Cross-validated RMSE scores: {rmse_cv}")
print(f"Mean CV RMSE: {rmse_cv.mean():.4f}")


# testing for overfitting
train_pred = rf.predict(X_train)
train_rmse = mean_squared_error(y_train, train_pred)
print(f"Training RMSE: {train_rmse:.4f}")