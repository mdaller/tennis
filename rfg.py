import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
tennis = pd.read_csv('tennis_stats.csv')
tennis.drop(columns=['Player'], inplace=True)
tennis_numeric = tennis.select_dtypes(include='number')

# Define features and target
X = tennis_numeric.drop(columns=['Winnings'])   
y = tennis_numeric['Wins']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop features weakly correlated with Wins
def drop_low_correlated_columns(df, target_series, threshold):
    df_with_target = df.copy()
    df_with_target['Winnings'] = target_series
    corr_matrix = df_with_target.corr()
    low_corr_cols = corr_matrix['Winnings'][abs(corr_matrix['Winnings']) < threshold].index
    df_reduced = df.drop(columns=low_corr_cols, errors='ignore')
    return df_reduced

X_train_filtered = drop_low_correlated_columns(X_train, y_train, threshold=0.2)
X_test_filtered = drop_low_correlated_columns(X_test, y_train, threshold=0.2)

print("Columns used for training:", X_train_filtered.columns.tolist())

# Grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                           param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_filtered, y_train)

best_model = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best CV RMSE:", -grid_search.best_score_)

# Predict on test set
y_pred = best_model.predict(X_test_filtered)

# Evaluate test set
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

# Distribution of target variable
sns.histplot(y, kde=True)
plt.title("Wins distribution")
plt.show()

# Feature importances
importances = best_model.feature_importances_
features = X_train_filtered.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importances (Random Forest)")
plt.show()

# Cross-validation scores
scores = cross_val_score(best_model, X_train_filtered, y_train, cv=5, scoring='neg_root_mean_squared_error')
rmse_cv = -scores
print(f"Cross-validated RMSE scores: {rmse_cv}")
print(f"Mean CV RMSE: {rmse_cv.mean():.4f}")

# Check overfitting (train RMSE)
train_pred = best_model.predict(X_train_filtered)
train_rmse = mean_squared_error(y_train, train_pred)
print(f"Training RMSE: {train_rmse:.4f}")
