import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
tennis = pd.read_csv('tennis_stats.csv')
tennis.drop(columns=['Player'], inplace=True)
tennis_numeric = tennis.select_dtypes(include='number')

X = tennis_numeric.drop(columns=['Winnings'])   
y = tennis_numeric['Wins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature filtering based on correlation
def drop_low_correlated_columns(df, target_series, threshold):
    df_with_target = df.copy()
    df_with_target['Winnings'] = target_series
    corr_matrix = df_with_target.corr()
    # Only keep columns with abs(corr) >= threshold, and exclude the target itself
    keep_cols = corr_matrix['Winnings'][abs(corr_matrix['Winnings']) >= threshold].index
    keep_cols = keep_cols.drop('Winnings')
    return list(keep_cols)
selected_columns = drop_low_correlated_columns(X_train, y_train, 0.2)

# Filter both train and test with same column list
X_train_filtered = X_train[selected_columns]
X_test_filtered = X_test[selected_columns]
# Define models and hyperparameters
models = {
    'RandomForest': {
        'estimator': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    },
    'GradientBoosting': {
        'estimator': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    }
}

results = []

# Train and evaluate each model
for name, config in models.items():
    print(f"\nTraining {name}...")
    
    grid = GridSearchCV(config['estimator'], config['params'], 
                        cv=5, scoring='neg_root_mean_squared_error')
    grid.fit(X_train_filtered, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_filtered)
    
    # Scores
    test_rmse = root_mean_squared_error(y_test, y_pred)
    train_rmse = root_mean_squared_error(y_train, best_model.predict(X_train_filtered))
    cv_rmse = -grid.best_score_
    
    print(f"Best params: {grid.best_params_}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"CV RMSE: {cv_rmse:.4f}")
    
    results.append({
        'Model': name,
        'Best Params': grid.best_params_,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'CV RMSE': cv_rmse
    })

# Summary
print("\nModel Comparison:")
results_df = pd.DataFrame(results)
print(results_df.sort_values(by='Test RMSE'))
