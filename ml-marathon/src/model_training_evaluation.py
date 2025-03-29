import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_and_evaluate_model(input_file, output_model_file):
    # Load the preprocessed dataset
    df = pd.read_csv(input_file)

    # Ensure 'MarathonTime' exists in the dataset
    assert 'MarathonTime' in df.columns, "Column 'MarathonTime' not found in the dataset."

    # Separate features and target variable
    X = df.drop(columns=['MarathonTime'])  # Features
    y = df['MarathonTime']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestRegressor(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform GridSearchCV to find the best parameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, output_model_file)
    print(f"Best model saved to {output_model_file}")

    # Make predictions with the best model
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")

if __name__ == "__main__":
    input_file = '../data/engineered_marathon_data.csv'
    output_model_file = 'best_random_forest_model.joblib'
    
    # Train and evaluate the model
    train_and_evaluate_model(input_file, output_model_file)
