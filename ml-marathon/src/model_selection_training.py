import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def model_selection_and_training(input_file):
    # Load the preprocessed dataset
    df = pd.read_csv(input_file)

    # Check for leading/trailing spaces or special characters in column names
    df.columns = df.columns.str.strip()  # Strip leading/trailing spaces

    # Ensure 'MarathonTime' exists in the dataset (case-sensitive)
    assert 'MarathonTime' in df.columns, "Column 'MarathonTime' not found in the dataset."

    # Separate features and target variable
    X = df.drop(columns=['MarathonTime'])  # Features
    y = df['MarathonTime']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a dictionary of models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Support Vector Machine': SVR()
    }

    # Define a dictionary to store the results
    results = {}

    # Iterate over each model, train and evaluate it
    for name, model in models.items():
        # Create a pipeline to standardize the data and train the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = pipeline.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
        print(f"{name} - Mean Squared Error (MSE): {mse}")

    # Grid Search for hyperparameter tuning on the best model (Random Forest as an example)
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(random_state=42))
        ]),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Make predictions with the best model
    y_pred_best = best_model.predict(X_test)

    # Calculate Mean Squared Error (MSE) for the best model
    mse_best = mean_squared_error(y_test, y_pred_best)
    print(f"\nBest Model - Random Forest with Grid Search - Mean Squared Error (MSE): {mse_best}")

    # Optionally, save the best model
    # Example:
    # from joblib import dump
    # dump(best_model, 'best_random_forest_model.joblib')

    return best_model, results

if __name__ == "__main__":
    input_file = '../data/engineered_marathon_data.csv'
    best_model, results = model_selection_and_training(input_file)
