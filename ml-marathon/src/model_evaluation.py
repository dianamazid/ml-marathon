import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, input_file):
    # Load the preprocessed dataset
    df = pd.read_csv(input_file)

    # Ensure 'MarathonTime' exists in the dataset
    assert 'MarathonTime' in df.columns, "Column 'MarathonTime' not found in the dataset."

    # Separate features and target variable
    X = df.drop(columns=['MarathonTime'])  # Features
    y = df['MarathonTime']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions with the model
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")

    # Plot true vs predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.show()

if __name__ == "__main__":
    input_file = '../data/engineered_marathon_data.csv'
    model_file = 'best_random_forest_model.joblib'

    # Load the best model
    best_model = joblib.load(model_file)
    
    # Evaluate the model
    evaluate_model(best_model, input_file)
