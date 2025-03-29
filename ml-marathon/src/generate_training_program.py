import pandas as pd
import joblib

def generate_training_program(model, input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Ensure 'MarathonTime' exists in the dataset
    assert 'MarathonTime' in df.columns, "Column 'MarathonTime' not found in the dataset."

    # Separate features and target variable
    X = df.drop(columns=['MarathonTime'])  # Features
    y = df['MarathonTime']  # Target variable

    # Generate predictions
    df['PredictedMarathonTime'] = model.predict(X)

    # Create a simple training program based on predictions
    df['TrainingProgram'] = df['PredictedMarathonTime'].apply(generate_program)

    # Save the training programs to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Training programs saved to {output_file}")

def generate_program(predicted_time):
    # Placeholder function to generate a training program based on predicted time
    if predicted_time < 3.5:
        return "Advanced"
    elif predicted_time < 4.5:
        return "Intermediate"
    else:
        return "Beginner"

if __name__ == "__main__":
    input_file = '../data/engineered_marathon_data.csv'
    output_file = '../data/training_programs.csv'
    model_file = 'best_random_forest_model.joblib'

    # Load the best model
    best_model = joblib.load(model_file)
    
    # Generate training programs
    generate_training_program(best_model, input_file, output_file)
