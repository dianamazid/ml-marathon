import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda_and_feature_engineering(input_file, output_file):
    # Load the preprocessed dataset
    df = pd.read_csv(input_file)
    
    # Clean up column names
    df.columns = df.columns.str.replace('[^\x00-\x7F]', '')  # Remove non-ASCII characters
    df.columns = df.columns.str.replace(' ', '_')            # Replace spaces with underscores
    df.columns = df.columns.str.strip()                      # Strip leading/trailing whitespaces
    
    # Check cleaned column names
    print("Cleaned Column names in the dataset:")
    print(df.columns)
    
    # Verify if 'marathon_time' (or 'MarathonTime') column exists
    target_column = 'marathon_time'  # Update this with the correct column name
    if target_column not in df.columns:
        target_column = 'MarathonTime'  # Another possible variation
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in the dataset.")
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Distribution of the target variable (marathon time)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column], kde=True)
    plt.title('Distribution of Marathon Times')
    plt.xlabel('Marathon Time')
    plt.ylabel('Frequency')
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    
    # Scatter plots for key features vs marathon time
    key_features = ['weekly_mileage', 'average_pace', 'longest_run_distance']
    for feature in key_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[feature], y=df[target_column])
        plt.title(f'{feature} vs Marathon Time')
        plt.xlabel(feature)
        plt.ylabel('Marathon Time')
        plt.show()
    
    # Feature Engineering
    df['pace_per_mile'] = df[target_column] / df['total_distance']
    df['training_intensity'] = df['average_pace'] / df['rest_days']
    
    # Display the first few rows of the dataset with new features
    print("\nDataset with engineered features:")
    print(df.head())
    
    # Save the dataset with engineered features
    df.to_csv(output_file, index=False)
    print(f"\nEngineered data saved to {output_file}")

if __name__ == "__main__":
    input_file = '../data/preprocessed_marathon_data.csv'
    output_file = '../data/engineered_marathon_data.csv'
    perform_eda_and_feature_engineering(input_file, output_file)
