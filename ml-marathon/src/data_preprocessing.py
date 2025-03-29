import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)
    
    # Handle missing values
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # Encode categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        encoder = OneHotEncoder()
        encoded_categorical = encoder.fit_transform(df[categorical_features])
        encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(), columns=encoder.get_feature_names_out(categorical_features))
        
        # Combine numerical and encoded categorical features
        df_processed = pd.concat([df[numeric_features], encoded_categorical_df], axis=1)
    else:
        df_processed = df[numeric_features]
    
    # Save preprocessed data
    df_processed.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    input_file = '../data/marathon_time_predictions.csv'
    output_file = '../data/preprocessed_marathon_data.csv'
    preprocess_data(input_file, output_file)
