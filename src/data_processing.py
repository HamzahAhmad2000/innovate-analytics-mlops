import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os

def load_data(raw_data_path):
    """Loads data from the specified path."""
    print(f"Loading raw data from {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded dataframe with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Applies basic preprocessing steps."""
    print("Preprocessing data...")
    # Example: Simple fillna or dropna - replace with actual logic
    df_processed = df.dropna()
    print(f"Shape after preprocessing: {df_processed.shape}")
    # Add more steps: encoding, feature engineering, scaling etc.
    return df_processed

def split_data(df, test_size, random_state, target_column='target'):
    """Splits data into train and test sets."""
    print(f"Splitting data with test_size={test_size}, random_state={random_state}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    # Combine features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df

def save_data(df, output_path):
    """Saves the processed data."""
    print(f"Saving data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data into training and testing sets.")
    parser.add_argument("--raw-data", required=True, help="Path to raw data CSV file")
    parser.add_argument("--processed-path", required=True, help="Directory to save processed data files")
    parser.add_argument("--test-split-ratio", type=float, default=0.2, help="Ratio for the test dataset split")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for data splitting")
    parser.add_argument("--target-column", type=str, default="target", help="Name of the target variable column")


    args = parser.parse_args()

    print("Starting data processing script...")
    raw_df = load_data(args.raw_data)
    processed_df = preprocess_data(raw_df) # Apply preprocessing
    train_df, test_df = split_data(processed_df, args.test_split_ratio, args.random_state, args.target_column)

    # Define output paths
    train_output_path = os.path.join(args.processed_path, "train.csv")
    test_output_path = os.path.join(args.processed_path, "test.csv")

    save_data(train_df, train_output_path)
    save_data(test_df, test_output_path)
    print(f"Data processing finished. Processed files saved in {args.processed_path}") 