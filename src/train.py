import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression # Example model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split # If not using pre-split files
import argparse
import os
import yaml # For reading params.yaml
import mlflow
import mlflow.sklearn # Import the sklearn flavor

# Define default paths relative to project root
DEFAULT_PROCESSED_DATA_DIR = 'data/processed'
DEFAULT_MODEL_OUTPUT_DIR = 'models'
DEFAULT_PARAMS_FILE = 'params.yaml'
DEFAULT_TARGET_COLUMN = 'target' # Make sure this matches data processing

def load_params(params_path=DEFAULT_PARAMS_FILE):
    """Loads parameters from a YAML file."""
    print(f"Loading parameters from {params_path}")
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        if params is None:
             print(f"Warning: {params_path} is empty or invalid.")
             return {}
        print(f"Parameters loaded: {params}")
        return params
    except FileNotFoundError:
        print(f"Warning: Parameter file {params_path} not found. Using defaults.")
        return {} # Return empty dict if file not found
    except Exception as e:
        print(f"Error loading parameters from {params_path}: {e}")
        return {}


def load_processed_data(data_dir, target_column):
    """Loads pre-split train and test CSV files."""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    print(f"Loading processed data from {train_path} and {test_path}")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train ({train_path}) or Test ({test_path}) file not found in {data_dir}. Ensure ETL ran.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

    if target_column not in train_df.columns or target_column not in test_df.columns:
         raise ValueError(f"Target column '{target_column}' not found in train/test data.")

    # Separate features (X) and target (y)
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, model_params):
    """Trains the machine learning model using parameters, logs params to MLflow."""
    print(f"Training model with parameters: {model_params}")
    # Example: Logistic Regression - get params specific to it
    lr_params = model_params.get('train', {}).get('logistic_regression', {})
    print(f"Using Logistic Regression parameters: {lr_params}")

    # Log parameters to MLflow
    mlflow.log_params(lr_params)

    model = LogisticRegression(**lr_params) # Unpack specific model params
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns metrics, logs metrics to MLflow."""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Prob for positive class

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted') # Use weighted for potential imbalance
    report = classification_report(y_test, y_pred, output_dict=True) # Get report as dict

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    # print("Classification Report:\n", classification_report(y_test, y_pred)) # Print locally

    # Log metrics to MLflow
    print("Logging metrics to MLflow...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score_weighted", f1)

    # Log classification report parts (example: precision/recall for class '1')
    if '1' in report: # Check if class '1' exists
        mlflow.log_metric("precision_class_1", report['1']['precision'])
        mlflow.log_metric("recall_class_1", report['1']['recall'])
        mlflow.log_metric("f1_score_class_1", report['1']['f1-score'])

    metrics = {'accuracy': accuracy, 'f1_score_weighted': f1}
    return metrics

def save_model(model, output_dir, model_name="model.joblib"):
    """Saves the trained model artifact."""
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    model_path = os.path.join(output_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model with MLflow tracking.")
    parser.add_argument("--data-dir", default=DEFAULT_PROCESSED_DATA_DIR, help=f"Directory containing train.csv and test.csv (default: {DEFAULT_PROCESSED_DATA_DIR})")
    parser.add_argument("--model-output-dir", default=DEFAULT_MODEL_OUTPUT_DIR, help=f"Directory to save the trained model (default: {DEFAULT_MODEL_OUTPUT_DIR})")
    parser.add_argument("--params", default=DEFAULT_PARAMS_FILE, help=f"Path to parameters YAML file (default: {DEFAULT_PARAMS_FILE})")
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN, help=f"Name of the target column (default: {DEFAULT_TARGET_COLUMN})")
    parser.add_argument("--experiment-name", default="InnovateAnalytics_Default", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name (optional)")


    args = parser.parse_args()

    print("Starting model training script with MLflow...")

    # Load parameters
    params = load_params(args.params)
    model_params = params.get('model_params', {}) # Get model-specific params section
    data_params = params.get('data_params', {})

    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run Started: experiment='{args.experiment_name}', run_id='{run_id}', run_name='{args.run_name}'")

        # Log general parameters or tags
        mlflow.log_param("data_directory", args.data_dir)
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("params_file", args.params)
        mlflow.set_tag("mlflow.user", os.getenv('USER', 'unknown')) # Track user
        mlflow.set_tag("training_script", os.path.basename(__file__))

        # Ensure DVC data is checked out
        print("Checking out DVC data...")
        dvc_checkout_cmd = f'dvc checkout {os.path.join(args.data_dir)}.dvc'
        print(f"Running: {dvc_checkout_cmd}")
        checkout_status = os.system(dvc_checkout_cmd)
        if checkout_status != 0:
             print(f"Warning: dvc checkout command failed with status {checkout_status}")
        # Log DVC data version (requires dvc >= 2.0)
        # try:
        #     dvc_repo = dvc.repo.Repo('.')
        #     data_status = dvc_repo.data_status([args.data_dir])
        #     if data_status.get(args.data_dir) == 'modified':
        #         print(f"Warning: DVC data in {args.data_dir} is modified.")
        #     # This part needs careful handling of DVC API changes
        #     # Example: Log the hash of the .dvc file itself
        #     dvc_file_path = f"{args.data_dir}.dvc"
        #     if os.path.exists(dvc_file_path):
        #         with open(dvc_file_path, 'r') as f:
        #             dvc_meta = yaml.safe_load(f)
        #             data_hash = dvc_meta.get('outs', [{}])[0].get('hash')
        #             if data_hash: mlflow.set_tag("dvc_data_hash", data_hash)
        # except Exception as e:
        #     print(f"Could not log DVC info: {e}")


        # Load Data
        X_train, y_train, X_test, y_test = load_processed_data(args.data_dir, args.target_column)

        # Train Model (logs params inside)
        model = train_model(X_train, y_train, model_params)

        # Evaluate Model (logs metrics inside)
        metrics = evaluate_model(model, X_test, y_test)

        # Log the trained model using MLflow's scikit-learn flavor
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model", # Path within MLflow run artifacts
            registered_model_name="InnovateAnalyticsModel_LR" # Optional: Register model
        )
        print("Model logged to MLflow artifacts.")

        # Optional: Save model locally as well if needed for non-MLflow use cases
        local_model_path = os.path.join(args.model_output_dir, "model.joblib")
        # joblib.dump(model, local_model_path)
        # print(f"Model also saved locally to {local_model_path}")

        # Log parameter file as artifact
        mlflow.log_artifact(args.params)
        print(f"Logged {args.params} as artifact.")

        print("\nMLflow run finished.")
        print(f"Final Metrics: {metrics}")
        print(f"To view run, execute 'mlflow ui' and navigate to experiment '{args.experiment_name}'")

    print("\nModel training and evaluation finished.")
    print(f"Final Metrics: {metrics}")

    # Trivial change to trigger CI/CD $(date) 