from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# --- Configuration ---
# IMPORTANT: This path MUST match the mount point of your project inside the Airflow worker container
PROJECT_ROOT = '/opt/airflow/project'
# Path to the Python executable within the virtual environment (if used inside container)
# Ensure packages from requirements.txt are installed in this environment or globally in the container.
PYTHON_EXEC = '/usr/local/bin/python' # Or '/opt/airflow/project/venv/bin/python' if using venv inside container

RAW_DATA_FILENAME = 'client_data.csv' # Name of your raw data file
TARGET_COLUMN_NAME = 'target' # Replace with your actual target column name

# Derived paths
RAW_DATA_PATH = f'{PROJECT_ROOT}/data/raw/{RAW_DATA_FILENAME}'
PROCESSED_PATH = f'{PROJECT_ROOT}/data/processed'
SCRIPT_PATH = f'{PROJECT_ROOT}/src/data_processing.py'

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='ml_data_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for processing ML data',
    schedule_interval=None, # Trigger manually or set schedule e.g., '@daily'
    start_date=datetime(2023, 1, 1), # Use a fixed past start date
    catchup=False,
    tags=['mlops', 'etl', 'innovate-analytics'],
) as dag:

    # Task to run the data processing script
    # NOTE: Using BashOperator for simplicity. For more complex dependencies or environments,
    # consider DockerOperator or PythonOperator with the project installed as a package.
    run_etl = BashOperator(
        task_id='run_data_processing_script',
        bash_command=(
            f"echo 'Running ETL script...'; "
            f"{PYTHON_EXEC} {SCRIPT_PATH} "
            f"--raw-data {RAW_DATA_PATH} "
            f"--processed-path {PROCESSED_PATH} "
            f"--test-split-ratio 0.2 "
            f"--random-state 42 "
            f"--target-column {TARGET_COLUMN_NAME}; " # Pass target column name
            f"echo 'ETL script finished.'"
        ),
        # Ensure the script logs errors properly for Airflow to catch failures
    )

    # Define task dependencies if needed (e.g., task_download >> run_etl)
    # run_etl # If it's the only task 