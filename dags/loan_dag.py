"""
Loan Default Prediction DAG
===========================
An Airflow DAG that orchestrates a loan default prediction pipeline
using Random Forest classification.

Tasks:
1. load_data_task: Load loan data from CSV
2. preprocess_task: Encode, scale, and split data
3. build_model_task: Train and save Random Forest model
4. evaluate_task: Evaluate model on test set

Author: Balaji
Course: MLOps - Northeastern University
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.pipeline import load_data, preprocess_data, build_save_model, evaluate_model
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for the DAG
default_args = {
    'owner': 'balaji',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG instance
dag = DAG(
    'Loan_Default_Prediction',
    default_args=default_args,
    description='A classification pipeline to predict loan defaults using Random Forest',
    schedule_interval=None,  # Manual triggering
    catchup=False,
    tags=['mlops', 'classification', 'finance']
)

# Task 1: Load data from CSV
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Preprocess data (encode, scale, split)
preprocess_task = PythonOperator(
    task_id='preprocess_task',
    python_callable=preprocess_data,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 3: Build and save the Random Forest model
build_model_task = PythonOperator(
    task_id='build_model_task',
    python_callable=build_save_model,
    op_args=[preprocess_task.output, "loan_model.pkl"],
    dag=dag,
)

# Task 4: Evaluate model on test set
evaluate_task = PythonOperator(
    task_id='evaluate_task',
    python_callable=evaluate_model,
    op_args=["loan_model.pkl", build_model_task.output, preprocess_task.output],
    dag=dag,
)


load_data_task >> preprocess_task >> build_model_task >> evaluate_task

# Allow CLI interaction
if __name__ == "__main__":
    dag.cli()