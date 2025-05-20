"""
ML Pipeline DAG for training and logging ML models using MLflow.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from create_model import train_model, preprocess_data, scale_data, split_data


default_args = {
    'owner': 'harsh',
    'start_date': datetime(2025, 5, 8),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    dag_id='ml_pipeline',
    default_args=default_args,
    description='A DAG to train and log ML models using MLflow',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'mlflow']
) as dag:
    

    preprocess_op = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        provide_context=True
    )

    scale_op = PythonOperator(
        task_id='scale_data',
        python_callable=scale_data,
        op_kwargs={'x': "{{ ti.xcom_pull(task_ids='preprocess_data')[0] }}"},
        provide_context=True
    )

    split_op = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={
            'x': "{{ ti.xcom_pull(task_ids='scale_data') }}",
            'y': "{{ ti.xcom_pull(task_ids='preprocess_data')[1] }}"
        },
        provide_context=True
    )

    train_op = PythonOperator(
        task_id='train',
        python_callable=train_model,
        provide_context=True
    )

    # Set task dependencies
    preprocess_op >> scale_op >> split_op >> train_op
