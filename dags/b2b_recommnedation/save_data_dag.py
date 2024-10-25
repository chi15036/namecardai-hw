from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from b2b_recommnedation.save_data import save_test_data_to_pg, save_train_data_to_pg

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 10, 23),
}

with DAG("load_data", default_args=default_args, schedule_interval="@daily") as dag:
    load_test_data_to_postgres = PythonOperator(
        task_id="load_test_data_to_postgres", python_callable=save_test_data_to_pg
    )

    load_train_data_to_postgres = PythonOperator(
        task_id="load_train_data_to_postgres",
        python_callable=save_train_data_to_pg,
    )
