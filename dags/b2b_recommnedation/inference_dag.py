import os
import pickle
from datetime import datetime
from itertools import combinations

import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sklearn.metrics.pairwise import cosine_similarity
from utils.openai_utils import OpenaiUtils
from utils.postgres_utils import PostgresUtils

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 10, 23),
}


def read_test_data(**kwargs):
    conn = PostgresUtils.connect_to_postgres()
    query = "SELECT member_no, background FROM test_data"
    df = PostgresUtils.read_sql(query, conn)
    PostgresUtils.commit_and_close(conn)
    kwargs["ti"].xcom_push(key="test_data", value=df)


def calculate_embeddings(**kwargs):
    openai_util = OpenaiUtils()
    df = kwargs["ti"].xcom_pull(key="test_data", task_ids="read_test_data")
    df["embedding"] = df["background"].apply(openai_util.get_embedding)
    kwargs["ti"].xcom_push(key="embeddings", value=df)


def generate_combinations(**kwargs):
    df = kwargs["ti"].xcom_pull(key="embeddings", task_ids="calculate_embeddings")
    combinations_list = list(combinations(df.to_dict("records"), 2))
    combined_df = pd.DataFrame(
        combinations_list, columns=["member_no", "match_member_no"]
    )
    kwargs["ti"].xcom_push(key="combinations", value=combined_df)


def calculate_cosine_similarity(**kwargs):
    df = kwargs["ti"].xcom_pull(key="combinations", task_ids="generate_combinations")
    df["cosine_similarity"] = df.apply(
        lambda row: cosine_similarity(
            [row["member_no"]["embedding"]], [row["match_member_no"]["embedding"]]
        )[0][0],
        axis=1,
    )
    kwargs["ti"].xcom_push(key="features", value=df)


def predict_results(**kwargs):
    df = kwargs["ti"].xcom_pull(key="features", task_ids="calculate_cosine_similarity")
    X = df["cosine_similarity"].values.reshape(-1, 1)

    model_path = os.path.join(os.path.dirname(__file__), "data", "catboost_model.pkl")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    df["prediction"] = model.predict(X)
    kwargs["ti"].xcom_push(key="predictions", value=df)


def generate_match_reason(**kwargs):
    df = kwargs["ti"].xcom_pull(key="predictions", task_ids="predict_results")
    df = df[df["prediction"] == 1]
    openai_util = OpenaiUtils()
    conn = PostgresUtils.connect_to_postgres()
    query = (
        "SELECT collaboration_reason FROM train_data WHERE previous_collaboration = 1"
    )
    reasons_df = PostgresUtils.read_sql(query, conn)
    PostgresUtils.commit_and_close(conn)
    reasons_list = reasons_df["collaboration_reason"].tolist()
    messages_list = [
        [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates collaboration reasons based on given backgrounds and past successful collaboration reasons. Don't gnerate more than one sentence",
            },
            {
                "role": "user",
                "content": f"Background 1: {row['member_no']['background']}\nBackground 2: {row['match_member_no']['background']}\nPast Reasons: {reasons_list}",
            },
        ]
        for _, row in df.iterrows()
    ]
    reasons = openai_util.mget_completion(messages_list)
    df["why"] = reasons
    kwargs["ti"].xcom_push(key="recommendation_reasons", value=df)


def format_output(**kwargs):
    df = kwargs["ti"].xcom_pull(
        key="recommendation_reasons", task_ids="generate_match_reason"
    )
    output_path = os.path.join(
        os.path.dirname(__file__), "data", "recommendation_reasons.csv"
    )
    df["member_no"] = df["member_no"].apply(lambda x: x["member_no"]).astype(int)
    df["match_member_no"] = (
        df["match_member_no"].apply(lambda x: x["member_no"]).astype(int)
    )
    swapped_df = df.copy()
    swapped_df["member_no"], swapped_df["match_member_no"] = (
        df["match_member_no"],
        df["member_no"],
    )
    df = pd.concat([df, swapped_df], ignore_index=True)
    df.sort_values(by=["member_no", "match_member_no"], inplace=True)
    df.insert(0, "match_id", range(1, len(df) + 1))  # Add match_id as a serial number
    df[["match_id", "member_no", "match_member_no", "why"]].to_csv(
        output_path, index=False
    )


with DAG(
    "inference_model", default_args=default_args, schedule_interval="@daily"
) as dag:
    read_test_data = PythonOperator(
        task_id="read_test_data",
        python_callable=read_test_data,
    )

    calculate_embeddings = PythonOperator(
        task_id="calculate_embeddings",
        python_callable=calculate_embeddings,
    )

    generate_combinations = PythonOperator(
        task_id="generate_combinations",
        python_callable=generate_combinations,
    )

    calculate_cosine_similarity = PythonOperator(
        task_id="calculate_cosine_similarity",
        python_callable=calculate_cosine_similarity,
    )

    predict_results = PythonOperator(
        task_id="predict_results",
        python_callable=predict_results,
    )

    generate_match_reason = PythonOperator(
        task_id="generate_match_reason",
        python_callable=generate_match_reason,
    )

    format_output = PythonOperator(
        task_id="format_output",
        python_callable=format_output,
    )

    (
        read_test_data
        >> calculate_embeddings
        >> generate_combinations
        >> calculate_cosine_similarity
        >> predict_results
        >> generate_match_reason
        >> format_output
    )
