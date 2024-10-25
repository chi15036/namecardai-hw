from datetime import datetime
import os
import pickle

import numpy as np
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from utils.openai_utils import OpenaiUtils
from utils.postgres_utils import PostgresUtils

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 10, 23),
}


def transform_background_to_features(**kwargs):
    openai_util = OpenaiUtils()
    conn = PostgresUtils.connect_to_postgres()
    query = "SELECT company_a_background, company_b_background, previous_collaboration FROM train_data"
    df = PostgresUtils.read_sql(query, conn)
    PostgresUtils.commit_and_close(conn)

    df["company_a_embedding"] = df["company_a_background"].apply(
        openai_util.get_embedding
    )
    df["company_b_embedding"] = df["company_b_background"].apply(
        openai_util.get_embedding
    )

    df["cosine_similarity"] = df.apply(
        lambda row: cosine_similarity(
            [row["company_a_embedding"]], [row["company_b_embedding"]]
        )[0][0],
        axis=1,
    )
    print(df["cosine_similarity"])
    kwargs["ti"].xcom_push(key="transformed_data", value=df)


def validate_model(**kwargs):
    df = kwargs["ti"].xcom_pull(
        key="transformed_data", task_ids="transform_background_to_features"
    )
    X = df["cosine_similarity"].values.reshape(-1, 1)
    y = df["previous_collaboration"]

    skf = StratifiedKFold(n_splits=5)
    aucs, precisions, recalls = [], [], []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = CatBoostClassifier(
            iterations=100,
            depth=3,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=0,
        )
        train_pool = Pool(data=X_train, label=y_train)
        model.fit(train_pool)
        model.set_probability_threshold(0.75)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        aucs.append(roc_auc_score(y_test, y_pred_proba))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))

    print(
        f"Mean AUC: {np.mean(aucs)}, Mean Precision: {np.mean(precisions)}, Mean Recall: {np.mean(recalls)}"
    )


def serialize_model(**kwargs):
    df = kwargs["ti"].xcom_pull(
        key="transformed_data", task_ids="transform_background_to_features"
    )
    X = df["cosine_similarity"].values.reshape(-1, 1)
    y = df["previous_collaboration"]

    final_model = CatBoostClassifier(
        iterations=100, depth=3, learning_rate=0.1, loss_function="Logloss", verbose=0
    )
    final_train_pool = Pool(data=X, label=y)
    final_model.fit(final_train_pool)
    final_model.set_probability_threshold(0.75)

    with open(
        os.path.join(os.path.dirname(__file__), "data", "catboost_model.pkl"), "wb"
    ) as model_file:
        pickle.dump(final_model, model_file)


with DAG("train_model", default_args=default_args, schedule_interval="@daily") as dag:
    transform_background_to_features = PythonOperator(
        task_id="transform_background_to_features",
        python_callable=transform_background_to_features,
    )

    validate_model = PythonOperator(
        task_id="validate_model", python_callable=validate_model
    )

    serialize_model = PythonOperator(
        task_id="serialize_model", python_callable=serialize_model
    )

    transform_background_to_features >> validate_model >> serialize_model
