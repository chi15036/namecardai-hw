from utils.postgres_utils import PostgresUtils
import os
import pandas as pd


def _create_table_test_data(cursor):
    PostgresUtils.execute_query(
        cursor,
        """
    CREATE TABLE IF NOT EXISTS test_data (
        member_no TEXT,
        name TEXT,
        company TEXT,
        title TEXT,
        background TEXT,
        company_url TEXT,
        linkedin_url TEXT,
        primary key(member_no)
    );
    """
    )


def _create_table_train_data(cursor):
    PostgresUtils.execute_query(
        cursor,
        """
    CREATE TABLE IF NOT EXISTS train_data (
        company_a TEXT,
        company_a_background TEXT,
        company_b TEXT,
        company_b_background TEXT,
        previous_collaboration INTEGER,
        collaboration_reason TEXT,
        primary key(company_a, company_b)
    );
    """
    )


def _upsert_data_test_data(cursor, data):
    for _, row in data.iterrows():
        PostgresUtils.execute_query(
            cursor,
            """
        INSERT INTO test_data (member_no, name, company, title, background, company_url, linkedin_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (member_no)
        DO UPDATE SET
            name = EXCLUDED.name,
            company = EXCLUDED.company,
            title = EXCLUDED.title,
            background = EXCLUDED.background,
            company_url = EXCLUDED.company_url,
            linkedin_url = EXCLUDED.linkedin_url;
        """,
            (
                row["member_no"],
                row["name"],
                row["company"],
                row["title"],
                row["background"],
                row["company_url"],
                row["linkedin_url"],
            ),
        )


def _upsert_data_train_data(cursor, data):
    for _, row in data.iterrows():
        PostgresUtils.execute_query(
            cursor,
            """
        INSERT INTO train_data (company_a, company_a_background, company_b, company_b_background, previous_collaboration, collaboration_reason)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (company_a, company_b)
        DO UPDATE SET
            company_a_background = EXCLUDED.company_a_background,
            company_b_background = EXCLUDED.company_b_background,
            previous_collaboration = EXCLUDED.previous_collaboration,
            collaboration_reason = EXCLUDED.collaboration_reason;
        """,
            (
                row["Company A"],
                row["Company A Background"],
                row["Company B"],
                row["Company B Background"],
                row["Previous Collaboration"],
                row["Collaboration Reason"],
            ),
        )


def save_test_data_to_pg():
    df = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "data", "SampleData.xlsx")
    )
    conn = PostgresUtils.connect_to_postgres()
    cursor = conn.cursor()
    _create_table_test_data(cursor)
    _upsert_data_test_data(cursor, df)
    PostgresUtils.commit_and_close(conn, cursor)


def save_train_data_to_pg():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "training_data.csv")
    )
    conn = PostgresUtils.connect_to_postgres()
    cursor = conn.cursor()
    _create_table_train_data(cursor)
    _upsert_data_train_data(cursor, df)
    PostgresUtils.commit_and_close(conn, cursor)
