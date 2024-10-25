import os
import psycopg2
import pandas as pd

class PostgresUtils:
    @staticmethod
    def connect_to_postgres():
        return psycopg2.connect(
            host="postgres",
            database=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
        )

    @staticmethod
    def read_sql(query, conn):
        return pd.read_sql(query, conn)

    @staticmethod
    def execute_query(cursor, query, params=None):
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

    @staticmethod
    def commit_and_close(conn, cursor=None):
        conn.commit()
        conn.close()
        if cursor:
            cursor.close()
