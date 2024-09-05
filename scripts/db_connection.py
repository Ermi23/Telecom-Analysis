# db_connection.py in /scripts folder

import psycopg2
import pandas as pd

class DBConnection:
    def __init__(self, dbname, user, password, host='localhost', port='5432'):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Successfully connected to the database!")
        except Exception as e:
            print(f"Error connecting to the database: {e}")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Connection closed.")

    def fetch_data(self, query):
        """Fetch data from the database and return it as a pandas DataFrame."""
        if self.conn is None:
            print("Connection is not established. Please connect first.")
            return None

        try:
            df = pd.read_sql_query(query, self.conn)
            if not df.empty:
                return df
            else:
                print("No data returned for the query.")
                return None
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
