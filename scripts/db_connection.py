from sqlalchemy import create_engine
import pandas as pd
from urllib.parse import quote_plus

class DBConnection:
    def __init__(self, dbname, user, password, host='localhost', port='5432'):
        self.dbname = dbname
        self.user = quote_plus(user)  # URL encode the user
        self.password = quote_plus(password)  # URL encode the password
        self.host = host
        self.port = port
        self.engine = None

    def connect(self):
        '''Establish a connection to the PostgreSQL database using SQLAlchemy.'''
        try:
            # Create the connection string using encoded credentials
            db_url = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'
            self.engine = create_engine(db_url)
            print("Successfully connected to the database!")
        except Exception as e:
            print(f"Error connecting to the database: {e}")

    def close(self):
        '''Close the database connection.'''
        if self.engine:
            self.engine.dispose()
            print("Connection closed.")

    def fetch_data(self, query):
        '''Fetch data from the database and return it as a pandas DataFrame.'''
        if self.engine is None:
            print("Connection is not established. Please connect first.")
            return None
        try:
            df = pd.read_sql_query(query, self.engine)
            if not df.empty:
                return df
            else:
                print("No data returned for the query.")
                return None
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
