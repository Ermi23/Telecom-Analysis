import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
class TelecomSatisfactionAnalytics:
    def __init__(self, df, columns_to_clean):
        self.df = df
        self.columns_to_clean = columns_to_clean

    def clean_data(self):
        """Fill missing values with the mean of each column."""
        for column in self.columns_to_clean:
            if column in self.df.columns:
                self.df[column].fillna(self.df[column].mean(), inplace=True)
    
    def fill_missing_with_mean(self):
        """Fill missing values in numeric columns with the mean of those columns."""
        if self.df is not None:
            # Select only numeric columns
            numeric_columns = self.df.select_dtypes(include=['number']).columns
            
            # Fill missing values in numeric columns with the mean of each column
            for col in numeric_columns:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    # print(f"Filled missing values in column: {col}")
            print("Missing values have been filled with the mean for numeric columns.")
        else:
            print("No DataFrame to analyze.")

    def perform_clustering(self, n_clusters=2):
        """Perform k-means clustering on engagement and experience metrics."""
        engagement_metrics = ['Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)', 'TCP DL Retrans. Vol (Bytes)']
        experience_metrics = ['Avg RTT UL (ms)', 'DL TP < 50 Kbps (%)', 'UL TP < 10 Kbps (%)']
        
        # Normalize the data
        metrics = self.df[engagement_metrics + experience_metrics]
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(metrics_scaled)

    def calculate_scores(self, engagement_cluster, experience_cluster):
        """Calculate engagement and experience scores."""
        engagement_metrics = ['Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)', 'TCP DL Retrans. Vol (Bytes)']
        experience_metrics = ['Avg RTT UL (ms)', 'DL TP < 50 Kbps (%)', 'UL TP < 10 Kbps (%)']

        engagement_center = self.df[self.df['Cluster'] == engagement_cluster][engagement_metrics].mean().to_numpy()
        experience_center = self.df[self.df['Cluster'] == experience_cluster][experience_metrics].mean().to_numpy()

        self.df['Engagement Score'] = np.linalg.norm(self.df[engagement_metrics].to_numpy() - engagement_center, axis=1)
        self.df['Experience Score'] = np.linalg.norm(self.df[experience_metrics].to_numpy() - experience_center, axis=1)

    def calculate_satisfaction_score(self):
        """Calculate the satisfaction score."""
        self.df['Satisfaction Score'] = (self.df['Engagement Score'] + self.df['Experience Score']) / 2

    def report_top_satisfied(self):
        """Report the top 10 satisfied customers."""
        return self.df.nlargest(10, 'Satisfaction Score')[['MSISDN/Number', 'Satisfaction Score']]
    
    def report_Low_satisfied(self):
        """Report the top 10 least satisfied customers."""
        return self.df.nsmallest(10, 'Satisfaction Score')[['MSISDN/Number', 'Satisfaction Score']]

    def build_regression_model(self):
        """Build and evaluate a regression model for predicting satisfaction scores."""
        engagement_metrics = ['Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)', 'TCP DL Retrans. Vol (Bytes)']
        experience_metrics = ['Avg RTT UL (ms)', 'DL TP < 50 Kbps (%)', 'UL TP < 10 Kbps (%)']

        X = self.df[engagement_metrics + experience_metrics]
        y = self.df['Satisfaction Score']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        print(f'Mean Squared Error: {mse}')

    def run_kmeans_on_scores(self):
        """Run k-means clustering on engagement and experience scores."""
        scores = self.df[['Engagement Score', 'Experience Score']]
        kmeans = KMeans(n_clusters=2, random_state=42)
        self.df['Score Cluster'] = kmeans.fit_predict(scores)

    def aggregate_scores(self):
        """Aggregate average satisfaction and experience score per score cluster."""
        return self.df.groupby('Score Cluster')[['Satisfaction Score', 'Experience Score']].mean()

    def export_to_mysql(self, connection_string):
        """Export final table to MySQL database."""
        engine = create_engine(connection_string)
        self.df.to_sql('user_scores', con=engine, if_exists='replace', index=False)
