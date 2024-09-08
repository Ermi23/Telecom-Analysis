# telecom_analytics.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import psycopg2
from sqlalchemy import create_engine

class TelecomExperienceAnalytics:
    def __init__(self, df, columns_to_clean):
        """
        Initialize the TelecomExperienceAnalytics class with a DataFrame and columns to clean.
        
        Args:
            df (pd.DataFrame): The dataset to be analyzed.
            columns_to_clean (list): List of column names that need cleaning.
        """
        self.df = df
        self.columns_to_clean = columns_to_clean

    def clean_data(self):
        """
        Cleans the dataset by imputing missing values with the mean.
        
        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        self.df[self.columns_to_clean] = self.df[self.columns_to_clean].fillna(self.df[self.columns_to_clean].mean())
        return self.df

    def display_head(self):
        """
        Displays the first few rows of the dataset.
        """
        if self.df is not None and not self.df.empty:
            print(self.df.head())
        else:
            print("No data available or data is empty.")

    def aggregate_per_customer(self, df):
        """
        Aggregates the dataset per customer, calculating average TCP retransmission, RTT, 
        and throughput, while also taking the most frequent handset type.
        
        Args:
            df (pd.DataFrame): The dataset to aggregate.
            
        Returns:
            pd.DataFrame: Aggregated DataFrame per customer.
        """
        agg_data = df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': lambda x: x.mode()[0]  # Mode for categorical values
        }).reset_index()

        agg_data.columns = ['MSISDN', 'Avg TCP DL Retransmission', 'Avg TCP UL Retransmission', 
                            'Avg RTT DL', 'Avg RTT UL', 'Avg Throughput DL', 'Avg Throughput UL', 'Handset Type']
        return agg_data
    
    def plot_top_10_average_values(self, agg_data, columns, id_column='MSISDN'):
        """
        Plots the top 10 average values for each network parameter using MSISDN as the customer identifier.
        
        Args:
            agg_data (pd.DataFrame): The aggregated dataset per customer.
            columns (list): List of column names (network parameters) to plot.
            id_column (str): The column name for the customer identifier (default is 'MSISDN').
        """
        for column in columns:
            # Step 1: Sort the data to get the top 10 values
            top_10 = agg_data.nlargest(10, column)
            
            # Step 2: Plot the top 10 values using a bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_10[id_column], y=top_10[column], palette="Blues_d")
            plt.title(f'Top 10 Customers (MSISDN) by Average {column}')
            plt.xlabel('MSISDN (Customer ID)')
            plt.ylabel(f'Average {column}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def top_bottom_frequent(self, df, column):
        """
        Finds the top 10, bottom 10, and most frequent values of a specified column.
        
        Args:
            df (pd.DataFrame): The dataset.
            column (str): The column name for which to find top/bottom/frequent values.
            
        Returns:
            tuple: (top_10, bottom_10, most_frequent) values for the column.
        """
        top_10 = df[column].nlargest(10)
        bottom_10 = df[column].nsmallest(10)
        most_frequent = df[column].value_counts().head(10)
        return top_10, bottom_10, most_frequent

    # def distribution_per_handset(self, df, column, agg_func='mean'):
    #     """
    #     Computes the distribution of a specified column per handset type and visualizes it.
        
    #     Args:
    #         df (pd.DataFrame): The dataset.
    #         column (str): The column name for which to compute the distribution.
    #         agg_func (str): The aggregation function ('mean' by default).
            
    #     Returns:
    #         pd.DataFrame: Distribution of the column per handset type.
    #     """
    #     dist = df.groupby('Handset Type')[column].agg(agg_func).reset_index()
    #     sns.barplot(x='Handset Type', y=column, data=dist)
    #     plt.xticks(rotation=90)
    #     plt.title(f'{agg_func.capitalize()} {column} per Handset Type')
    #     plt.show()
    #     return dist
    
    def distribution_per_handset(self, df, column, agg_func='mean'):
        """
        Computes the distribution of a specified column per handset type.
        
        Args:
            df (pd.DataFrame): The dataset.
            column (str): The column name for which to compute the distribution.
            agg_func (str): The aggregation function ('mean' by default).
            
        Returns:
            pd.DataFrame: Distribution of the column per handset type.
        """
        dist = df.groupby('Handset Type')[column].agg(agg_func).reset_index()
        return dist

    # def kmeans_clustering(self, df, n_clusters=3):
    #     """
    #     Performs K-Means clustering based on selected experience metrics.
        
    #     Args:
    #         df (pd.DataFrame): The aggregated dataset containing experience metrics.
    #         n_clusters (int): Number of clusters to use for K-Means.
            
    #     Returns:
    #         pd.DataFrame: DataFrame with cluster labels assigned.
    #     """
    #     X = df[['Avg TCP DL Retransmission', 'Avg RTT DL', 'Avg Throughput DL']]
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)

    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    #     df['Cluster'] = kmeans.fit_predict(X_scaled)
    #     return df

    def plot_throughput_distribution(self, df, column):
        """
        Plots the distribution of average throughput across handset types.

        Args:
            avg_throughput (pd.DataFrame): DataFrame containing average throughput per handset type.
        """
        avg_throughput = df.groupby('Handset Type')[column].mean().reset_index()
        plt.figure(figsize=(14, 7))
        
        # Plot distribution of average throughput
        sns.histplot(avg_throughput[column], bins=50, kde=True)
        plt.title(f'Distribution of Average {column} per Handset Type')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_top_bottom_throughput(self, df, column, top_n=10, bottom_n=10):
        """
        Plots the top and bottom N handset types based on average throughput.

        Args:
            avg_throughput (pd.DataFrame): DataFrame containing average throughput per handset type.
            top_n (int): Number of top handset types to display.
            bottom_n (int): Number of bottom handset types to display.
        """
        avg_throughput = df.groupby('Handset Type')[column].mean().reset_index()
        top_n_handsets = avg_throughput.nlargest(top_n, column)
        bottom_n_handsets = avg_throughput.nsmallest(bottom_n, column)
        
        plt.figure(figsize=(14, 7))
        
        # Plot top N handsets
        sns.barplot(x=column, y='Handset Type', data=top_n_handsets, palette='viridis', label='Top N')
        # Plot bottom N handsets
        sns.barplot(x=column, y='Handset Type', data=bottom_n_handsets, palette='magma', label='Bottom N')
        
        plt.title(f'Top {top_n} and Bottom {bottom_n} Handset Types by Average {column}')
        plt.xlabel(column)
        plt.ylabel('Handset Type')
        plt.legend()
        plt.show()

    def average_throughput_per_handset(self, df,column):
        """
        Computes the average throughput per handset type.

        Args:
            df (pd.DataFrame): The dataset containing throughput and handset type information.

        Returns:
            pd.DataFrame: Average throughput per handset type.
        """
        # Group by 'Handset Type' and calculate the mean throughput
        avg_throughput = df.groupby('Handset Type')[column].mean().reset_index()
        
        # Rename the column for clarity
        avg_throughput.rename(columns={'Avg Bearer TP DL (kbps)': 'Average Throughput (kbps)'}, inplace=True)
        
        return avg_throughput
    
    def perform_kmeans_clustering(self, df, columns, k=3):
        """
        Performs K-Means clustering on the specified columns and returns the clusters.
        
        Args:
            df (pd.DataFrame): The dataset.
            columns (list): List of column names to be used for clustering.
            k (int): Number of clusters. Default is 3.
            
        Returns:
            pd.DataFrame: DataFrame with a new 'Cluster' column indicating the assigned cluster.
            KMeans: Trained KMeans model.
        """
        # Step 1: Extract the relevant columns for clustering
        data = df[columns]
        
        # Step 2: Handle missing values by filling with the mean
        data.fillna(data.mean(), inplace=True)
        
        # Step 3: Standardize the data (z-score normalization)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Step 4: Apply K-Means clustering with k=3
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Step 5: Add the clusters to the original DataFrame
        df['Cluster'] = clusters
        
        # Step 6: Return the DataFrame with cluster labels and the trained k-means model
        return df, kmeans
    
    def perform_kmeans_clustering_with_visualization(self, df, columns, k=3, use_pca=True):
        """
        Performs K-Means clustering on the specified columns and visualizes the clusters.
        
        Args:
            df (pd.DataFrame): The dataset.
            columns (list): List of column names to be used for clustering.
            k (int): Number of clusters. Default is 3.
            use_pca (bool): Whether to apply PCA for dimensionality reduction.
            
        Returns:
            pd.DataFrame: DataFrame with a new 'Cluster' column indicating the assigned cluster.
            KMeans: Trained KMeans model.
        """
        # Step 1: Extract the relevant columns for clustering
        data = df[columns]
        
        # Step 2: Handle missing values by filling with the mean
        data.fillna(data.mean(), inplace=True)
        
        # Step 3: Standardize the data (z-score normalization)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Step 4: Apply K-Means clustering with k=3
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Step 5: Add the clusters to the original DataFrame
        df['Cluster'] = clusters
        
        # Step 6: Visualization
        if use_pca:
            # Apply PCA to reduce dimensions to 2 for visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            # Create a DataFrame for the PCA results and cluster assignments
            pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
            pca_df['Cluster'] = clusters
            
            # Plot the PCA results with cluster coloring
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set1', s=100)
            plt.title('K-Means Clustering (k=3) with PCA')
            plt.show()
        else:
            # If PCA is not used, plot based on first two features (for simplicity)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data.columns[0], y=data.columns[1], hue=clusters, palette='Set1', s=100)
            plt.title(f'K-Means Clustering (k={k}) without PCA')
            plt.xlabel(data.columns[0])
            plt.ylabel(data.columns[1])
            plt.show()
        
        # Step 7: Return the DataFrame with cluster labels and the trained k-means model
        return df, kmeans
    
    def select_metrics(self, metrics):
        """Select relevant metrics for clustering."""
        self.X = self.df[metrics]
        return self.X

    def normalize_data(self):
        """Normalize the selected metrics."""
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        return self.X_scaled
    def apply_kmeans(self, n_clusters=3):
        """Apply K-means clustering and add cluster labels to the DataFrame."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(self.X_scaled)
        return self.df

    def evaluate_clusters(self):
        """Compute silhouette score for cluster evaluation."""
        silhouette_avg = silhouette_score(self.X_scaled, self.df['Cluster'])
        return silhouette_avg

    def visualize_clusters(self):
        """Visualize the clusters."""
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=self.df['Avg Bearer TP DL (kbps)'], 
                        y=self.df['Avg RTT DL (ms)'], 
                        hue=self.df['Cluster'], 
                        palette='Set1', 
                        s=100)
        plt.title('K-means Clustering of Users Based on Network Metrics')
        plt.xlabel('Average Throughput (kbps)')
        plt.ylabel('Average RTT (ms)')
        plt.legend(title='Cluster')
        plt.grid()
        plt.show()

    def cluster_descriptions(self, metrics):
        """Provide descriptive statistics for each cluster."""
        descriptions = {}
        for i in range(self.df['Cluster'].nunique()):
            descriptions[i] = self.df[self.df['Cluster'] == i][metrics].describe()
        return descriptions
