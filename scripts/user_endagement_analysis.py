# data_analysis.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set display options to show more rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

class UserEngagementAnalysis:
    def __init__(self, dataframe, applications):
        self.dataframe = dataframe
        self.app = applications

    def info(self):
        """Display information about the DataFrame."""
        if self.dataframe is not None:
            print("DataFrame Info:")
            self.dataframe.info()
        else:
            print("No DataFrame to analyze.")

    def shape(self):
        """Return the shape of the DataFrame."""
        if self.dataframe is not None:
            return self.dataframe.shape
        else:
            print("No DataFrame to analyze.")
            return None

    def columns(self):
        """Return the columns of the DataFrame."""
        if self.dataframe is not None:
            return self.dataframe.columns
        else:
            print("No DataFrame to analyze.")
            return None

    def describe(self):
        """Display statistics of the DataFrame."""
        if self.dataframe is not None:
            try:
                print("DataFrame Description:")
                print(self.dataframe.describe())
            except Exception as e:
                print(f"Error describing DataFrame: {e}")
        else:
            print("No DataFrame to analyze.")

    def check_missing_values(self, print_output=True):
        """Check for missing values in the DataFrame, converting common null representations to NaN."""
        if self.dataframe is not None:
            # Convert common null representations to NaN
            self.dataframe.replace(["NULL", "N/A", "", " "], pd.NA, inplace=True)
            
            # Calculate missing values
            total_rows = len(self.dataframe)
            missing_values = self.dataframe.isnull().sum()
            missing_percent = (missing_values / total_rows) * 100  # Calculate percentage of missing values

            # Prepare a DataFrame with missing values and percentages
            missing_summary = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage (%)': (missing_percent.round(2).astype(str) + '%')
            })
            
            # Remove columns with 0 missing values
            missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
            
            if print_output:
                if not missing_summary.empty:
                    print("Missing Values in DataFrame (columns with missing data):")
                    print(missing_summary)
                else:
                    print("No missing values in the DataFrame.")
            
            return missing_summary
        else:
            print("No DataFrame to analyze.")
            return None

    def fill_missing_with_mean(self):
        """Fill missing values in numeric columns with the mean of those columns."""
        if self.dataframe is not None:
            # Select only numeric columns
            numeric_columns = self.dataframe.select_dtypes(include=['number']).columns
            
            # Fill missing values in numeric columns with the mean of each column
            for col in numeric_columns:
                if self.dataframe[col].isnull().sum() > 0:
                    self.dataframe[col].fillna(self.dataframe[col].mean(), inplace=True)
                    print(f"Filled missing values in column: {col}")
            # print("Missing values have been filled with the mean for numeric columns.")
        else:
            print("No DataFrame to analyze.")
            
    def inspect_null_representations(self):
        """Inspect if any columns contain 'null-like' values such as 'NULL', 'N/A', or empty strings."""
        if self.dataframe is not None:
            null_reps = ["NULL", "N/A", "", " "]
            for col in self.dataframe.columns:
                for null_rep in null_reps:
                    count = (self.dataframe[col] == null_rep).sum()
                    if count > 0:
                        print(f"Column '{col}' contains {count} '{null_rep}' values.")
        else:
            print("No DataFrame to inspect.")

    def aggregate_data(self):
        """Calculate total session durations and aggregate data."""
        # Assuming 'grouped_dataframe' is defined elsewhere in your code
        self.agg_dataframe = self.dataframe.groupby('MSISDN/Number').agg({
            'Dur. (ms)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()

        # Calculate the total duration for all sessions for each user
        self.agg_dataframe['Total Session Duration'] = self.agg_dataframe['Dur. (ms)'] / 1000  # Convert milliseconds to seconds

        # Compute the total data (DL+UL) for each user
        self.agg_dataframe['Total Data (DL+UL)'] = self.agg_dataframe['Total DL (Bytes)'] + self.agg_dataframe['Total UL (Bytes)']

    def segment_deciles(self):
        """Segment users into deciles based on total session duration."""
        self.agg_dataframe['Decile Class'] = pd.qcut(self.agg_dataframe['Total Session Duration'], q=10, labels=False)

    def analyze_deciles(self):
        """Analyze and print total data per decile class."""
        total_data_per_decile = self.agg_dataframe.groupby('Decile Class')['Total Data (DL+UL)'].sum()
        print("Total Data (DL+UL in Gigabytes) per Decile Class:")
        print(total_data_per_decile / 1073741824)

        # Sort and get top 5 deciles
        top_5_deciles = total_data_per_decile.sort_values(ascending=False).head(5)
        print("\nTop 5 Deciles based on Total Data (DL+UL):")
        print(top_5_deciles / 1073741824)
        
        return top_5_deciles

    def average_session_duration(self):
        """Calculate and print average session duration for each decile class."""
        avg_session_duration_per_decile = self.agg_dataframe.groupby('Decile Class')['Total Session Duration'].mean()
        top_5_highest_duration_deciles = avg_session_duration_per_decile.sort_values(ascending=False).head(5)

        print("Top 5 Highest Duration Decile Classes and Their Total Duration:")
        for decile_class, avg_duration in top_5_highest_duration_deciles.items():
            total_duration = self.agg_dataframe[self.agg_dataframe['Decile Class'] == decile_class]['Total Session Duration'].sum()
            print(f"Decile Class: {decile_class}, Total Duration: {total_duration} seconds")

    def plot_data_usage(self):
        """Plot the data usage against session duration."""
        self.dataframe['Total UL + DL'] = self.dataframe['Total UL (Bytes)'] + self.dataframe['Total DL (Bytes)']

        plt.figure(figsize=(10, 6))
        plt.scatter(self.dataframe['Dur. (ms)'] / 1000, self.dataframe['Total UL + DL'], alpha=0.5)
        plt.title('Duration of xDR Sessions vs Total UL + DL Usage')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Total UL + DL Usage (Bytes)')
        plt.grid(True)
        plt.show()

    def calculate_total_ul_dl(self):
        """Calculate Total UL + DL Usage."""
        self.dataframe['Total UL + DL'] = self.dataframe['Total UL (Bytes)'] + self.dataframe['Total DL (Bytes)']

    def plot_hexbin(self):
        """Plot a hexbin plot of session duration vs. total UL + DL usage."""
        self.calculate_total_ul_dl()  # Ensure that Total UL + DL is calculated
        plt.figure(figsize=(10, 6))
        plt.hexbin(self.dataframe['Dur. (ms)'], self.dataframe['Total UL + DL'], gridsize=50, cmap='viridis')
        plt.colorbar(label='Count in bin')
        plt.title('Hexbin Plot: Session Duration vs. Total UL + DL Usage')
        plt.xlabel('Session Duration (ms)')
        plt.ylabel('Total UL + DL Usage (Bytes)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_session_duration_distribution(self):
        """Analyze and plot session duration distribution."""
        session_duration_distribution = self.dataframe['Dur. (ms)'] / 1000  # Convert milliseconds to seconds
        plt.figure(figsize=(8, 6))
        plt.hist(session_duration_distribution, bins=30, color='skyblue', edgecolor='black')
        plt.title('Session Duration Distribution')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.show()

    def analyze_network_performance(self):
        """Analyze and print network performance metrics."""
        network_performance_metrics = self.dataframe[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
                                                'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
                                                'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].describe()
        print("Network Performance Metrics:")
        print(network_performance_metrics)

    def calculate_session_duration_statistics(self):
        """Calculate and print mean and median session duration."""
        self.dataframe['Session Duration (seconds)'] = self.dataframe['Dur. (ms)'] / 1000
        session_duration_mean = self.dataframe['Session Duration (seconds)'].mean()
        session_duration_median = self.dataframe['Session Duration (seconds)'].median()

        print("Mean Session Duration:", session_duration_mean, "seconds")
        print("Median Session Duration:", session_duration_median, "seconds")

    def preprocess_data(self):
        """Identify datetime columns and convert them to numeric."""
        datetime_columns = self.dataframe.select_dtypes(include=['datetime64']).columns
        print("Datetime columns identified:", datetime_columns)

        # Remove non-numeric columns and convert datetime columns to numeric
        self.numeric_dataframe = self.dataframe.select_dtypes(include=['number'])
        for column in datetime_columns:
            self.numeric_dataframe[column] = pd.to_numeric(self.dataframe[column])

        print("Numeric DataFrame:\n", self.numeric_dataframe.head())

    def perform_pca(self):
        """Perform PCA on the numeric DataFrame and calculate explained variance."""
        self.pca_model = PCA()
        self.pca_model.fit(self.numeric_dataframe)

        self.explained_variance_ratio = self.pca_model.explained_variance_ratio_
        self.cumulative_variance_explained = self.pca_model.explained_variance_ratio_.cumsum()

    def plot_explained_variance(self):
        """Plot the explained variance ratio for each principal component."""
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.explained_variance_ratio) + 1), self.explained_variance_ratio, alpha=0.7, align='center')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Component')
        plt.show()

    def plot_cumulative_variance(self):
        """Plot the cumulative variance explained by principal components."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.cumulative_variance_explained) + 1), self.cumulative_variance_explained, marker='o', linestyle='-')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Cumulative Variance Explained by Principal Components')
        plt.grid(True)
        plt.show()
    
    def determine_optimal_components(self, threshold=0.8):
        """Determine the optimal number of components to explain a significant portion of variance."""
        num_components = len(self.cumulative_variance_explained[self.cumulative_variance_explained <= threshold])
        print("Number of Components to Explain 80% Variance:", num_components)
        return num_components

    def analyze_loadings(self):
        """Analyze the loadings of each feature on the principal components."""
        principal_components = self.pca_model.components_
        print("Principal Components shape:", principal_components.shape)
        return principal_components

    def aggregate_metric_analyze(self):
        """Aggregate user engagement metrics and analyze them."""
        # Implement your aggregation logic here
        # Example: Calculate session frequency, duration, and total data traffic
        # This is a placeholder for actual implementation
        print("Aggregating user engagement metrics...")
        return self.dataframe[['Session Duration (seconds)', 'Dur. (ms)', 'Total UL + DL']].describe()

    def normalize_and_cluster(self):
        """Normalize user engagement metrics and perform clustering."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.numeric_dataframe)

        kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as necessary
        clusters = kmeans.fit_predict(scaled_data)

        self.dataframe['Cluster'] = clusters
        return self.dataframe

    def plot_scatter_cluster(self, features):
        """Plot a scatter plot for clustered data."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.dataframe[features[0]], self.dataframe[features[1]], c=self.dataframe['Cluster'], cmap='viridis', alpha=0.5)
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('Scatter Plot of Clusters')
        plt.grid(True)
        plt.show()

    def visualize_distributions(self):
        """Visualize the distributions of clusters using violin plots."""
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Cluster', y='Total UL + DL', data=self.dataframe)
        plt.title('Cluster Distribution of Total UL + DL')
        plt.show()

    def analyze_clusters(self):
        """Analyze clusters and get descriptive statistics."""
        cluster_data = self.dataframe.groupby('Cluster')
        stats = cluster_data[['Dur. (ms)', 'Total UL + DL', 'Session Duration (seconds)']].agg(['min', 'max', 'mean', 'sum'])
        return stats

    def plot_bar_cluster_stats(self, stats, feature, stat_to_plot):
        """Plot bar chart for cluster statistics."""
        plt.figure(figsize=(10, 6))
        stats[feature].plot(kind='bar', title=f'{stat_to_plot.capitalize()} of {feature} by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(stat_to_plot.capitalize())
        plt.show()

    def plot_pie_cluster_stats(self, stats, feature, stat_to_plot):
        """Plot pie chart for cluster statistics."""
        plt.figure(figsize=(8, 8))
        # Extract the required statistic from the stats DataFrame
        pie_data = stats[feature][stat_to_plot]  # Get the mean, min, max, etc. for the feature
        pie_data.plot(kind='pie', title=f'{stat_to_plot.capitalize()} of {feature} by Cluster', autopct='%1.1f%%')
        plt.ylabel('')
        plt.show()

    def convert_units(self):
        """Convert all relevant metrics to consistent units (e.g., seconds for duration)."""
        if 'Dur. (ms)' in self.dataframe.columns:
            self.dataframe['Dur. (seconds)'] = self.dataframe['Dur. (ms)'] / 1000  # Convert milliseconds to seconds

class NeededMethodsForUserEngagementAnalysis:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.agg_dataframe = None

    def info(self):
        """Display information about the DataFrame."""
        if self.dataframe is not None:
            print("DataFrame Info:")
            self.dataframe.info()
        else:
            print("No DataFrame to analyze.")

    def aggregate_data(self):
        """Calculate total session durations and aggregate data."""
        self.agg_dataframe = self.dataframe.groupby('MSISDN/Number').agg({
            'Dur. (ms)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()
        
        self.agg_dataframe['Total Session Duration'] = self.agg_dataframe['Dur. (ms)'] / 1000  # Convert to seconds
        self.agg_dataframe['Total Data (DL+UL)'] = self.agg_dataframe['Total DL (Bytes)'] + self.agg_dataframe['Total UL (Bytes)']

    def top_customers_by_metric(self, metric, top_n=10):
        """Return top N customers by a specified engagement metric."""
        if self.agg_dataframe is not None and metric in self.agg_dataframe.columns:
            top_customers = self.agg_dataframe.nlargest(top_n, metric)
            print(f"Top {top_n} customers by {metric}:")
            print(top_customers[['MSISDN/Number', metric]])
            return top_customers
        else:
            print("No aggregated DataFrame to analyze or invalid metric.")
            return None

    def normalize_and_cluster(self):
        """Normalize user engagement metrics and perform clustering."""
        scaler = StandardScaler()
        self.numeric_dataframe = self.agg_dataframe[['Total Session Duration', 'Total Data (DL+UL)']]
        scaled_data = scaler.fit_transform(self.numeric_dataframe)

        kmeans = KMeans(n_clusters=3)  # Initial number of clusters
        clusters = kmeans.fit_predict(scaled_data)

        self.agg_dataframe['Cluster'] = clusters

    def analyze_clusters(self):
        """Analyze clusters and get descriptive statistics."""
        if self.agg_dataframe is not None:
            cluster_data = self.agg_dataframe.groupby('Cluster')
            stats = cluster_data[['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 
                                  'Total Session Duration', 'Total Data (DL+UL)']].agg(['min', 'max', 'mean', 'sum'])
            return stats
        else:
            print("No aggregated DataFrame to analyze.")
            return None

    def plot_cluster_stats(self, stats, feature):
        """Plot bar chart for cluster statistics."""
        plt.figure(figsize=(10, 6))
        stats[feature].plot(kind='bar', title=f'Statistics of {feature} by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(feature)
        plt.show()

    def aggregate_traffic_per_application(self):
        """Aggregate user total traffic per application."""
        app_columns = ['HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Social Media DL (Bytes)', 
                       'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                        'Google DL (Bytes)', 'Google UL (Bytes)',
                        'Email DL (Bytes)', 'Email UL (Bytes)',
                        'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                        'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                        'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                        'Other DL (Bytes)', 'Other UL (Bytes)']
        
        if 'MSISDN/Number' in self.dataframe.columns:
            self.app_traffic = self.dataframe.groupby('MSISDN/Number')[app_columns].sum().reset_index()
            self.app_traffic['Total Traffic'] = self.app_traffic[app_columns].sum(axis=1)
            return self.app_traffic.nlargest(10, 'Total Traffic')
        else:
            print("No DataFrame to aggregate application traffic.")
            return None

    def plot_top_applications(self, top_n=3):
        """Plot the top N most used applications."""
        app_columns = ['HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Social Media DL (Bytes)', 
                'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                'Google DL (Bytes)', 'Google UL (Bytes)',
                'Email DL (Bytes)', 'Email UL (Bytes)',
                'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                'Other DL (Bytes)', 'Other UL (Bytes)']
        if hasattr(self, 'app_traffic'):
            top_apps = self.app_traffic.nlargest(top_n, 'Total Traffic')
            top_apps.set_index('MSISDN/Number')[app_columns].plot(kind='bar', figsize=(12, 6))
            plt.title(f'Top {top_n} Applications Used by Users')
            plt.xlabel('User (MSISDN)')
            plt.ylabel('Total Traffic (Bytes)')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print("No application traffic data to plot.")

    def elbow_method(self, max_k=10):
        """Determine the optimal number of clusters using the elbow method."""
        if self.numeric_dataframe is not None:
            inertia = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(self.numeric_dataframe)
                inertia.append(kmeans.inertia_)
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_k + 1), inertia, marker='o')
            plt.title('Elbow Method for Optimal k')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.grid(True)
            plt.show()
        else:
            print("Numeric DataFrame is not available for elbow method.")

    def interpret_clusters(self, stats):
        """Interpret the clustering results and provide insights."""
        if stats is not None:
            for cluster in stats.index:
                print(f"Cluster {cluster}:")
                print(f"  Min Total Session Duration: {stats.loc[cluster, ('Total Session Duration', 'min')]:.2f} seconds")
                print(f"  Max Total Session Duration: {stats.loc[cluster, ('Total Session Duration', 'max')]:.2f} seconds")
                print(f"  Avg Total Session Duration: {stats.loc[cluster, ('Total Session Duration', 'mean')]:.2f} seconds")
                print(f"  Total Data Usage: {stats.loc[cluster, ('Total Data (DL+UL)', 'sum')]} Bytes")
                print("")