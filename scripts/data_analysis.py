# data_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set display options to show more rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

class DataAnalysis:
    def __init__(self, dataframe):
        self.dataframe = dataframe

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

    # def data_types(self):
    #     """Display data types of the DataFrame columns."""
    #     if self.dataframe is not None:
    #         print("DataFrame Data Types:")
    #         print(self.dataframe.dtypes)
    #     else:
    #         print("No DataFrame to analyze.")

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

    def plot_missing_values_heatmap(self):
        """Plot a heatmap of missing values in the DataFrame."""
        if self.dataframe is not None:
            # Convert common null representations to NaN
            self.dataframe.replace(["NULL", "N/A", "", " "], pd.NA, inplace=True)
            
            # Create a boolean DataFrame where True represents missing values
            missing_data = self.dataframe.isna()
            
            # Plot the heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(missing_data, cbar=False, cmap='viridis', yticklabels=False, xticklabels=True)
            plt.title('Heatmap of Missing Values')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.show()
        else:
            print("No DataFrame to plot.")

    def get_top_10_handsets(self):
        """Extract the 'Handset Type' column, count occurrences, sort, and return top 10."""
        # Check if 'Handset Type' column exists
        if 'Handset Type' not in self.dataframe.columns:
            print("The DataFrame does not contain a 'Handset Type' column.")
            return None
        
        # Step 1: Extract the 'Handset Type' column
        handset_data = self.dataframe['Handset Type']

        # Step 2: Count the occurrences of each handset type
        handset_counts = handset_data.value_counts()

        # Step 3: Sort the counts in descending order and select the top 10
        top_10_handsets = handset_counts.head(10)

        # Step 4: Print the top 10 handsets
        print("Top 10 Handsets:")
        print(top_10_handsets)

        return top_10_handsets

    def plot_top_10_handsets(self, top_10_handsets):
        """Plot a bar chart of the top 10 handsets."""
        # Step 5: Visualize the top 10 handsets using a bar plot
        plt.figure(figsize=(10, 6))
        top_10_handsets.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Handsets by Count')
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def get_top_3_manufacturers(self):
        """Count occurrences of each handset manufacturer and return the top 3."""
        # Check if 'Handset Type' column exists
        if 'Handset Type' not in self.dataframe.columns:
            print("The DataFrame does not contain a 'Handset Type' column.")
            return None
        
        # Step 1: Extract the manufacturer from the 'Handset Type' column
        # Handle cases where 'Handset Type' might be None or NaN
        self.dataframe['Manufacturer'] = self.dataframe['Handset Type'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)
        
        # Step 2: Drop rows with None in 'Manufacturer'
        self.dataframe = self.dataframe.dropna(subset=['Manufacturer'])
        
        # Step 3: Count the occurrences of each manufacturer
        manufacturer_counts = self.dataframe['Manufacturer'].value_counts()

        # Step 4: Sort the counts in descending order and select the top 3 manufacturers
        top_3_manufacturers = manufacturer_counts.head(3)

        # Step 5: Print the top 3 manufacturers
        print("Top 3 Handset Manufacturers:")
        print(top_3_manufacturers)

        return top_3_manufacturers

    def plot_top_3_manufacturers(self, top_3_manufacturers):
        """Plot a bar chart of the top 3 handset manufacturers."""
        if top_3_manufacturers is None or top_3_manufacturers.empty:
            print("No data available for plotting.")
            return
        
        # Visualize the top 3 manufacturers using a bar plot
        plt.figure(figsize=(8, 5))
        top_3_manufacturers.plot(kind='bar', color='skyblue')
        plt.title('Top 3 Handset Manufacturers by Count')
        plt.xlabel('Manufacturer')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def get_top_5_handsets_per_manufacturer(self):
        """Get top 5 handsets for each of the top 3 manufacturers."""
        # Get the top 3 manufacturers first
        top_3_manufacturers = self.get_top_3_manufacturers()

        if top_3_manufacturers is None:
            print("No manufacturers found.")
            return None

        # Dictionary to hold the top 5 handsets per manufacturer
        top_5_handsets_per_manufacturer = {}

        # Step 1: Loop through each top manufacturer
        for manufacturer in top_3_manufacturers.index:
            # Filter the DataFrame for handsets belonging to this manufacturer
            manufacturer_handsets = self.dataframe[self.dataframe['Manufacturer'] == manufacturer]
            
            # Step 2: Count occurrences of each handset type for this manufacturer
            handset_counts = manufacturer_handsets['Handset Type'].value_counts()

            # Step 3: Get the top 5 handsets for this manufacturer
            top_5_handsets = handset_counts.head(5)
            
            # Add the top 5 handsets to the dictionary
            top_5_handsets_per_manufacturer[manufacturer] = top_5_handsets

        # Step 4: Print and return the top 5 handsets for each top 3 manufacturer
        for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
            print(f"\nTop 5 handsets for {manufacturer}:")
            print(handsets)
        
        return top_5_handsets_per_manufacturer
    
    def visualize_top_5_handsets(self):
        """Visualize the top 5 handsets per manufacturer."""
        # Get top 5 handsets per manufacturer
        top_5_handsets_per_manufacturer = self.get_top_5_handsets_per_manufacturer()

        if top_5_handsets_per_manufacturer is None:
            return
        
        # Step 1: Setup for visualization
        num_manufacturers = len(top_5_handsets_per_manufacturer)
        fig, axes = plt.subplots(1, num_manufacturers, figsize=(16, 5), sharey=True)

        if num_manufacturers == 1:
            axes = [axes]  # Ensure axes is always a list

        # Step 2: Plot each manufacturer’s top 5 handsets
        for i, (manufacturer, handsets) in enumerate(top_5_handsets_per_manufacturer.items()):
            ax = axes[i]
            handsets.plot(kind='barh', ax=ax, color='teal')
            ax.set_title(f"Top 5 Handsets: {manufacturer}")
            ax.set_xlabel("Number of Handsets")
            ax.set_ylabel("Handset Type")

        # Step 3: Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def calculate_session_metrics(self):
        """Calculate xDR session counts, session duration, and total DL/UL data."""
        # Check for necessary columns
        required_columns = ['Start', 'Start ms', 'End', 'End ms', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']
        for col in required_columns:
            if col not in self.dataframe.columns:
                print(f"Required column {col} is missing in the DataFrame.")
                return None

        # Step 1: Count unique xDR sessions (assuming each row represents a unique session)
        num_sessions = len(self.dataframe)

        # Step 2: Calculate total session duration (Dur. (ms) column already provides the duration in seconds)
        total_session_duration_ms = self.dataframe['Dur. (ms)'].sum()
        total_session_duration_s = total_session_duration_ms / 1000  # Convert ms to seconds
        total_session_duration_h = total_session_duration_s / 3600  # Convert seconds to hours

        # Step 3: Calculate total download (DL) and upload (UL) data
        total_download_data_bytes = self.dataframe['Total DL (Bytes)'].sum()
        total_upload_data_bytes = self.dataframe['Total UL (Bytes)'].sum()
        total_download_data_mb = total_download_data_bytes / (1024 * 1024)  # Convert Bytes to MegaBytes
        total_upload_data_mb = total_upload_data_bytes / (1024 * 1024)  # Convert Bytes to MegaBytes
        total_download_data_gb = total_download_data_mb / 1024  # Convert MegaBytes to GigaBytes
        total_upload_data_gb = total_upload_data_mb / 1024  # Convert MegaBytes to GigaBytes

        # Round to two decimal places
        total_session_duration_h = round(total_session_duration_h, 2)
        total_download_data_gb = round(total_download_data_gb, 2)
        total_upload_data_gb = round(total_upload_data_gb, 2)

        # Aggregate metrics
        session_metrics = {
            'Number of Sessions': num_sessions,
            'Total Session Duration (hours)': total_session_duration_h,
            'Total Download Data (GB)': total_download_data_gb,
            'Total Upload Data (GB)': total_upload_data_gb
        }

        # Format the output as a string with new lines
        formatted_metrics = (
            f"Number of Sessions: {session_metrics['Number of Sessions']}\n"
            f"Total Session Duration (hours): {session_metrics['Total Session Duration (hours)']}\n"
            f"Total Download Data (GB): {session_metrics['Total Download Data (GB)']}\n"
            f"Total Upload Data (GB): {session_metrics['Total Upload Data (GB)']}"
        )

        return formatted_metrics
    
    def visualize_dl_ul_data(self):
        """Visualize the download and upload data."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.dataframe['Total DL (Bytes)'] / (1024 * 1024 * 1024), label='Download Data (GB)')
        plt.plot(self.dataframe['Total UL (Bytes)'] / (1024 * 1024 * 1024), label='Upload Data (GB)')
        plt.xlabel('Session Index')
        plt.ylabel('Data (GB)')
        plt.title('Download and Upload Data per Session')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # def visualize_dl_ul_data(dataframe):
    #     """Visualize total DL and UL data per session."""
    #     # Plot total download and upload data
    #     dataframe[['Total DL (Bytes)', 'Total UL (Bytes)']].plot(kind='bar', figsize=(10, 6))
    #     plt.title('Total Download and Upload Data per Session')
    #     plt.xlabel('Session Index')
    #     plt.ylabel('Data (Bytes)')
    #     plt.xticks(rotation=0)
    #     plt.legend(['Download (Bytes)', 'Upload (Bytes)'])
    #     plt.show()
            
    def filter_relevant_columns(self):
        """Filter the DataFrame to keep only relevant columns."""
        relevant_columns = ['Dur. (ms)', 'MSISDN/Number', 'Total DL (Bytes)', 'Total UL (Bytes)', 
                            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                            'Google DL (Bytes)', 'Google UL (Bytes)',
                            'Email DL (Bytes)', 'Email UL (Bytes)',
                            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                            'Other DL (Bytes)', 'Other UL (Bytes)']
        return self.dataframe[relevant_columns]

    def aggregate_per_user(self, filtered_df):
        """Aggregate metrics per user (MSISDN/Number)."""
        # Group by 'MSISDN/Number'
        grouped_df = filtered_df.groupby('MSISDN/Number')

        # Aggregate the data
        agg_df = grouped_df.agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum',
            'Social Media DL (Bytes)': 'sum',
            'Social Media UL (Bytes)': 'sum',
            'Google DL (Bytes)': 'sum',
            'Google UL (Bytes)': 'sum',
            'Email DL (Bytes)': 'sum',
            'Email UL (Bytes)': 'sum',
            'Youtube DL (Bytes)': 'sum',
            'Youtube UL (Bytes)': 'sum',
            'Netflix DL (Bytes)': 'sum',
            'Netflix UL (Bytes)': 'sum',
            'Gaming DL (Bytes)': 'sum',
            'Gaming UL (Bytes)': 'sum',
            'Other DL (Bytes)': 'sum',
            'Other UL (Bytes)': 'sum',
        })

        # Calculate number of xDR sessions and session duration
        agg_df['Number of xDR sessions'] = grouped_df.size()
        agg_df['Session duration (s)'] = grouped_df['Dur. (ms)'].sum() / 1000  # Convert ms to seconds

        return agg_df.reset_index()

    def analyze_xdr_sessions(self, overview_df):
        """Analyze the number of xDR sessions per user."""
        # Extract the 'Number of xDR sessions' column
        xdr_sessions = overview_df['Number of xDR sessions']

        # Return descriptive statistics
        return xdr_sessions.describe()

    def visualize_data_volume(self):
        """Visualize total data volume per application/service."""
        data_volume_per_application = self.dataframe[['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
                                                      'Google DL (Bytes)', 'Google UL (Bytes)', 
                                                      'Email DL (Bytes)', 'Email UL (Bytes)', 
                                                      'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
                                                      'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
                                                      'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
                                                      'Other DL (Bytes)', 'Other UL (Bytes)']].sum()

        plt.figure(figsize=(10, 6))
        data_volume_per_application.plot(kind='bar', color='lightgreen')
        plt.title('Total Data Volume per Application/Service')
        plt.xlabel('Application/Service')
        plt.ylabel('Data Volume (Bytes)')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def visualize_upload_download_per_application(self):
        """Visualize upload and download data per application in a single plot."""
        # Calculate total download (DL) and upload (UL) per application
        download_data = self.dataframe[['Social Media DL (Bytes)', 'Google DL (Bytes)', 
                                         'Email DL (Bytes)', 'Youtube DL (Bytes)', 
                                         'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 
                                         'Other DL (Bytes)']].sum()

        upload_data = self.dataframe[['Social Media UL (Bytes)', 'Google UL (Bytes)', 
                                       'Email UL (Bytes)', 'Youtube UL (Bytes)', 
                                       'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 
                                       'Other UL (Bytes)']].sum()
        
        # Convert Bytes to GigaBytes for better readability
        download_data_gb = download_data / (1024 * 1024 * 1024)
        upload_data_gb = upload_data / (1024 * 1024 * 1024)

        # Print the numerical values in tabular format
        print("| Source               | Download Data (Bytes) | Download Data (GB) |")
        print("|----------------------|-----------------------|---------------------|")
        for app in download_data.index:
            print(f"| {app:<20} | {download_data[app]:<21.2f} | {download_data_gb[app]:<19.2f} |")

                # Print the numerical values in tabular format
        print("| Source               | Download Data (Bytes) | Download Data (GB) |")
        print("|----------------------|-----------------------|---------------------|")
        for app in upload_data.index:
            print(f"| {app:<20} | {upload_data[app]:<21.2f} | {upload_data_gb[app]:<19.2f} |")

        # Create a DataFrame for plotting
        data = pd.DataFrame({
            'Download Data (Bytes)': download_data,
            'Upload Data (Bytes)': upload_data
        })

                # Plot download data
        plt.figure(figsize=(10, 6))
        download_data.plot(kind='bar', color='lightblue')
        plt.title('Total Download Data per Application')
        plt.xlabel('Application')
        plt.ylabel('Download Data (Bytes)')
        plt.xticks(rotation=45, ha='right')
        plt.show()

        # Plot upload data
        plt.figure(figsize=(10, 6))
        upload_data.plot(kind='bar', color='lightcoral')
        plt.title('Total Upload Data per Application')
        plt.xlabel('Application')
        plt.ylabel('Upload Data (Bytes)')
        plt.xticks(rotation=45, ha='right')
        plt.show()

        # Plot the upload and download correlation
        plt.figure(figsize=(14, 8))  # Make the figure wider
        data.plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.title('Total Download and Upload Data per Application')
        plt.xlabel('Application')
        plt.ylabel('Data (Bytes)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
