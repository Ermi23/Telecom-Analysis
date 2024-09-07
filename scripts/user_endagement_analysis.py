# data_analysis.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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