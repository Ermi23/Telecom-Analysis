# Data Analysis Project

This repository contains two Python scripts, `data_analysis.py` and `user_engagement_analysis.py`, designed for comprehensive data analysis, focusing on user engagement metrics and data volume analysis. The scripts utilize popular libraries such as Pandas, NumPy, Seaborn, and Matplotlib for data manipulation and visualization.

## Table of Contents
- Overview
- Requirements
- Usage
- Functions
- Example
- License

## Overview
The purpose of these scripts is to analyze user engagement data, specifically focusing on:
- Understanding the structure and statistics of the data.
- Identifying and handling missing values.
- Visualizing user engagement metrics through various plots.
- Aggregating data per user and application.
- Calculating session metrics and data volumes for different applications.

## Requirements
To run these scripts, you'll need to have the following libraries installed:
- Python 3.6 or later
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn (for PCA analysis)

You can install the necessary packages using pip:
- bash
 pip install pandas numpy matplotlib seaborn scikit-learn

# Usage
## Load Your Data
You need to load your dataset into a Pandas DataFrame. Ensure that the dataset includes columns relevant to user engagement metrics.

## Initialize the DataAnalysis Class
Create an instance of the DataAnalysis class by passing your DataFrame and a list of applications to analyze.

import pandas as pd
from data_analysis import DataAnalysis

# Load your dataset
you can fine the database schema on data/database schema folder

# Initialize the analysis
analysis = DataAnalysis(dataframe=df, applications=['Social Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Other'])

# Perform Analysis
Use the various methods provided in the DataAnalysis class to perform different analyses.

# Visualize Results
The scripts provide multiple visualization methods to help you interpret the data effectively.

# Functions
## DataAnalysis Class
__init__(self, dataframe, applications): Initializes the class with a DataFrame and a list of applications.
info(): Displays information about the DataFrame.
shape(): Returns the shape of the DataFrame.
columns(): Returns the columns of the DataFrame.
describe(): Displays statistics of the DataFrame.
check_missing_values(print_output=True): Checks for missing values in the DataFrame.
fill_missing_with_mean(): Fills missing values in numeric columns with the mean of those columns.
plot_missing_values_heatmap(): Plots a heatmap of missing values.
get_top_10_handsets(): Extracts and returns the top 10 handsets based on usage.
plot_top_10_handsets(top_10_handsets): Plots a bar chart of the top 10 handsets.
calculate_session_metrics(): Calculates session metrics such as session counts and total data.
visualize_data_volume(): Visualizes total data volume per application.
User Engagement Analysis Functions
Functions for aggregating data per user, calculating total data volumes, and visualizing engagement metrics.
