import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    data = pd.read_excel('data/data_source.xlsx')
    return data

# Main function to run the dashboard
def main():
    st.title("Data Insights Dashboard")
    
    # Load data
    data = load_data()
    
    # Sidebar for user input
    st.sidebar.header("User Input Features")
    selected_manufacturer = st.sidebar.selectbox("Select Handset Manufacturer", data['Handset Manufacturer'].unique())
    
    # Filter data based on selected manufacturer
    filtered_data = data[data['Handset Manufacturer'] == selected_manufacturer]
    
    # Display basic information
    st.subheader("Data Overview")
    st.write(filtered_data.describe())
    
    # KPIs
    st.subheader("Key Performance Indicators (KPIs)")
    st.metric("Total Sessions", len(filtered_data))
    
    # Visualizations
    st.subheader("Visualizations")
    
    # Plot Total DL and UL Bytes
    fig, ax = plt.subplots()
    filtered_data[['Total DL (Bytes)', 'Total UL (Bytes)']].plot(kind='bar', ax=ax)
    plt.title('Total Download and Upload Bytes')
    plt.xlabel('Sessions')
    plt.ylabel('Bytes')
    st.pyplot(fig)

    # Additional visualizations can be added here

if __name__ == "__main__":
    main()