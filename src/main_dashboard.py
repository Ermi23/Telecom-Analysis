import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_excel('data/data_source.xlsx')
    
    # Convert 'Start' and 'End' columns to datetime
    data['Start'] = pd.to_datetime(data['Start'], format='%m/%d/%Y %H:%M', errors='coerce')
    data['End'] = pd.to_datetime(data['End'], format='%m/%d/%Y %H:%M', errors='coerce')
    
    return data

# Main function to run the dashboard
def main():
    st.title("Data Insights Dashboard")
    
    # Load data
    data = load_data()
    
    # Sidebar for date input
    st.sidebar.header("Filter by Date")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-09-09'))

    # Filter data based on selected date range
    filtered_data = data[(data['Start'] >= pd.to_datetime(start_date)) & (data['Start'] <= pd.to_datetime(end_date))]

    # Display key metrics
    st.subheader("Key Data Points")
    
    # User Engagement Metrics
    total_active_users = filtered_data['MSISDN/Number'].nunique()
    avg_session_duration = filtered_data['Dur. (ms)'].mean()
    st.metric("Total Active Users", total_active_users)
    st.metric("Average Session Duration (s)", round(avg_session_duration / 1000, 2))  # Convert to seconds
    
    # Data Usage Patterns
    st.subheader("Data Usage Patterns in GB")
    total_data_DL_volume = filtered_data['Total DL (Bytes)'].sum() / (1024 * 1024 * 1024)
    st.metric("Total Download Data Volume (Gb)", total_data_DL_volume)
    total_Upload_data_volume = filtered_data['Total UL (Bytes)'].sum() / (1024 * 1024 * 1024)
    st.metric("Total Upload Data Volume (Gb)", total_Upload_data_volume)
    total_data_volume = (filtered_data['Total DL (Bytes)'].sum() + filtered_data['Total UL (Bytes)'].sum()) / (1024 * 1024 * 1024)
    st.metric("Total Data Volume (Gb)", total_data_volume)

    # Data Usage by Category
    st.subheader("Data Usage by Category in Bytes")
    usage_categories = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    data_usage_by_category = filtered_data[usage_categories].sum()
    st.bar_chart(data_usage_by_category)

    # Average Bearer Throughput
    st.subheader("Average Bearer Throughput in Kbps")
    avg_bearer_tp_dl = filtered_data['Avg Bearer TP DL (kbps)'].mean()
    avg_bearer_tp_ul = filtered_data['Avg Bearer TP UL (kbps)'].mean()
    st.metric("Average Bearer Throughput DL (kbps)", round(avg_bearer_tp_dl, 2))
    st.metric("Average Bearer Throughput UL (kbps)", round(avg_bearer_tp_ul, 2))

    # Quality of Service Metrics
    st.subheader("Quality of Service Metrics in Milliseconds")
    avg_rtt_dl = filtered_data['Avg RTT DL (ms)'].mean()
    avg_rtt_ul = filtered_data['Avg RTT UL (ms)'].mean()
    st.metric("Average RTT DL (ms)", round(avg_rtt_dl, 2))
    st.metric("Average RTT UL (ms)", round(avg_rtt_ul, 2))

    # Retransmission Volumes
    st.subheader("Retransmission Volumes in Bytes")
    retrans_volumes = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    retransmission_data = filtered_data[retrans_volumes].sum()
    st.bar_chart(retransmission_data)

    # Performance Metrics
    st.subheader("Performance Metrics in Kbps")
    throughput_ranges = {
        'DL TP < 50 Kbps': (filtered_data['Avg Bearer TP DL (kbps)'] < 50).sum(),
        '50 Kbps < DL TP < 250 Kbps': ((filtered_data['Avg Bearer TP DL (kbps)'] >= 50) & (filtered_data['Avg Bearer TP DL (kbps)'] < 250)).sum(),
        'DL TP > 250 Kbps': (filtered_data['Avg Bearer TP DL (kbps)'] >= 250).sum()
    }
    st.bar_chart(throughput_ranges)

    # Customer Demographics
    st.subheader("Customer Demographics")
    handset_distribution = filtered_data['Handset Manufacturer'].value_counts()
    st.bar_chart(handset_distribution)

    # IMSI and MSISDN Distribution
    st.subheader("IMSI and MSISDN Distribution")
    st.write(filtered_data[['IMSI', 'MSISDN/Number']].drop_duplicates())

    # Visualizations
    st.subheader("Trends Over Time")
    
    # Active Users Over Time
    # active_users_over_time = filtered_data.groupby(filtered_data['Start'].dt.to_period('M'))['MSISDN/Number'].nunique()
    # st.line_chart(active_users_over_time)

    # Session Duration Trends
    # session_duration_over_time = filtered_data.groupby(filtered_data['Start'].dt.to_period('M'))['Dur. (ms)'].mean()
    # st.line_chart(session_duration_over_time)

    # Data Usage vs. Session Duration
    st.subheader("Data Usage vs. Session Duration")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_data, x='Dur. (ms)', y='Total DL (Bytes)', ax=ax)
    ax.set_title("Data Usage vs. Session Duration")
    ax.set_xlabel("Session Duration (s)")
    ax.set_ylabel("Total DL (Bytes)")
    st.pyplot(fig)

    # Number of Users Per Day for the Selected Month
    # st.subheader("Number of Users Per Day for the Selected Month")
    # selected_start_date = pd.to_datetime(start_date)
    # selected_month = selected_start_date.month
    # selected_year = selected_start_date.year

    # # Filter for the selected month and year
    # monthly_users = data[(data['Start'].dt.month == selected_month) & (data['Start'].dt.year == selected_year)]
    # daily_users = monthly_users.groupby(monthly_users['Start'].dt.day)['MSISDN/Number'].nunique()
    
    # # Create a bar chart for daily users
    # st.bar_chart(daily_users)

if __name__ == "__main__":
    main()