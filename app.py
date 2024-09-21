import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set up Streamlit interface
st.title('Stock Management Clustering App')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Prepare the data
    df['InvDt'] = pd.to_datetime(df['InvDt'])
    df['Month'] = df['InvDt'].dt.month  # Extract month number
    df['CateVol'] = df['CateName'] + '-' + df['Capacity'].astype(str)
    df['TotalSales'] = df['ItmQty'] * df['ItmPrice']  # Calculate total sales
    df['Profit'] = (df['ItmPrice'] - df['CostPrice']) * df['ItmQty']  # Calculate profit

    # Group by CateVol and Month, and aggregate sales and profit
    grouped = df.groupby(['CateVol', 'Month']).agg({'TotalSales': 'sum', 'Profit': 'sum'}).reset_index()

    # Prepare data for clustering
    X = grouped[['TotalSales', 'Profit']]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering to categorize into high, mid, low clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    grouped['Cluster'] = kmeans.fit_predict(X_scaled)

    # Assign proper cluster names (high, mid, low based on sales and profit)
    cluster_labels = {0: 'Low Sales-Low Profit', 1: 'Mid Sales-Mid Profit', 2: 'High Sales-High Profit'}
    
    # Re-assign human-readable cluster names
    grouped['ClusterName'] = grouped['Cluster'].map(cluster_labels)

    # Function to predict percentages for a given month
    def get_stock_management(month):
        # Filter data for the given month
        month_data = grouped[grouped['Month'] == month]
        
        # Calculate the total sales for the month
        total_sales = month_data['TotalSales'].sum()
        
        # Calculate percentage for each CateVol in each cluster
        category_ratios = {}
        for cluster in ['Low Sales-Low Profit', 'Mid Sales-Mid Profit', 'High Sales-High Profit']:
            cluster_data = month_data[month_data['ClusterName'] == cluster]
            for catevol in cluster_data['CateVol'].unique():
                catevol_sales = cluster_data[cluster_data['CateVol'] == catevol]['TotalSales'].sum()
                ratio = (catevol_sales / total_sales) * 100  # Convert to percentage
                category_ratios[f'{cluster} - {catevol}'] = ratio

        return category_ratios
    # Streamlit UI for month selection
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Streamlit UI for month selection (display month names instead of numbers)
    selected_month_name = st.selectbox("Select Month", options=month_names)
    
    # Map selected month name back to its corresponding number
    month = month_names.index(selected_month_name) + 1

    # Display the stock management percentages for the selected month
    if st.button('Show Stock Management Percentages'):
        percentages = get_stock_management(month)
        st.write(f"Stock management percentages for month {selected_month_name}:")
        for key, value in percentages.items():
            st.write(f'{key}: {value:.2f}%')

    # Optionally, plot the cluster distribution
    if st.button('Show Cluster Distribution'):
        fig, ax = plt.subplots()
        grouped.groupby('ClusterName')['TotalSales'].sum().plot(kind='bar', ax=ax)
        ax.set_title('Total Sales by Cluster')
        ax.set_ylabel('Total Sales')
        st.pyplot(fig)

else:
    st.write("Please upload a CSV file to proceed.")
