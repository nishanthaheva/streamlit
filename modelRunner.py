import streamlit as st
import pandas as pd
import joblib

# Load the saved KMeans model and scaler
data = joblib.load('kmeans_model.pkl')
kmeans = data['model']
scaler = data['scaler']

# Load and prepare data (recreate the grouped DataFrame)
@st.cache_data  # Cache the data loading and processing for faster performance
def load_and_prepare_data():
    df = pd.read_csv('Data1.csv')
    df['InvDt'] = pd.to_datetime(df['InvDt'])
    df['Month'] = df['InvDt'].dt.month  # Extract month number
    df['CateVol'] = df['CateName'] + '-' + df['Capacity'].astype(str)
    df['TotalSales'] = df['ItmQty'] * df['ItmPrice']  # Calculate total sales
    df['Profit'] = (df['ItmPrice'] - df['CostPrice']) * df['ItmQty']  # Calculate profit
    
    # Group by CateVol and Month, and aggregate sales and profit
    grouped = df.groupby(['CateVol', 'Month']).agg({'TotalSales': 'sum', 'Profit': 'sum'}).reset_index()
    
    # Prepare data for clustering
    X = grouped[['TotalSales', 'Profit']]
    
    # Standardize the data using the loaded scaler
    X_scaled = scaler.transform(X)
    
    # Predict clusters using the loaded KMeans model
    grouped['Cluster'] = kmeans.predict(X_scaled)
    
    # Assign proper cluster names (high, mid, low based on sales and profit)
    cluster_labels = {0: 'Low Sales-Low Profit', 1: 'Mid Sales-Mid Profit', 2: 'High Sales-High Profit'}
    grouped['ClusterName'] = grouped['Cluster'].map(cluster_labels)
    
    return grouped

# Load and process data
grouped = load_and_prepare_data()

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

# Streamlit UI
st.title("Stock Management Percentages by Month")

# Create a select box for month selection
month_name_to_number = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
    "June": 6, "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12
}

# Select box with month names
selected_month_name = st.selectbox("Select a month:", list(month_name_to_number.keys()))
selected_month = month_name_to_number[selected_month_name]

# Get stock management percentages for the selected month
percentages = get_stock_management(selected_month)

# Display the results
st.subheader(f"Stock management percentages for {selected_month_name}:")
for key, value in percentages.items():
    st.write(f"{key}: {value:.2f}%")
