import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and prepare data
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

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering to categorize into high, mid, low clusters
kmeans = KMeans(n_clusters=3, random_state=0)
grouped['Cluster'] = kmeans.fit_predict(X_scaled)

# Assign proper cluster names (high, mid, low based on sales and profit)
cluster_labels = {0: 'Low Sales-Low Profit', 1: 'Mid Sales-Mid Profit', 2: 'High Sales-High Profit'}

# Assign human-readable cluster names
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

    # Return the percentages
    return category_ratios

month = 3  # Example month (March)
percentages = get_stock_management(month)
print(f"Stock management percentages for month {month}:")
for key, value in percentages.items():
    print(f'{key}: {value:.2f}%')


