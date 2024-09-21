import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib for saving model

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

# Save the KMeans model and the scaler as a .pkl file
joblib.dump({'model': kmeans, 'scaler': scaler}, 'kmeans_model.pkl')

print("Model and scaler saved as kmeans_model.pkl")
