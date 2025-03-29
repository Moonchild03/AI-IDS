import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset (replace with the actual dataset path)
base_dir = os.path.expanduser("C:\\Users\\Kick Buttosky\\Desktop\\project_1\\AnomalyDetectionProject-20241125T101827Z-001\\AnomalyDetectionProject")
dataset_dir = os.path.join(base_dir, "datasets\\Downloads")
processed_dir = os.path.join(dataset_dir, "processed")
df= os.path.join(dataset_dir, "combined_dataset.csv")
# Selecting the numeric features for correlation
numeric_features = ['Packet Size', 'TTL', 'Flow Duration', 'Byte Count', 'Packets per Flow']

# Calculate the correlation matrix
corr_matrix = df[numeric_features].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()
