import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA  # For dimensionality reduction to 2D

# Define paths
base_dir = os.path.expanduser("C:\\Users\\Kick Buttosky\\Desktop\\project_1\\AnomalyDetectionProject-20241125T101827Z-001\\AnomalyDetectionProject")
dataset_dir = os.path.join(base_dir, "datasets\\Downloads")
processed_dir = os.path.join(dataset_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)

# Input and output paths
data_path = os.path.join(dataset_dir, "combined_dataset.csv")
train_data_path = os.path.join(processed_dir, "train_data.csv")
test_data_path = os.path.join(processed_dir, "test_data.csv")

# Check if the file exists
if not os.path.exists(data_path):
    print(f"File not found at {data_path}")
else:
    print(f"File found at {data_path}")

# Initialize LabelEncoder and StandardScaler
le = LabelEncoder()
scaler = StandardScaler()

# Chunk size - Adjust to handle system memory
chunk_size = 50000  
print("Processing dataset in chunks...")

# Initialize lists to store results
X_train_all = []
y_train_all = []
X_test_all = []
y_test_all = []

# Class distribution tracking
class_distribution = []

# Process dataset in chunks
for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size)):
    print(f"Processing chunk {i+1} with shape {chunk.shape}...")

    # Handle missing values
    chunk.fillna(0, inplace=True)

    # Process non-numeric columns
    for col in chunk.select_dtypes(include=["object"]).columns:
        chunk[col] = le.fit_transform(chunk[col].astype(str))

    # Separate features and labels
    X_chunk = chunk.drop("Label", axis=1, errors="ignore")
    y_chunk = chunk["Label"] if "Label" in chunk.columns else None

    # Replace infinite and large values
    X_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_chunk.fillna(0, inplace=True)

    # Limit extremely large values for float64 compatibility
    X_chunk = np.clip(X_chunk, a_min=-1e10, a_max=1e10)

    # Scale numeric features
    X_scaled_chunk = scaler.fit_transform(X_chunk.select_dtypes(include=["number"]))

    # Perform train-test split after preprocessing each chunk
    if y_chunk is not None:
        # Store class distribution
        class_distribution.append(y_chunk.value_counts(normalize=True))

        X_train_chunk, X_test_chunk, y_train_chunk, y_test_chunk = train_test_split(
            X_scaled_chunk, y_chunk, test_size=0.3, random_state=42
        )

        # Append the data to the lists
        X_train_all.append(X_train_chunk)
        y_train_all.append(y_train_chunk)
        X_test_all.append(X_test_chunk)
        y_test_all.append(y_test_chunk)

# Combine class distributions for visualization
class_distribution_df = pd.concat(class_distribution, axis=1).fillna(0)

# Concatenate the chunks and save to CSV files
print("Saving processed dataset to CSV...")

# Combine all the chunks together
X_train_combined = np.vstack(X_train_all)
y_train_combined = np.hstack(y_train_all)
X_test_combined = np.vstack(X_test_all)
y_test_combined = np.hstack(y_test_all)

# Save the final datasets to CSV
pd.DataFrame(X_train_combined).to_csv(train_data_path, index=False)
pd.DataFrame(X_test_combined).to_csv(test_data_path, index=False)
pd.DataFrame(y_train_combined).to_csv(f"{processed_dir}/train_labels.csv", index=False)
pd.DataFrame(y_test_combined).to_csv(f"{processed_dir}/test_labels.csv", index=False)

print(f"Processed dataset saved in {processed_dir}")

# Visualization of class distribution
print("\nGenerating class distribution plots...")

# Density plot for class distribution in chunks
plt.figure(figsize=(12, 6))
for i, dist in enumerate(class_distribution_df.columns):
    sns.kdeplot(class_distribution_df[dist], label=f"Chunk {i+1}")
plt.title("Density Plot of Class Distribution Across Chunks")
plt.xlabel("Class")
plt.ylabel("Density")
plt.legend()
plt.show()

# Bar plot for final class distribution in training data
final_class_distribution = pd.Series(y_train_combined).value_counts(normalize=True)
plt.figure(figsize=(8, 6))
sns.barplot(x=final_class_distribution.index, y=final_class_distribution.values, palette="viridis")
plt.title("Class Distribution After Processing")
plt.xlabel("Class")
plt.ylabel("Proportion")
plt.show()

# Scatter plot for anomaly distribution
print("\nGenerating scatter plot for anomaly distribution...")

# Apply PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_combined)

# Scatter plot of anomalies (malicious traffic) vs benign traffic
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train_combined, palette='coolwarm', markers=["o", "X"])
plt.title("Anomaly Distribution: Benign vs Malicious Traffic")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Class")
plt.show()
