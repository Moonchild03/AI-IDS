import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Define paths
base_dir = os.path.expanduser("C:\\Users\\Kick Buttosky\\Desktop\\project_1\\AnomalyDetectionProject-20241125T101827Z-001\\AnomalyDetectionProject")
dataset_dir = os.path.join(base_dir, "datasets\\Downloads")
processed_dir = os.path.join(dataset_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)

# Input and output paths
data_path = os.path.join(dataset_dir, "combined_dataset.csv")
train_data_path = os.path.join(processed_dir, "train_data.csv")
test_data_path = os.path.join(processed_dir, "test_data.csv")

# Initialize LabelEncoder and StandardScaler
le = LabelEncoder()
scaler = StandardScaler()

# Chunk size
chunk_size = 100000
print(f"Processing dataset in chunks from {data_path}...")

# Helper function to clean and validate data
def clean_chunk(chunk):
    # Drop unnecessary columns
    columns_to_drop = ['Flow ID', 'Timestamp', 'Attempted Category']
    chunk.drop(columns=[col for col in columns_to_drop if col in chunk], axis=1, inplace=True)

    # Handle missing values
    chunk.fillna(0, inplace=True)

    # Process non-numeric columns
    for col in chunk.select_dtypes(include=["object"]).columns:
        if col == 'Label':
            chunk[col] = le.fit_transform(chunk[col])  # Encode labels
        else:
            # Encode other categorical columns like IPs or ports
            chunk[col] = chunk[col].astype('category').cat.codes

    # Replace invalid numeric values
    chunk.replace([float('inf'), float('-inf')], 0, inplace=True)

    return chunk

# Process dataset in chunks
for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size)):
    print(f"Processing chunk {i+1} with shape {chunk.shape}...")

    # Clean the chunk
    chunk = clean_chunk(chunk)

    # Separate features and labels
    X_chunk = chunk.drop("Label", axis=1)
    y_chunk = chunk["Label"]

    # Validate numeric values and replace NaN/Inf
    X_chunk.replace([float('inf'), float('-inf')], 0, inplace=True)
    X_chunk.fillna(0, inplace=True)

    # Scale numeric features
    X_chunk_numeric = X_chunk.select_dtypes(include=["number"])
    X_scaled_chunk = scaler.fit_transform(X_chunk_numeric)

    # Save processed chunk
    if i == 0:
        pd.DataFrame(X_scaled_chunk).to_csv(train_data_path, index=False, header=True)
        pd.DataFrame(y_chunk).to_csv(test_data_path, index=False, header=True)
    else:
        pd.DataFrame(X_scaled_chunk).to_csv(train_data_path, index=False, mode='a', header=False)
        pd.DataFrame(y_chunk).to_csv(test_data_path, index=False, mode='a', header=False)

print(f"Processed dataset saved in {processed_dir}")
