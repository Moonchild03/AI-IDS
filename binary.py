import pandas as pd
import os

# Define paths
dataset_path = "C:\\Users\\Kick Buttosky\\Desktop\\project_1\\AnomalyDetectionProject-20241125T101827Z-001\\AnomalyDetectionProject\\datasets\\Downloads\\combined_dataset.csv"
binary_dataset_path = "C:\\Users\\Kick Buttosky\\Desktop\\project_1\\AnomalyDetectionProject-20241125T101827Z-001\\AnomalyDetectionProject\\datasets\\Downloads\\binary_combined_dataset.csv"

# Set chunk size
chunk_size = 100000

# Initialize class distribution counters for both original and binary labels
original_class_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
binary_class_distribution = {0: 0, 1: 0}

# Process the dataset in chunks
print("Processing dataset in chunks...")
for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
    # Ensure 'Label' is numeric and drop rows with invalid labels
    chunk['Label'] = pd.to_numeric(chunk['Label'], errors='coerce')
    chunk = chunk.dropna(subset=['Label'])

    # Count original class distribution
    for label in original_class_distribution:
        original_class_distribution[label] += (chunk['Label'] == label).sum()
    
    # Relabel dataset (convert multi-class to binary)
    chunk['Binary_Label'] = chunk['Label'].apply(lambda x: 0 if x == 0 else 1)
    
    # Count binary class distribution
    binary_class_distribution[0] += (chunk['Binary_Label'] == 0).sum()
    binary_class_distribution[1] += (chunk['Binary_Label'] == 1).sum()
    
    # Save chunk to binary dataset file
    chunk.to_csv(binary_dataset_path, mode='a', index=False, header=not os.path.exists(binary_dataset_path))

# Final class distribution for both original and binary labels
print("\nFinal Class Distribution (Original Labels):")
for label, count in original_class_distribution.items():
    print(f"Class {label}: {count}")

print("\nFinal Class Distribution (Binary Labels):")
for label, count in binary_class_distribution.items():
    print(f"Class {label}: {count}")

print(f"\nBinary classification dataset saved at: {binary_dataset_path}")
