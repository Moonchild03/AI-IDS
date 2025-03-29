import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Define paths
base_dir = os.path.expanduser("C:\\Users\\Kick Buttosky\\Desktop\\project_1\\AnomalyDetectionProject-20241125T101827Z-001\\AnomalyDetectionProject")
dataset_dir = os.path.join(base_dir, "datasets\\Downloads")
processed_dir = os.path.join(dataset_dir, "processed")

# Load preprocessed data
train_data = pd.read_csv(os.path.join(processed_dir, "train_data.csv"))
test_data = pd.read_csv(os.path.join(processed_dir, "test_data.csv"))
train_labels = pd.read_csv(os.path.join(processed_dir, "train_labels.csv"))
test_labels = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))

# Rename columns to 'Label' if necessary
if 'Label' not in train_labels.columns:
    train_labels.columns = ['Label']
    print("Renamed train_labels columns:", train_labels.columns)

if 'Label' not in test_labels.columns:
    test_labels.columns = ['Label']
    print("Renamed test_labels columns:", test_labels.columns)

# Load the trained model
rf_model = joblib.load(os.path.join(base_dir, "models/optimized_random_forest_model.pkl"))

# Evaluate the model on the training data
train_pred = rf_model.predict(train_data)
print("\nClassification Report for Training Data:")
print(classification_report(train_labels['Label'], train_pred))
print("\nConfusion Matrix for Training Data:")
print(confusion_matrix(train_labels['Label'], train_pred))

# Evaluate the model on the test data
test_pred = rf_model.predict(test_data)
print("\nClassification Report for Test Data:")
print(classification_report(test_labels['Label'], test_pred))
print("\nConfusion Matrix for Test Data:")
print(confusion_matrix(test_labels['Label'], test_pred))

# Calculate and compare performance metrics
train_accuracy = (train_pred == train_labels['Label']).mean()
test_accuracy = (test_pred == test_labels['Label']).mean()
print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Check for overfitting
if train_accuracy - test_accuracy > 0.05:
    print("\nWarning: The model is likely overfitting!")
else:
    print("\nThe model does not show significant signs of overfitting.")
