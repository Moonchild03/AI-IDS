import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# Define paths
base_dir = os.path.expanduser("C:\\Users\\Kick Buttosky\\Desktop\\project_1\\AnomalyDetectionProject-20241125T101827Z-001\\AnomalyDetectionProject")
dataset_dir = os.path.join(base_dir, "datasets\\Downloads")
processed_dir = os.path.join(dataset_dir, "processed")

# Load preprocessed data
print("Loading preprocessed data...")
train_data = pd.read_csv(os.path.join(processed_dir, "train_data.csv"))
test_data = pd.read_csv(os.path.join(processed_dir, "test_data.csv"))
train_labels = pd.read_csv(os.path.join(processed_dir, "train_labels.csv"))
test_labels = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))
print("Data loaded successfully.")

# Check for 'Label' column
if 'Label' not in train_labels.columns:
    train_labels.columns = ['Label']
if 'Label' not in test_labels.columns:
    test_labels.columns = ['Label']

# Analyze class distribution
print("\nClass Distribution Before Balancing (Training Data):")
print(train_labels['Label'].value_counts(normalize=True) * 100)

# Apply SMOTE in smaller batches
smote = SMOTE(random_state=42)
batch_size = 100000
balanced_data = []
balanced_labels = []

print("\nApplying SMOTE in smaller batches...")
for start in range(0, len(train_data), batch_size):
    end = min(start + batch_size, len(train_data))
    print(f"Processing batch: {start} to {end}")

    batch_data = train_data.iloc[start:end]
    batch_labels = train_labels.iloc[start:end]

    if batch_labels['Label'].nunique() < 2:
        print(f"Skipping batch {start} to {end} due to insufficient classes.")
        balanced_data.append(batch_data)
        balanced_labels.append(batch_labels)
        continue

    smote_k = min(5, batch_labels['Label'].value_counts().min() - 1)
    if smote_k < 1:
        print(f"Skipping SMOTE for batch {start} to {end} as it cannot apply SMOTE with k_neighbors={smote_k}.")
        balanced_data.append(batch_data)
        balanced_labels.append(batch_labels)
        continue

    smote.set_params(k_neighbors=smote_k)
    resampled_data, resampled_labels = smote.fit_resample(batch_data, batch_labels)
    balanced_data.append(pd.DataFrame(resampled_data, columns=train_data.columns))
    balanced_labels.append(pd.DataFrame(resampled_labels, columns=train_labels.columns))

# Combine balanced batches
train_data = pd.concat(balanced_data, axis=0)
train_labels = pd.concat(balanced_labels, axis=0)

print("\nClass Distribution After SMOTE:")
print(train_labels['Label'].value_counts(normalize=True) * 100)

# Reduce dataset size for GridSearchCV
sample_size = 100000  # Set sample size for grid search
train_data_sample, _, train_labels_sample, _ = train_test_split(
    train_data, train_labels, stratify=train_labels, test_size=(1 - sample_size / len(train_data)), random_state=42
)

# Define Random Forest model
print("\nDefining Random Forest model...")
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'max_samples': [0.8, 1.0]
}

# Set up GridSearchCV for hyperparameter tuning
print("\nRunning GridSearchCV for hyperparameter tuning on a sample dataset...")
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_data_sample, train_labels_sample['Label'])

# Best parameters and score from grid search
print("\nBest Parameters from Grid Search:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Train the model with the best parameters on the full dataset
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(train_data, train_labels['Label'])

# Evaluate the model on the test data
y_pred = best_rf_model.predict(test_data)

print("\nClassification Report on Test Data:\n", classification_report(test_labels, y_pred))
print("\nConfusion Matrix on Test Data:\n", confusion_matrix(test_labels, y_pred))

# Save the trained model
print("\nSaving the optimized model...")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "optimized_random_forest_model.pkl")
joblib.dump(best_rf_model, model_path, compress=3)
print(f"Optimized model saved at: {model_path}")
