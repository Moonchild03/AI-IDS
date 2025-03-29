import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
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
print("Features used in the model:")
print(train_data.columns)

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

# Visualize class distribution after SMOTE
print("\nClass Distribution After SMOTE:")
class_distribution = train_labels['Label'].value_counts(normalize=True) * 100
print(class_distribution)

# Bar plot for class distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values, palette="viridis")
plt.title("Class Distribution After SMOTE Processing")
plt.xlabel("Class")
plt.ylabel("Proportion (%)")
plt.show()

# Reduce dataset size for RandomizedSearchCV
sample_size = 100000  # Set sample size
train_data_sample, _, train_labels_sample, _ = train_test_split(
    train_data, train_labels, stratify=train_labels, test_size=(1 - sample_size / len(train_data)), random_state=42
)

# Define Random Forest model
print("\nDefining Random Forest model...")
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'max_samples': [0.8, 0.9, 1.0]
}

# Set up RandomizedSearchCV for hyperparameter tuning
print("\nRunning RandomizedSearchCV for hyperparameter tuning on a sample dataset...")
random_search = RandomizedSearchCV(
    estimator=rf_model, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1, verbose=2
)
random_search.fit(train_data_sample, train_labels_sample['Label'])

# Best parameters and score from random search
print("\nBest Parameters from RandomizedSearchCV:", random_search.best_params_)
print("Best Cross-validation Score:", random_search.best_score_)

# Train the model with the best parameters on the full dataset
best_rf_model = random_search.best_estimator_
print("\nTraining model on the full dataset...")
best_rf_model.fit(train_data, train_labels['Label'])

# Evaluate the model on the test data
y_pred = best_rf_model.predict(test_data)

print("\nClassification Report on Test Data:\n", classification_report(test_labels, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=train_labels['Label'].unique(), yticklabels=train_labels['Label'].unique())
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nSaving the optimized model...")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "optimized_random_forest_model.pkl")
joblib.dump(best_rf_model, model_path, compress=3)
print(f"Optimized model saved at: {model_path}")
