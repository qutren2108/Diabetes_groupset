import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Step 1: Load the prepared data
# Correct paths to the processed_data folder
X_train = pd.read_csv('../data_preparation/processed_data/X_train.csv')
X_test = pd.read_csv('../data_preparation/processed_data/X_test.csv')
y_train = pd.read_csv('../data_preparation/processed_data/y_train.csv').squeeze()
y_test = pd.read_csv('../data_preparation/processed_data/y_test.csv').squeeze()

# Check the values in y_train
print("Unique values in y_train:", y_train.unique())
print("Unique values in y_test:", y_test.unique())

# Step 2: Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = svm_model.predict(X_test)
y_pred_prob = svm_model.predict_proba(X_test)[:, 1]

# Step 4: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Step 5: Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('../models/confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")

# Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [accuracy, precision, recall, f1, roc_auc]
})

try:
    metrics.to_csv('../models/svm_metrics.csv', index=False)
    print("Metrics successfully saved to ../models/svm_metrics.csv!")
except Exception as e:
    print(f"Error while saving metrics: {e}")

# Perform 5-fold cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation results
print("Cross-validation accuracy scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())
print("Standard deviation of accuracy:", cv_scores.std())

# Check class distribution
print("\nChecking class distribution...")
print("Training target distribution:")
print(y_train.value_counts())

print("Testing target distribution:")
print(y_test.value_counts())

# Check for potential data leakage
print("\nChecking for data leakage...")
print("Feature columns in X_train:")
print(X_train.columns)

# Drop target-related features if they exist
columns_to_drop = ['LBXGLU', 'LBXGH']
X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
X_test = X_test.drop(columns=columns_to_drop, errors='ignore')

print("Updated feature columns in X_train:")
print(X_train.columns)

# Save cross-validation results to the metrics file
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Mean Cross-Validation Accuracy", "CV Accuracy Std Dev"],
    "Score": [accuracy, precision, recall, f1, roc_auc, cv_scores.mean(), cv_scores.std()]
})

metrics.to_csv('../models/svm_metrics.csv', index=False)
print("\nMetrics including cross-validation saved to svm_metrics.csv!")
