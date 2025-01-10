import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

X_train = pd.read_csv('C:/Users/User/OneDrive/Documents/Study/Notes/ML/Group Asg/diabetes_group_set/data_preparation/processed_data/X_train.csv')
X_test = pd.read_csv('C:/Users/User/OneDrive/Documents/Study/Notes/ML/Group Asg/diabetes_group_set/data_preparation/processed_data/X_test.csv')
y_train = pd.read_csv('C:/Users/User/OneDrive/Documents/Study/Notes/ML/Group Asg/diabetes_group_set/data_preparation/processed_data/y_train.csv')
y_test = pd.read_csv('C:/Users/User/OneDrive/Documents/Study/Notes/ML/Group Asg/diabetes_group_set/data_preparation/processed_data/y_test.csv')


X_train_np = X_train.drop(columns=['SEQN']).values  
y_train_np = y_train['target'].values
X_test_np = X_test.drop(columns=['SEQN']).values
y_test_np = y_test['target'].values


model = MLPClassifier(hidden_layer_sizes=(128, 64), 
                      activation='relu', 
                      solver='adam', 
                      alpha=0.0001, 
                      batch_size=32, 
                      learning_rate='adaptive', 
                      max_iter=500, 
                      random_state=42)

model.fit(X_train_np, y_train_np)


predictions = model.predict(X_test_np)
accuracy = accuracy_score(y_test_np, predictions)
report = classification_report(y_test_np, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)