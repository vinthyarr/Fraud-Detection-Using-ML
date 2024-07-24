import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load the dataset
df = pd.read_csv('credit_card_fraud.csv')
# Explore the dataset
print(df.head())
print(df.info())
print(df['Class'].value_counts())
# Handling missing values (if any)
df = df.dropna()
# Split the data into features (X) and labels (y)
X = df.drop('Class', axis=1)
y = df['Class']
# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Initialize the ANN classifier
ann_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
# Train the ANN on the training data
ann_classifier.fit(X_train, y_train)
# Calculate the mean and covariance matrix of the training data
mean_vector = np.mean(X_train, axis=0)
cov_matrix = np.cov(X_train.T)
# Calculate the Mahalanobis distance for each data point in the test set
mahalanobis_distance = np.zeros(X_test.shape[0])
for i in range(X_test.shape[0]):
    mahalanobis_distance[i] = np.sqrt((X_test[i] - mean_vector).T.dot(np.linalg.inv(cov_matrix)).dot(X_test[i] - mean_vector))
# Make predictions using the ANN
ann_predictions = ann_classifier.predict(X_test)
# Threshold for T2 Control Chart (you can fine-tune this value based on validation)
t2_threshold = 3.0
# Combine predictions using logical AND
hybrid_predictions = np.logical_and(ann_predictions == 1, mahalanobis_distance > t2_threshold)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print("Hybrid Model Evaluation:")
print(classification_report(y_test, hybrid_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, hybrid_predictions))
print("ROC-AUC Score:")
print(roc_auc_score(y_test, hybrid_predictions))


