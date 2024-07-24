import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
df = pd.read_csv('credit_card_fraud3.csv')
print(df.head())
print(df.info())
print(df['Class'].value_counts())
df = df.dropna()
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
ann_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
ann_classifier.fit(X_train, y_train)
mean_vector = np.mean(X_train, axis=0)
cov_matrix = np.cov(X_train.T)
mahalanobis_distance = np.zeros(X_test.shape[0])
epsilon = 1e-6
for i in range(X_test.shape[0]):
    cov_matrix_reg = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])
    mahalanobis_distance[i] = np.sqrt((X_test[i] - mean_vector).T.dot(np.linalg.inv(cov_matrix_reg)).dot(X_test[i] - mean_vector))
ann_predictions = ann_classifier.predict(X_test)
t2_threshold = 3.0
hybrid_predictions = np.logical_and(ann_predictions == 1, mahalanobis_distance > t2_threshold)
print("Hybrid Model Evaluation:")
print(classification_report(y_test, hybrid_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, hybrid_predictions))
print("ROC-AUC Score:")
print(roc_auc_score(y_test, hybrid_predictions))
