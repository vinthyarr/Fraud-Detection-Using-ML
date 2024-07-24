import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from scipy.stats import f

# Load the dataset from CSV
data = pd.read_csv('credit_card_fraud.csv')

# Split the dataset into input features (X) and labels (y)
X = data.drop('Class', axis=1).values
y = data['Class'].values

# Train the neural network
clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, random_state=42, activation='logistic', solver='adam')
clf.fit(X, y)

alpha = 0.05
T2_values = []

# Perform real-time monitoring (modify as needed)
for i in range(len(X)):
    prediction = clf.predict(X[i].reshape(1, -1))
    residuals = X[i] - clf.predict_proba(X[i].reshape(1, -1))[0, 1]
    coefs = clf.coefs_[0]
if coefs.shape[0] == coefs.shape[1]:
    coefs_inv = np.linalg.inv(coefs)
    t2_value = np.dot(residuals, coefs_inv.T @ residuals)
else:
    # Handle the case when the matrix is not square
    t2_value = None  # or any other appropriate handling

    t2_value = np.dot(residuals, np.linalg.inv(clf.coefs_[0]).T @ residuals)
    T2_values.append(t2_value)
    
    if i >= 3:
        n = len(T2_values)
        T2_mean = np.mean(T2_values[:n-1])
        T2_std = np.std(T2_values[:n-1], ddof=1)
        control_limit = np.sqrt((n - 1) * (n - 4) / (n * (n - 2))) * f.ppf(1 - alpha, n, n - 2)

        if t2_value > T2_mean + control_limit:
            print(f"Fraud detected at index {i}!")
        else:
            print(f"No fraud detected at index {i}.")
