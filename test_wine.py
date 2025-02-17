# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 08:48:31 2025

@author: harik
"""

# -*- coding: utf-8 -*-

import numpy as np
import os
from sklearn.model_selection import train_test_split
from GLS_Coding_Classifier import GLSCompressionClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_wine

# Load Iris dataset
data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

from collections import Counter

train_counts = Counter(y_train)
test_counts = Counter(y_test)

# Print results
print("Train data instances per class:", dict(train_counts))
print("Test data instances per class:", dict(test_counts))

# Load the best threshold value from hyperparameter tuning
threshold_file = "hyperparameter-tuning/wine/best_threshold.npy"
if not os.path.exists(threshold_file):
    raise FileNotFoundError(f"Threshold file {threshold_file} not found. Run hyperparameter tuning first.")

best_threshold = np.load(threshold_file)
print(f"Loaded Best Threshold: {best_threshold}")

# Train the model using the best threshold
# Initialize the GLSCompressionClassifier with the best threshold
n_classes = len(np.unique(y_train))  # 3 classes in BNA dataset
classifier = GLSCompressionClassifier(n_classes, threshold=best_threshold, alpha=0.001, fast_mode=True)

# Fit the classifier on the training data
classifier.fit(X_train_norm, y_train)
# Predict on the test data
predicted_classes = classifier.predict(X_test_norm)
# Compute performance metrics
accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes, average='macro', zero_division=0)
recall = recall_score(y_test, predicted_classes, average='macro', zero_division=0)
f1 = f1_score(y_test, predicted_classes, average='macro', zero_division=0)

# Store results in a dictionary
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "best_threshold": best_threshold
}

# Define result folder and save results
output_folder = "test-results/wine/"
os.makedirs(output_folder, exist_ok=True)

# Save metrics as a .npy file
np.save(os.path.join(output_folder, "test_metrics.npy"), metrics)

# Print results
print("\nTest Results (wine):")
for key, value in metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")

# Load the saved results to verify
loaded_metrics = np.load(os.path.join(output_folder, "test_metrics.npy"), allow_pickle=True).item()
print("\nLoaded Test Results:")
for key, value in loaded_metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")


avg_class_probabilities = classifier.avg_class_probabilities


import numpy as np
import matplotlib.pyplot as plt

# Assume avg_class_probabilities is a dictionary with keys as class labels (0 to n_classes-1)
# and values as dictionaries with keys (0,0), (0,1), (1,1), (1,0). For example:
# avg_class_probabilities = { 
#    0: {(0,0): 0.2, (0,1): 0.3, (1,1): 0.3, (1,0): 0.2},
#    1: {(0,0): 0.15, (0,1): 0.35, (1,1): 0.3, (1,0): 0.2},
#    ... 
# }
#
# Also assume that n_classes is defined.

for class_label in range(n_classes):
    # Retrieve the average probabilities for this class.
    avg_probs = avg_class_probabilities[class_label]
    p00 = avg_probs[(0,0)]
    p01 = avg_probs[(0,1)]
    p11 = avg_probs[(1,1)]
    p10 = avg_probs[(1,0)]
    
    # Generate x values and compute y values using the second return map.
    x_values = np.linspace(0, 1, 1000)
    y_values = [classifier.second_return_map_skew_tent(x, p00, p01, p11, p10) for x in x_values]
    
    # Plot the function.
    plt.figure(figsize=(6, 6))
    plt.plot(x_values, y_values, color='blue', linewidth=2)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("T(T(x))", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save the figure with a filename that includes the class label.
    filename = f"second_return_skew_tent_map_wine_class_{class_label}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()