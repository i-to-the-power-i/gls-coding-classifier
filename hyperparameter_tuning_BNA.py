# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:03:04 2025

@author: harik
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from GLS_CODING_SUPPORT_FUNCTION import GLS_CODING_FIT_SECOND, GLS_CODING_PREDICT_SECOND_AVG
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

#import the BANK NOTE AUTHENTICATION Dataset 
bank = np.array(pd.read_csv('data_banknote_authentication.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = bank[:,range(0,bank.shape[1]-1)], bank[:,bank.shape[1]-1]
y = y.reshape(len(y),1)
# X = X.astype(float)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Normalize dataset
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Hyperparameter tuning setup
k = 5  # K-folds
kf = KFold(n_splits=k, shuffle=True, random_state=90)
threshold_values = np.arange(0.01, 1.00, 0.01)

# Store results
threshold_f1_scores = []

# Perform k-fold cross-validation for each threshold
for threshold in threshold_values:
    f1_scores = []
    
    for train_idx, val_idx in kf.split(X_train_norm):
        X_train_, X_val = X_train_norm[train_idx], X_train_norm[val_idx]
        y_train_, y_val = y_train[train_idx], y_train[val_idx]
        
        # Train model
        avg_class_probabilities = GLS_CODING_FIT_SECOND(X_train_, y_train_, threshold)
        
        # Predict on validation set
        predicted_classes = GLS_CODING_PREDICT_SECOND_AVG(X_val, len(np.unique(y_train_)), avg_class_probabilities, threshold)
        
        # Compute macro F1-score
        f1 = f1_score(y_val, predicted_classes, average='macro', zero_division=0)
        f1_scores.append(f1)
    
    # Average F1-score over folds
    mean_f1 = np.mean(f1_scores)
    threshold_f1_scores.append((threshold, mean_f1))

# Find best threshold
best_threshold, best_f1 = max(threshold_f1_scores, key=lambda x: x[1])

# Save results
results_dir = "hyperparameter-tuning/BNA/"
os.makedirs(results_dir, exist_ok=True)

# Save best threshold
np.save(os.path.join(results_dir, "best_threshold.npy"), best_threshold)

# Load and verify the saved threshold
loaded_threshold = np.load(os.path.join(results_dir, "best_threshold.npy"))
assert np.isclose(loaded_threshold, best_threshold), "Error: Loaded threshold does not match saved value!"

# Save tuning results as CSV
np.savetxt(os.path.join(results_dir, "tuning_results.csv"), threshold_f1_scores, delimiter=",", header="Threshold,F1-Score", comments="")

# Plot F1-score vs. Threshold
thresholds, f1_scores = zip(*threshold_f1_scores)
plt.plot(thresholds, f1_scores, marker='o', linestyle='-')
plt.xlabel("Threshold", fontsize=15)
plt.ylabel("Average Macro F1-Score", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title("Hyperparameter Tuning: Threshold vs. F1-Score")
plt.grid()
plt.savefig(os.path.join(results_dir, "f1_vs_threshold_plot.png"))
plt.show()

print(f"Best threshold: {best_threshold} with F1-score: {best_f1}")


