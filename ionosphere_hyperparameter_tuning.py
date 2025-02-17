# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:03:04 2025

@author: harik
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from GLS_Coding_Classifier import GLSCompressionClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

#import the BANK NOTE AUTHENTICATION Dataset 
#import the IONOSPHERE Dataset 
ionosphere = np.array(pd.read_csv('ionosphere_data.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = ionosphere[:,range(0,ionosphere.shape[1]-1)], ionosphere[:,ionosphere.shape[1]-1].astype(str)


#Norm: B -> 0;  G -> 1
y = y.reshape(len(y),1)
y = np.char.replace(y, 'b', '0', count=None)
y = np.char.replace(y, 'g', '1', count=None)
y = y.astype(int)


scaler = MinMaxScaler()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the training and test data
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Hyperparameter tuning setup
threshold_values = np.arange(0.01, 1.00, 0.01)  # Range of thresholds to explore
k = 5  # K-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=90)

best_threshold = None
best_f1_score = 0

# Store results
threshold_f1_scores = []

# Perform k-fold cross-validation for each threshold
for threshold in threshold_values:
    f1_scores = []
    
    for train_idx, val_idx in kf.split(X_train_norm):
        X_train_, X_val = X_train_norm[train_idx], X_train_norm[val_idx]
        y_train_, y_val = y_train[train_idx], y_train[val_idx]
        
        # Initialize the GLSCompressionClassifier with the current threshold
        n_classes = len(np.unique(y))  # 3 classes in ionosphere dataset
        classifier = GLSCompressionClassifier(n_classes, threshold=threshold, alpha=0.001, fast_mode=True)
        
        # Fit the classifier on the training data
        classifier.fit(X_train_, y_train_)
        
        # Predict on the validation data
        y_pred = classifier.predict(X_val)
        
        # Calculate the F1 score for this fold
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        f1_scores.append(f1)
    
    # Average F1-score over all folds
    mean_f1 = np.mean(f1_scores)
    threshold_f1_scores.append((threshold, mean_f1))
    
    # Update the best threshold if the current one is better
    if mean_f1 > best_f1_score:
        best_f1_score = mean_f1
        best_threshold = threshold

# Print the best threshold and F1 score
print(f"Best threshold: {best_threshold} with F1-score: {best_f1_score}")

# Save the results
results_dir = "hyperparameter_tuning/ionosphere/"
os.makedirs(results_dir, exist_ok=True)

# Save best threshold
np.save(os.path.join(results_dir, "best_threshold.npy"), best_threshold)

# Save tuning results as CSV
np.savetxt(os.path.join(results_dir, "tuning_results.csv"), threshold_f1_scores, delimiter=",", header="Threshold,F1-Score", comments="")

# Plot F1-score vs. Threshold
thresholds, f1_scores = zip(*threshold_f1_scores)
import matplotlib.pyplot as plt
plt.plot(thresholds, f1_scores, marker='o', linestyle='-')
plt.xlabel("Threshold", fontsize=15)
plt.ylabel("Average Macro F1-Score", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.savefig(os.path.join(results_dir, "f1_vs_threshold_plot.png"))
plt.show()

