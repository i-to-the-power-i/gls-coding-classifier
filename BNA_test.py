

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from GLS_Coding_Classifier import GLSCompressionClassifier
import os

#import the BANK NOTE AUTHENTICATION Dataset 
bank = np.array(pd.read_csv('data_banknote_authentication.txt', sep=",", header=None))


#reading data and labels from the dataset
X, y = bank[:,range(0,bank.shape[1]-1)], bank[:,bank.shape[1]-1]
y = y.reshape(len(y),1)
# X = X.astype(float)


# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the training and test data
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Load the best threshold from the saved file
best_threshold = np.load("hyperparameter_tuning/BNA/best_threshold.npy")

# Initialize the GLSCompressionClassifier with the best threshold
n_classes = len(np.unique(y))  # 3 classes in BNA dataset
classifier = GLSCompressionClassifier(n_classes, threshold=best_threshold, alpha=0.001, fast_mode=True)

# Fit the classifier on the training data
classifier.fit(X_train_norm, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test_norm)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Calculate the required metrics
accuracy = accuracy_score(y_test, y_pred)
macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_f1 = f1_score(y_test, y_pred, average='macro')

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Macro Precision: {macro_precision}")
print(f"Macro Recall: {macro_recall}")
print(f"Macro F1 Score: {macro_f1}")
# Assuming you have y_test (true labels) and predicted_classes (predicted labels)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


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
    filename = f"second_return_skew_tent_map_bna_class_{class_label}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
