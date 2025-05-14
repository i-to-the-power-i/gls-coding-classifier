


import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from GLS_Coding_Classifier import GLSCompressionClassifier

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

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
        
        n_classes = len(np.unique(y))  # 3 classes in BNA dataset
        classifier = GLSCompressionClassifier(n_classes, threshold=threshold, alpha=1, fast_mode=True)
        
        # Fit the classifier on the training data
        classifier.fit(X_train_, y_train_)
        
        # Predict on the validation data
        predicted_classes = classifier.predict(X_val)
        
        # Compute macro F1-score
        f1 = f1_score(y_val, predicted_classes, average='macro', zero_division=0)
        f1_scores.append(f1)
    
    # Average F1-score over folds
    mean_f1 = np.mean(f1_scores)
    threshold_f1_scores.append((threshold, mean_f1))

# Find best threshold
best_threshold, best_f1 = max(threshold_f1_scores, key=lambda x: x[1])

# Save results
results_dir = "hyperparameter-tuning/breast-cancer/"
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



