# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 08:22:28 2025

@author: harik
"""


import numpy as np
import matplotlib.pyplot as plt


def second_return_map_skew_tent(x, p_00, p_01, p_11, p_10):
    if x < p_00:
        return x / p_00
    elif (x >= p_00) and (x < p_00 + p_01):
        return (p_00 + p_01 - x) / p_01
    elif (x >= p_00 + p_01) and (x < p_00 + p_01 + p_11):
        return (x - (p_00 + p_01)) / p_11
    else:
        return (1 - x) / p_10

# # Define parameters
# p_00 = 0.15
# p_01 = 0.3
# p_11 = 0.3
# p_10 = 0.25
# # Generate x values
# x_values = np.linspace(0, 1, 1000)
# y_values = [second_return_map_skew_tent(x, p_00, p_01, p_11, p_10) for x in x_values]

# # Plot the function
# plt.figure(figsize=(6, 6))
# plt.plot(x_values, y_values, color='blue', linewidth=2)

# # Labels and ticks
# plt.xlabel("x", fontsize=14)
# plt.ylabel("T(x)", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(True, linestyle="--", alpha=0.6)

# # Save the figure
# plt.savefig("skew_tent_map.png", dpi=300, bbox_inches='tight')
# plt.show()



import numpy as np
from collections import Counter


from collections import Counter

# def compute_pair_probabilities(data):
#     # Generate non-overlapping pairs
#     # Laplace Smoothing Applied - For Iris data use this code
#     z = data
#     pairs = [(z[i], z[i+1]) for i in range(0, len(z)-1, 2)]
    
#     # Count occurrences of each pair
#     pair_counts = Counter(pairs)
    
#     # Check if any pair has a zero count
#     all_pairs = [(0,0), (0,1), (1,1), (1,0)]
#     apply_smoothing = any(pair_counts.get(pair, 0) == 0 for pair in all_pairs)
    
#     # Apply Laplace smoothing only if needed
#     if apply_smoothing:
#         total_pairs = sum(pair_counts.values()) + 4
#         probabilities = {pair: (pair_counts.get(pair, 0) + 1) / total_pairs for pair in all_pairs}
#     else:
#         total_pairs = sum(pair_counts.values())
#         probabilities = {pair: pair_counts.get(pair, 0) / total_pairs for pair in all_pairs}
    
#     return probabilities


def compute_pair_probabilities(data):
    # Generate non-overlapping pairs 
    # Without Laplace Smoothing
    z = data
    pairs = [(z[i], z[i+1]) for i in range(0, len(z)-1, 2)]
    
    # Count occurrences of each pair
    pair_counts = Counter(pairs)
    total_pairs = sum(pair_counts.values())
    
    # Ensure all pairs are present
    all_pairs = [(0,0), (0,1), (1,1), (1,0)]
    probabilities = {pair: (pair_counts.get(pair, 0)) / total_pairs for pair in all_pairs}
    
    return probabilities

def skew_tent_map_back_iter(p_00, p_01, p_11, p_10, data):
    '''
    Parameters
    ----------
    p_00, p_01, p_11, p_10 : SCALARS
        DESCRIPTION: Parameters of the Skew Tent Map.
    z : 1D ARRAY
        DESCRIPTION: Symbolic Sequence.
    
    Returns
    -------
    initial value: scalar
        DESCRIPTION: Returns the initial value from a given symbolic sequence and
        given Skew Tent Map. The initial condition is the mean value of the interval obtained
        after back iteration on the map.
    '''
    
    SS = data  # Reverse sequence
    out = np.array([0, 0]).astype(float)
    
    pair_sequence = [(SS[i], SS[i+1]) for i in range(0, len(SS)-1, 2)]
    #print(pair_sequence)
    if pair_sequence[0] == (0, 0):
        out[0] = 0
        out[1] = p_00
    elif pair_sequence[0] == (0, 1):
        out[0] = p_00 
        out[1] = p_00 + p_01 
    elif pair_sequence[0] == (1, 1):
        out[0] =  p_00 + p_01
        out[1] = p_11 + p_00 + p_01
    elif pair_sequence[0] == (1, 0):
        out[0] = 1 - (p_00 + p_01 + p_11)
        out[1] = 1 
    
    for pair in pair_sequence[1:]:
        if pair == (0, 0):
            out[0] = out[0] * p_00
            out[1] = out[1] * p_00
        elif pair == (0, 1): 
            out[0] = p_00 + p_01 - p_01 * out[0]
            out[1] = p_00 + p_01 - p_01 * out[1]
        elif pair == (1, 1):
            out[0] = p_11 * out[0] + p_00 + p_01
            out[1] = p_11 * out[1] + p_00 + p_01
        elif pair == (1, 0):
            out[0] = 1 - p_10 * out[0]
            out[1] = 1 - p_10 * out[1]
        
        if out[0] > out[1]:
            out[0], out[1] = out[1], out[0]
    
    return np.mean(out), out

def compressed_size_of_interval(interval):
    return np.ceil(-np.log2(interval[1] - interval[0] + 1e-50))




def GLS_CODING_FIT_SECOND_UPDATED(X_train_norm, y_train, threshold):
    """
    Fit the GLS Coding classifier using the second return map approach.
    
    For each class, this function computes the probabilities for each sample,
    then aggregates these by computing the average probability for each pair 
    ((0,0), (0,1), (1,1), (1,0)) across all samples of that class.
    
    Using these average probability values and a representative symbolic 
    sequence (the first sample of that class), it computes the back iteration 
    interval and from that the compressed file size.
    
    Parameters
    ----------
    X_train_norm : numpy array
        The normalized training features.
    y_train : numpy array
        The training labels.
    threshold : float
        The threshold used for binarization.
    
    Returns
    -------
    class_probabilities : dict
        For each class, the list of probability dictionaries for each sample.
    class_intervals : dict
        For each class, the list of intervals computed per sample.
    class_initial_val : dict
        For each class, the list of estimated initial values (mean of interval) per sample.
    compressed_file_size : dict
        For each class, the array of compressed file sizes (absolute values) computed per sample.
    train_data_compressed_file_size : list
        The list of average compressed file sizes for each class (computed from the sample sizes).
    avg_class_probabilities : dict
        For each class, a dictionary with the average probability for each pair:
        {(0,0): avg_p00, (0,1): avg_p01, (1,1): avg_p11, (1,0): avg_p10}.
    """
    # Binarize training data
    X_train_bin = (X_train_norm >= threshold).astype(int)
    n_classes = len(np.unique(y_train))
    
    # Initialize dictionaries/lists to store results per class.
    class_probabilities = {}
    class_intervals = {}
    class_initial_val = {}
    compressed_file_size = {}
    train_data_compressed_file_size = []
    avg_class_probabilities = {}
    
    for class_label in range(n_classes):
        # Select samples for the current class.
        class_data = X_train_bin[y_train.flatten() == class_label]
        probabilities_list = []
        intervals_list = []
        initial_val_list = []
        sizes = []
        
        for row in class_data:
            # Compute the pair probabilities for this sample.
            probs = compute_pair_probabilities(row)
            probabilities_list.append(probs)
            
            # Extract probability values for each pair.
            p_00, p_01, p_11, p_10 = probs[(0,0)], probs[(0,1)], probs[(1,1)], probs[(1,0)]
            # Compute the initial value and interval via back iteration.
            
        
        # Compute average probabilities across all samples for this class.
        avg_p00 = np.mean([prob[(0,0)] for prob in probabilities_list])
        avg_p01 = np.mean([prob[(0,1)] for prob in probabilities_list])
        avg_p11 = np.mean([prob[(1,1)] for prob in probabilities_list])
        avg_p10 = np.mean([prob[(1,0)] for prob in probabilities_list])
        avg_class_probabilities[class_label] = {(0,0): avg_p00, (0,1): avg_p01, (1,1): avg_p11, (1,0): avg_p10}
        
        for row in class_data:
            initial_value, interval = skew_tent_map_back_iter(avg_p00, avg_p01, avg_p11, avg_p10, row)
            intervals_list.append(interval)
            initial_val_list.append(initial_value)
            sizes.append(compressed_size_of_interval(interval))
    
    # Store per-sample results for the class.
        class_probabilities[class_label] = probabilities_list
        class_intervals[class_label] = intervals_list
        class_initial_val[class_label] = initial_val_list
        compressed_file_size[class_label] = np.abs(sizes)
        train_data_compressed_file_size.append(np.mean(np.abs(sizes)))
    return (class_probabilities, class_intervals, class_initial_val, 
            compressed_file_size, train_data_compressed_file_size, avg_class_probabilities)

 
#### this will be faster ###
def GLS_CODING_FIT_SECOND(X_train_norm, y_train, threshold):
    """
    Fit the GLS Coding classifier using the second return map approach.
    
    For each class, this function computes the probabilities for each sample,
    then aggregates these by computing the average probability for each pair 
    ((0,0), (0,1), (1,1), (1,0)) across all samples of that class.
    
    Using these average probability values and a representative symbolic 
    sequence (the first sample of that class), it computes the back iteration 
    interval and from that the compressed file size.
    
    Parameters
    ----------
    X_train_norm : numpy array
        The normalized training features.
    y_train : numpy array
        The training labels.
    threshold : float
        The threshold used for binarization.
    
    Returns
    -------
    class_probabilities : dict
        For each class, the list of probability dictionaries for each sample.
    class_intervals : dict
        For each class, the list of intervals computed per sample.
    class_initial_val : dict
        For each class, the list of estimated initial values (mean of interval) per sample.
    compressed_file_size : dict
        For each class, the array of compressed file sizes (absolute values) computed per sample.
    train_data_compressed_file_size : list
        The list of average compressed file sizes for each class (computed from the sample sizes).
    avg_class_probabilities : dict
        For each class, a dictionary with the average probability for each pair:
        {(0,0): avg_p00, (0,1): avg_p01, (1,1): avg_p11, (1,0): avg_p10}.
    """
    # Binarize training data
    X_train_bin = (X_train_norm >= threshold).astype(int)
    n_classes = len(np.unique(y_train))
    
    # Initialize dictionaries/lists to store results per class.

    avg_class_probabilities = {}
    
    for class_label in range(n_classes):
        # Select samples for the current class.
        class_data = X_train_bin[y_train.flatten() == class_label]
        probabilities_list = []

        
        for row in class_data:
            # Compute the pair probabilities for this sample.
            probs = compute_pair_probabilities(row)
            probabilities_list.append(probs)
            
     
            
        
        # Compute average probabilities across all samples for this class.
        avg_p00 = np.mean([prob[(0,0)] for prob in probabilities_list])
        avg_p01 = np.mean([prob[(0,1)] for prob in probabilities_list])
        avg_p11 = np.mean([prob[(1,1)] for prob in probabilities_list])
        avg_p10 = np.mean([prob[(1,0)] for prob in probabilities_list])
        avg_class_probabilities[class_label] = {(0,0): avg_p00, (0,1): avg_p01, (1,1): avg_p11, (1,0): avg_p10}
        
      
    
    # Store per-sample results for the class.
  
    return avg_class_probabilities


       
    
def GLS_CODING_PREDICT_SECOND_AVG(X_test_norm, n_classes, avg_class_probabilities, threshold):
    """
    Predict class labels for test data using the average class probabilities computed during training
    with the second return map approach.

    For each test instance, the function uses the average probability values for each class to perform
    back iteration using skew_tent_map_back_iter, computes the compressed file size, and then assigns the
    class corresponding to the smallest compressed file size.

    Parameters
    ----------
    X_test_norm : numpy array
        Normalized test features.
    n_classes : int
        Number of classes.
    avg_class_probabilities : dict
        Dictionary where each key is a class label and each value is a dictionary containing the average 
        probability values for each pair:
            {(0,0): avg_p00, (0,1): avg_p01, (1,1): avg_p11, (1,0): avg_p10}
    threshold : float
        Threshold used for binarization.

    Returns
    -------
    predicted_classes : numpy array
        Array of predicted class labels for each test instance.
    """
    # Binarize the test data using the provided threshold
    X_test_bin = (X_test_norm >= threshold).astype(int)
    # Initialize an array to store the computed compressed file sizes for each test instance and class.
    test_compressed_sizes = np.zeros((X_test_bin.shape[0], n_classes))
    
    # Process each test instance
    for i, test_row in enumerate(X_test_bin):
        for class_label in range(n_classes):
            # Retrieve the average probability dictionary for the current class
            avg_probs = avg_class_probabilities[class_label]
            p00 = avg_probs[(0,0)]
            p01 = avg_probs[(0,1)]
            p11 = avg_probs[(1,1)]
            p10 = avg_probs[(1,0)]
            # Compute the back-iteration interval using the average probabilities
            _, interval = skew_tent_map_back_iter(p00, p01, p11, p10, test_row)
            # Compute the compressed file size from the obtained interval
            size = compressed_size_of_interval(interval)
            test_compressed_sizes[i, class_label] = size

    # For each test instance, choose the class with the smallest compressed file size
    predicted_classes = np.argmin(test_compressed_sizes, axis=1)
    return predicted_classes
