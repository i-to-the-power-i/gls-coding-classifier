import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class GLSCompressionClassifier:
    def __init__(self, n_classes, threshold=0.5, alpha=1, fast_mode=False):
        self.n_classes = n_classes
        self.threshold = threshold
        self.alpha = alpha
        self.fast_mode = fast_mode
        self.avg_class_probabilities = {}
        self.class_probabilities = {}
        self.class_intervals = {}
        self.class_initial_val = {}
        self.compressed_file_size = {}
        self.train_data_compressed_file_size = []

    def second_return_map_skew_tent(self, x, p_00, p_01, p_11, p_10):
        if x < p_00:
            return x / p_00
        elif (x >= p_00) and (x < p_00 + p_01):
            return (p_00 + p_01 - x) / p_01
        elif (x >= p_00 + p_01) and (x < p_00 + p_01 + p_11):
            return (x - (p_00 + p_01)) / p_11
        else:
            return (1 - x) / p_10

    def compute_pair_probabilities(self, data):
        # Generate non-overlapping pairs
        z = data
        pairs = [(z[i], z[i+1]) for i in range(0, len(z)-1, 2)]
        # Count occurrences of each pair
        pair_counts = Counter(pairs)
        total_pairs = sum(pair_counts.values())
        # Ensure all pairs are present
        all_pairs = [(0,0), (0,1), (1,1), (1,0)]
        probabilities = {pair: (pair_counts.get(pair, 0)) / total_pairs for pair in all_pairs}
        return probabilities

    def skew_tent_map_back_iter(self, p_00, p_01, p_11, p_10, data):
        SS = data  # Reverse sequence
        out = np.array([0, 0]).astype(float)

        pair_sequence = [(SS[i], SS[i+1]) for i in range(0, len(SS)-1, 2)]

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

    def compressed_size_of_interval(self, interval):
        return np.ceil(-np.log2(interval[1] - interval[0] + 1e-50))

    def fit(self, X_train_norm, y_train):
        X_train_bin = (X_train_norm >= self.threshold).astype(int)
        n_classes = len(np.unique(y_train))
        
        for class_label in range(n_classes):
            class_data = X_train_bin[y_train.flatten() == class_label]
            probabilities_list = []
            intervals_list = []
            initial_val_list = []
            sizes = []
            
            for row in class_data:
                probs = self.compute_pair_probabilities(row)
                probabilities_list.append(probs)
            
            avg_p00 = np.mean([prob[(0,0)] for prob in probabilities_list])
            avg_p01 = np.mean([prob[(0,1)] for prob in probabilities_list])
            avg_p11 = np.mean([prob[(1,1)] for prob in probabilities_list])
            avg_p10 = np.mean([prob[(1,0)] for prob in probabilities_list])

            if avg_p00 == 0 or avg_p01 == 0 or avg_p11 == 0 or avg_p10 == 0:
                print("Laplace Smoothing Done")
                
                avg_p00 = (np.sum([prob[(0,0)] for prob in probabilities_list]) + self.alpha) / (len(class_data) + 4 * self.alpha)
                avg_p01 = (np.sum([prob[(0,1)] for prob in probabilities_list]) + self.alpha) / (len(class_data) + 4 * self.alpha)
                avg_p11 = (np.sum([prob[(1,1)] for prob in probabilities_list]) + self.alpha) / (len(class_data) + 4 * self.alpha)
                avg_p10 = (np.sum([prob[(1,0)] for prob in probabilities_list]) + self.alpha) / (len(class_data) + 4 * self.alpha)

            self.avg_class_probabilities[class_label] = {(0,0): avg_p00, (0,1): avg_p01, (1,1): avg_p11, (1,0): avg_p10}

            if not self.fast_mode:
                for row in class_data:
                    initial_value, interval = self.skew_tent_map_back_iter(avg_p00, avg_p01, avg_p11, avg_p10, row)
                    intervals_list.append(interval)
                    initial_val_list.append(initial_value)
                    sizes.append(self.compressed_size_of_interval(interval))

                self.class_probabilities[class_label] = probabilities_list
                self.class_intervals[class_label] = intervals_list
                self.class_initial_val[class_label] = initial_val_list
                self.compressed_file_size[class_label] = np.abs(sizes)
                self.train_data_compressed_file_size.append(np.mean(np.abs(sizes)))
        
    def predict(self, X_test_norm):
        X_test_bin = (X_test_norm >= self.threshold).astype(int)
        n_classes = self.n_classes
        test_compressed_sizes = np.zeros((X_test_bin.shape[0], n_classes))

        for i, test_row in enumerate(X_test_bin):
            for class_label in range(n_classes):
                avg_probs = self.avg_class_probabilities[class_label]
                p00 = avg_probs[(0,0)]
                p01 = avg_probs[(0,1)]
                p11 = avg_probs[(1,1)]
                p10 = avg_probs[(1,0)]
                _, interval = self.skew_tent_map_back_iter(p00, p01, p11, p10, test_row)
                size = self.compressed_size_of_interval(interval)
                test_compressed_sizes[i, class_label] = size

        predicted_classes = np.argmin(test_compressed_sizes, axis=1)
        return predicted_classes
