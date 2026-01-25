import numpy as np
import torch
from tqdm import tqdm

def get_transition_matrix(dataset, n_classes, epsilon=1e-6):
    """
    Computes the transition matrix A[i, j] = P(state_j | state_i)
    by counting transitions in the training dataset.
    """
    print("Computing HMM Transition Matrix from dataset...")
    
    # Initialize matrix of counts
    trans_counts = np.zeros((n_classes, n_classes))
    
    # Iterate over the dataset to count transitions
    # We use the base dataset (full tracks), not the segmented one, 
    # to capture long-term transitions.
    for i in tqdm(range(len(dataset))):
        # dataset[i] returns (audio, frame_labels)
        _, labels = dataset[i]
        labels = labels.numpy()
        
        # We only care about valid transitions (ignore -100/padding if any)
        # Assuming your dataset returns valid class indices for frames
        for t in range(len(labels) - 1):
            curr_state = labels[t]
            next_state = labels[t+1]
            
            # bounds check
            if 0 <= curr_state < n_classes and 0 <= next_state < n_classes:
                trans_counts[curr_state, next_state] += 1

    # Normalize rows to get probabilities: P(next | current)
    # Add epsilon to avoid log(0)
    trans_counts = trans_counts + epsilon
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    transition_matrix = trans_counts / row_sums
    
    return transition_matrix

def make_transition_matrix_sticky(matrix, self_prob=0.99):
    """
    Overwrites the diagonal of the transition matrix to enforce stability.
    
    Args:
        matrix: The original transition matrix (n_classes, n_classes)
        self_prob: Probability of staying in the same chord (0.95 to 0.999)
    """
    n = matrix.shape[0]
    new_matrix = np.zeros_like(matrix)
    
    # 1. Fill off-diagonals (probability of changing)
    # We distribute (1 - self_prob) among all other n-1 classes
    off_diagonal_prob = (1.0 - self_prob) / (n - 1)
    new_matrix.fill(off_diagonal_prob)
    
    # 2. Fill diagonal (probability of staying)
    np.fill_diagonal(new_matrix, self_prob)
    
    # (Optional) You can blend this with your learned matrix if you want
    # combined = 0.5 * matrix + 0.5 * new_matrix
    
    return new_matrix