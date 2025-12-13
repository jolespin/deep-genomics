import torch

def binary_confusion_matrix(predictions, targets, threshold=0.5):
    """
    Compute confusion matrix metrics for binary predictions.
    
    Args:
        predictions: Predicted probabilities or logits, shape [batch_size, n_features]
                    If logits, will be converted to probabilities via sigmoid
        targets: True binary values (0 or 1), shape [batch_size, n_features]
        threshold: Threshold for converting probabilities to binary predictions (default: 0.5)
    
    Returns:
        dict: Dictionary containing:
            - 'tp': True positives (count)
            - 'fp': False positives (count)
            - 'tn': True negatives (count)
            - 'fn': False negatives (count)
            - 'precision': TP / (TP + FP), handles division by zero
            - 'recall': TP / (TP + FN), handles division by zero
            - 'f1': 2 * (precision * recall) / (precision + recall)
    
    Note: All metrics are computed across all features (flattened view)
    """
    # Convert logits to probabilities if needed (predictions > 1 suggests logits)
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    # Threshold predictions to binary
    pred_binary = (predictions >= threshold).float()
    
    # Flatten for easier computation
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # Compute confusion matrix components
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().float()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().float()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().float()
    
    # Compute metrics with zero-division handling
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }