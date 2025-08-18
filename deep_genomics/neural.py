# neural.py
import pickle
import gzip
import os
import random
import pandas as pd
import anndata as ad
import numpy as np
from tqdm import tqdm
from loguru import logger
from collections import (
    Counter,
    OrderedDict,
)
from typing import Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    DataLoader, 
    TensorDataset, 
    WeightedRandomSampler,
)

import wandb

from scipy.stats import (
    mannwhitneyu,
    false_discovery_control,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve, 
    precision_recall_curve, 
    auc,
)

from .utils import *

# Model
class BinaryMLPClassifier(nn.Module):
    def __init__(self, 
                 n_features: int,  
                 hidden_sizes=[512, 256, 128], 
                 dropout_rates=[0.3, 0.3, 0.2], 
                 activation_fn=nn.ReLU,  
                 seed=0, 
                 return_logits=False,
                 **activation_kwargs,
                 ):
        """
        Binary classifier with configurable architecture
        
        Args:
            n_features: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rates: List of dropout rates for each hidden layer
            activation_fn: Activation function to use (default: ReLU)
            seed: Random seed for weight initialization
            return_logits: Whether to return logits or probabilities
            **activation_kwargs: Keyword arguments for activation function
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.n_features = n_features
        
        # Build layers dynamically
        layers = []
        input_size = n_features
        
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                activation_fn(**activation_kwargs),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(
            nn.Linear(input_size, 1),
        )
        
        if not return_logits:
            layers.append(
                nn.Sigmoid()
            )
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
# Loss
class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor ('val_loss', 'val_accuracy', 'val_sensitivity')
            mode: 'min' for loss, 'max' for accuracy/sensitivity
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda current, best: current < (best - min_delta)
        else:
            self.best = -float('inf')
            self.monitor_op = lambda current, best: current > (best + min_delta)
    
    def __call__(self, current_value, epoch):
        """
        Returns True if training should stop
        """
        if self.monitor_op(current_value, self.best):
            self.best = current_value
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

def get_sample_weights(
    y:Union[pd.Series, np.ndarray, torch.Tensor],
    smoothing_factor: float = 0.0,
    ) -> tuple:
    """
    Use this insted? https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    
    Calculate sample weights for imbalanced classes
    
    Args:
        y: Target labels [n_samples,]
    
    Returns:
        sample_to_weight: Weight for each sample
        class_to_weight: Weight for each class
        class_to_count: Number of samples in each class
    """
    index = None
    if isinstance(y, pd.Series):
        index = y.index
        y = y.values
    elif isinstance(y, torch.Tensor):
        y = y.numpy().flatten()
    
    y_np = y.flatten()
    
    # Count class occurrences
    class_to_count = Counter(y)
    number_of_samples = len(y)
    
    # Calculate class weights (inverse frequency)
    class_to_weight = dict()
    for class_label, count in class_to_count.items():
        adjusted_count = count + smoothing_factor
        class_size = len(class_to_count)
        inverted_frequency_smoothed = number_of_samples / (class_size * adjusted_count)
        class_to_weight[class_label] = inverted_frequency_smoothed
    
    # Assign weight to each sample based on its class
    sample_to_weight = np.array([class_to_weight[label] for label in y])
    
    logger.info(f"Class distribution: {dict(class_to_count)}")
    logger.info(f"Class weights: {class_to_weight}")
    logger.info(f"Sample weights range: {sample_to_weight.min():.4f} - {sample_to_weight.max():.4f}")
    # if index is not None:
    #     sample_to_weight = pd.Series(sample_to_weight, index=index)
    #     class_to_weight = pd.Series(class_to_weight)
    return sample_to_weight, class_to_weight, class_to_count



def get_data_loader(
    X:Union[pd.DataFrame, np.ndarray, torch.Tensor],
    y:Union[pd.Series, np.ndarray, torch.Tensor],
    batch_size:int, 
    shuffle=True, 
    num_workers=1,    # Use multiple CPU cores
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    device=None, 
    use_weighted_sampling=True,
    smoothing_factor=0, 
    sampling_strategy='balanced', 
    samples_per_epoch=None,
    **data_loader_kwargs,
    ):
    """
    Create a DataLoader with weighted sampling for imbalanced classes
    
    Args:
        X: Input features
        y: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if using weighted sampling)
        device: Device to move data to
        use_weighted_sampling: Whether to use WeightedRandomSampler
        smoothing_factor: Smoothing parameter for weight calculation
        sampling_strategy: 'original', 'balanced', 'oversample', or 'custom'
        samples_per_epoch: Custom number of samples per epoch (for 'custom' strategy)
    """
    _dataloader_kwargs = dict(
        num_workers=num_workers,    # Use multiple CPU cores
        pin_memory=pin_memory,  # Faster GPU transfer
        persistent_workers=persistent_workers,  # Keep workers alive
        batch_size=batch_size, 
        shuffle=shuffle,
    )
    if data_loader_kwargs:
        _dataloader_kwargs.update(data_loader_kwargs)
    
    # Convert to tensors if they aren't already
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y).unsqueeze(1)  # Add dimension for binary classification
    
    # Move to device if specified
    if device is not None:
        X = X.to(device)
        y = y.to(device)
    
    dataset = TensorDataset(X, y)
    
    if use_weighted_sampling:
        # Calculate sample weights for balanced sampling (with smoothing)
        sample_to_weights, class_to_weights, class_to_count = get_sample_weights(y, smoothing_factor=smoothing_factor)
        
        # Determine number of samples per epoch based on strategy
        original_size = len(sample_to_weights)
        
        if sampling_strategy == 'original':
            # Same as original dataset size
            sampling_size = original_size
        
        elif sampling_strategy == 'balanced':
            # Sample enough to see each class equally (recommended default)
            max_class_size = max(class_to_count.values())
            number_of_classes = len(class_to_count)
            sampling_size = max_class_size * number_of_classes
            
        elif sampling_strategy == 'oversample':
            # 2x original size for more diversity
            sampling_size = original_size * 2
            
        elif sampling_strategy == 'custom':
            # User-specified number
            sampling_size = samples_per_epoch if samples_per_epoch else original_size
            
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
        
        logger.info(f"Sampling strategy: {sampling_strategy}")
        logger.info(f"Original dataset size: {original_size}")
        logger.info(f"Samples per epoch: {sampling_size}")
        logger.info(f"Epoch size ratio: {sampling_size/original_size:.2f}x")
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_to_weights,
            num_samples=sampling_size,
            replacement=True
        )
        del _dataloader_kwargs["shuffle"]  # Remove shuffle since sampler handles it
        return DataLoader(dataset, sampler=sampler, **_dataloader_kwargs)
    else:
        return DataLoader(dataset, **_dataloader_kwargs)

def confusion_matrix(y_true, y_pred, threshold=0.5, eps=1e-8):
    """
    Calculate confusion matrix components using PyTorch
    y_true: True labels (1D tensor)
    y_pred: Predicted probabilities (1D tensor)
    threshold: Threshold to convert probabilities to binary predictions
    eps: Small value to avoid division by zero
    
    Returns:
        Dictionary with confusion matrix components and metrics
        {'tp': True Positives, 'tn': True Negatives, 'fp': False Positives, 'fn': False Negatives,
         'sensitivity': Recall, 'specificity': True Negative Rate, 
         'precision': Positive Predictive Value, 'accuracy': Overall Accuracy}
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_pred_binary = (y_pred >= threshold).float()
    
    # Calculate all confusion matrix components
    tp = torch.sum((y_true == 1) & (y_pred_binary == 1)).float()
    tn = torch.sum((y_true == 0) & (y_pred_binary == 0)).float()
    fp = torch.sum((y_true == 0) & (y_pred_binary == 1)).float()
    fn = torch.sum((y_true == 1) & (y_pred_binary == 0)).float()
    
    # Calculate all metrics
    sensitivity = tp / (tp + fn + eps)  # Recall, True Positive Rate
    specificity = tn / (tn + fp + eps)  # True Negative Rate
    precision = tp / (tp + fp + eps)    # Positive Predictive Value
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        'tp': tp.item(), 'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(), 
        'precision': precision.item(),
        'accuracy': accuracy.item()
    }

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    epochs=100, 
    device=None,
    log_frequency=20,
    use_wandb=True,
    config=None,
    early_stopping=None,  # EarlyStopping object or None
):
    """
    Train model with optional wandb logging and early stopping
    
    Args:
        early_stopping: EarlyStopping instance or None to disable early stopping
    """
    if device is None:
        device = get_device()
    
    if use_wandb and not wandb.run:
        raise RuntimeError("wandb.init() must be called before using use_wandb=True")
    
    # Initialize best model tracking
    best_model_state = None
    best_epoch = 0
    
    if use_wandb:
        wandb_config = {
            "epochs": epochs,
            "device": str(device),
            "model_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "optimizer": type(optimizer).__name__,
            "criterion": type(criterion).__name__,
        }
        
        if early_stopping is not None:
            wandb_config.update({
                "early_stopping_patience": early_stopping.patience,
                "early_stopping_min_delta": early_stopping.min_delta,
                "monitor_metric": early_stopping.monitor
            })
        
        wandb.config.update(wandb_config)
        if config:
            wandb.config.update(config)
        logger.info(f"WandB run: {wandb.run.name}")
    
    logger.info(f"Using device: {device}")
    if early_stopping is not None:
        logger.info(f"Early stopping enabled: patience={early_stopping.patience}, monitoring {early_stopping.monitor}")
    
    model = model.to(device)
    
    # Store metrics for history tracking and final results
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'val_precision': []
    }
    final_train_metrics = None
    final_val_metrics = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_logits = []
        all_train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Always collect training predictions for metrics calculation
            all_train_logits.append(outputs.detach().cpu())
            all_train_targets.append(batch_y.cpu())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_logits = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                all_val_logits.append(outputs.cpu())
                all_val_targets.append(batch_y.cpu())
        
        # Calculate training metrics for current epoch
        all_train_logits_cat = torch.cat(all_train_logits).flatten()
        all_train_targets_cat = torch.cat(all_train_targets).flatten()
        all_train_probs = torch.sigmoid(all_train_logits_cat)
        train_metrics = confusion_matrix(all_train_targets_cat, all_train_probs)
        
        # Calculate validation metrics
        all_val_logits = torch.cat(all_val_logits).flatten()
        all_val_targets = torch.cat(all_val_targets).flatten()
        all_val_probs = torch.sigmoid(all_val_logits)
        val_metrics = confusion_matrix(all_val_targets, all_val_probs)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics in history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_sensitivity'].append(val_metrics['sensitivity'])
        training_history['val_specificity'].append(val_metrics['specificity'])
        training_history['val_precision'].append(val_metrics['precision'])
        
        # Determine metric value for early stopping
        if early_stopping is not None:
            if early_stopping.monitor == 'val_loss':
                monitor_value = avg_val_loss
            elif early_stopping.monitor == 'val_accuracy':
                monitor_value = val_metrics['accuracy']
            elif early_stopping.monitor == 'val_sensitivity':
                monitor_value = val_metrics['sensitivity']
            elif early_stopping.monitor == 'val_specificity':
                monitor_value = val_metrics['specificity']
            elif early_stopping.monitor == 'val_precision':
                monitor_value = val_metrics['precision']
            else:
                monitor_value = avg_val_loss
            
            # Check for improvement and save best model
            if early_stopping.monitor_op(monitor_value, early_stopping.best):
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                # Store best epoch metrics
                final_train_metrics = train_metrics.copy()
                final_train_metrics['loss'] = avg_train_loss
                final_val_metrics = val_metrics.copy()
                final_val_metrics['loss'] = avg_val_loss
        else:
            # No early stopping - always update final metrics (last epoch is best)
            final_train_metrics = train_metrics.copy()
            final_train_metrics['loss'] = avg_train_loss
            final_val_metrics = val_metrics.copy()
            final_val_metrics['loss'] = avg_val_loss
            best_epoch = epoch
        
        # Log to wandb
        if use_wandb:
            wandb_log = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_metrics['accuracy'],
                "val_sensitivity": val_metrics['sensitivity'],
                "val_specificity": val_metrics['specificity'],
                "val_precision": val_metrics['precision'],
            }
            
            if early_stopping is not None:
                wandb_log.update({
                    "best_epoch": best_epoch + 1,
                    "patience_counter": early_stopping.wait,
                    f"best_{early_stopping.monitor}": early_stopping.best if early_stopping.best != float('inf') and early_stopping.best != -float('inf') else None
                })
            
            wandb.log(wandb_log)
        
        # Console logging
        if (epoch + 1) % log_frequency == 0:
            log_message = (
                f'Epoch [{epoch+1}/{epochs}] - '
                f'Loss[Train]: {avg_train_loss:.4f}, '
                f'Loss[Val]: {avg_val_loss:.4f}, '
                f'Accuracy[Val]: {val_metrics["accuracy"]:.4f}, '
                f'Sensitivity[Val]: {val_metrics["sensitivity"]:.4f}, '
                f'Specificity[Val]: {val_metrics["specificity"]:.4f}'
            )
            
            if early_stopping is not None:
                log_message += f', Patience: {early_stopping.wait}/{early_stopping.patience}'
            
            logger.info(log_message)
        
        # Check early stopping
        if early_stopping is not None and early_stopping(monitor_value, epoch):
            logger.info(f'Early stopping triggered at epoch {epoch + 1}')
            logger.info(f'Best {early_stopping.monitor}: {early_stopping.best:.4f} at epoch {best_epoch + 1}')
            break
    
    # Load best model if early stopping was used
    if early_stopping is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f'Loaded best model from epoch {best_epoch + 1}')
    
    # Create final summary table
    if use_wandb and final_train_metrics and final_val_metrics:
        summary_data = []
        
        # Training row
        summary_data.append([
            "Training",
            final_train_metrics['loss'],
            final_train_metrics['accuracy'],
            final_train_metrics['sensitivity'],
            final_train_metrics['specificity'],
            final_train_metrics['precision'],
            final_train_metrics['tp'],
            final_train_metrics['tn'],
            final_train_metrics['fp'],
            final_train_metrics['fn']
        ])
        
        # Validation row
        summary_data.append([
            "Validation",
            final_val_metrics['loss'],
            final_val_metrics['accuracy'],
            final_val_metrics['sensitivity'],
            final_val_metrics['specificity'],
            final_val_metrics['precision'],
            final_val_metrics['tp'],
            final_val_metrics['tn'],
            final_val_metrics['fp'],
            final_val_metrics['fn']
        ])
        
        # Create wandb table
        columns = ["Phase", "Loss", "Accuracy", "Sensitivity", "Specificity", "Precision", "TP", "TN", "FP", "FN"]
        training_summary_table = wandb.Table(data=summary_data, columns=columns)
        
        wandb.log({"Training_Summary": training_summary_table})
        logger.info("Training summary table logged to wandb")
    
    # Create summary for both wandb and non-wandb usage
    results = {
        'training': final_train_metrics,
        'validation': final_val_metrics,
        'history': training_history,
        'best_epoch': best_epoch + 1,  # Convert to 1-indexed
        'total_epochs': epoch + 1,     # Actual epochs run
        'early_stopped': early_stopping is not None and early_stopping.stopped_epoch > 0
    }
    
    # Log final summary to wandb
    if use_wandb and early_stopping is not None:
        wandb.summary.update({
            "final_epoch": epoch + 1,
            "best_epoch": best_epoch + 1,
            "early_stopped": early_stopping.stopped_epoch > 0,
            f"best_{early_stopping.monitor}": early_stopping.best if early_stopping.best != float('inf') and early_stopping.best != -float('inf') else None
        })
    
    # Print summary regardless of wandb usage
    if not use_wandb:
        logger.info("=== TRAINING RESULTS ===")
        logger.info(f"Best epoch: {results['best_epoch']}")
        logger.info(f"Total epochs: {results['total_epochs']}")
        if results['early_stopped']:
            logger.info("Training stopped early")
        
        if final_train_metrics:
            logger.info("\n--- Training Metrics (Best Epoch) ---")
            logger.info(f"Loss: {final_train_metrics['loss']:.4f}")
            logger.info(f"Accuracy: {final_train_metrics['accuracy']:.4f}")
            logger.info(f"Sensitivity: {final_train_metrics['sensitivity']:.4f}")
            logger.info(f"Specificity: {final_train_metrics['specificity']:.4f}")
            logger.info(f"Precision: {final_train_metrics['precision']:.4f}")
        
        if final_val_metrics:
            logger.info("\n--- Validation Metrics (Best Epoch) ---")
            logger.info(f"Loss: {final_val_metrics['loss']:.4f}")
            logger.info(f"Accuracy: {final_val_metrics['accuracy']:.4f}")
            logger.info(f"Sensitivity: {final_val_metrics['sensitivity']:.4f}")
            logger.info(f"Specificity: {final_val_metrics['specificity']:.4f}")
            logger.info(f"Precision: {final_val_metrics['precision']:.4f}")
    else:
        # Also log summary for wandb users in console
        logger.info("=== TRAINING RESULTS ===")
        logger.info(f"Best epoch: {results['best_epoch']}")
        logger.info(f"Total epochs: {results['total_epochs']}")
        if results['early_stopped']:
            logger.info("Training stopped early")
    
    return results


def evaluate_model(
    model, 
    test_loader, 
    criterion=None,
    device=None, 
    use_wandb=True,
    log_plots=True
):
    """
    Evaluate model with optional wandb logging
    """
    if device is None:
        device = get_device()
    
    if use_wandb and not wandb.run:
        raise RuntimeError("wandb.init() must be called before using use_wandb=True")
    
    if use_wandb:
        logger.info(f"WandB run: {wandb.run.name}")
    
    logger.info(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    all_logits = []
    all_targets = []
    test_loss = 0.0
    
    # Collect predictions
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            if criterion is not None:
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
            
            all_logits.append(outputs.cpu())
            all_targets.append(batch_y.cpu())
    
    # Calculate metrics
    all_logits = torch.cat(all_logits).flatten()
    all_targets = torch.cat(all_targets).flatten()
    all_probs = torch.sigmoid(all_logits)  # Convert logits to probabilities
    metrics = confusion_matrix(all_targets, all_probs)
    
    # Convert to numpy for sklearn curves
    targets_np = all_targets.numpy()
    probs_np = all_probs.numpy()
    predictions_np = (all_probs >= 0.5).numpy()
    
    # Calculate curves
    results = {
        'test_loss': test_loss / len(test_loader) if criterion else None,
        'accuracy': metrics['accuracy'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'precision': metrics['precision'],
        'targets': targets_np,
        'probabilities': probs_np,
        'predictions': predictions_np,
        'confusion_matrix': metrics  # Include full confusion matrix
    }
    
    if log_plots:
        fpr, tpr, _ = roc_curve(targets_np, probs_np)
        precision_curve, recall, _ = precision_recall_curve(targets_np, probs_np)
        
        results.update({
            'roc_auc': auc(fpr, tpr),
            'pr_auc': auc(recall, precision_curve)
        })
        
        # Log to wandb
        if use_wandb:
            probs_2d = np.column_stack([1 - probs_np, probs_np])
            
            wandb.log({
                'test_accuracy': results['accuracy'],
                'test_sensitivity': results['sensitivity'],
                'test_specificity': results['specificity'],
                'test_precision': results['precision'],
                'test_roc_auc': results['roc_auc'],
                'test_pr_auc': results['pr_auc'],
                'test_roc_curve': wandb.plot.roc_curve(targets_np, probs_2d, labels=["Class 0", "Class 1"]),
                'test_pr_curve': wandb.plot.pr_curve(targets_np, probs_2d, labels=["Class 0", "Class 1"])
            })
    
    # Create evaluation summary table for wandb
    if use_wandb:
        eval_data = [[
            "Test",
            results['test_loss'] if results['test_loss'] else "N/A",
            metrics['accuracy'],
            metrics['sensitivity'],
            metrics['specificity'],
            metrics['precision'],
            metrics['tp'],
            metrics['tn'],
            metrics['fp'],
            metrics['fn']
        ]]
        
        if log_plots:
            eval_data[0].extend([results['roc_auc'], results['pr_auc']])
            columns = ["Phase", "Loss", "Accuracy", "Sensitivity", "Specificity", "Precision", "TP", "TN", "FP", "FN", "ROC_AUC", "PR_AUC"]
        else:
            columns = ["Phase", "Loss", "Accuracy", "Sensitivity", "Specificity", "Precision", "TP", "TN", "FP", "FN"]
        
        evaluation_summary_table = wandb.Table(data=eval_data, columns=columns)
        wandb.log({"Evaluation_Summary": evaluation_summary_table})
        logger.info("Evaluation summary table logged to wandb")
    
    # Console output
    logger.info("=== EVALUATION RESULTS ===")
    if results['test_loss']:
        logger.info(f"Test Loss: {results['test_loss']:.4f}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Sensitivity: {results['sensitivity']:.4f}")
    logger.info(f"Specificity: {results['specificity']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    if log_plots:
        logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
        logger.info(f"PR AUC: {results['pr_auc']:.4f}")
    logger.info(f"Confusion Matrix - TP: {metrics['tp']:.0f}, TN: {metrics['tn']:.0f}, FP: {metrics['fp']:.0f}, FN: {metrics['fn']:.0f}")
    
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(targets_np, results['predictions'], target_names=['Class 0', 'Class 1'])}")
    
    return results

def save_model_state(
    model, 
    filepath,
    optimizer=None, 
    epoch=None, 
    metrics=None, 
    config=None,
    ):
   """
   Save model weights and optional training states
   
   Args:
       model: PyTorch model
       optimizer: Optimizer (optional)
       epoch: Current epoch number (optional)
       metrics: Dict of final metrics like {'val_loss': 0.45, 'val_accuracy': 0.87} (optional)
       config: Dict of hyperparameters/config (optional)
       filepath: Where to save
   """
   save_dict = {
       'model_state_dict': model.state_dict(),
   }
   
   if optimizer is not None:
       save_dict['optimizer_state_dict'] = optimizer.state_dict()
   
   if epoch is not None:
       save_dict['epoch'] = epoch
   
   if metrics is not None:
       save_dict['metrics'] = metrics
       
   if config is not None:
       save_dict['config'] = config
   
   # Add timestamp for reference
   from datetime import datetime
   save_dict['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
   torch.save(save_dict, filepath)
   print(f"Model state saved to {filepath}")

def load_model_state(
    model, 
    filepath, 
    optimizer=None, 
    device=None,
    ):
   """
   Load model weights and optional training states
   
   Returns:
       tuple: (model, optimizer, epoch, metrics, config)
              None returned for components not saved/requested
   """
   checkpoint = torch.load(filepath, map_location=device)
   
   # Load model
   model.load_state_dict(checkpoint['model_state_dict'])
   
   # Load optimizer if provided and exists in checkpoint
   if optimizer is not None and 'optimizer_state_dict' in checkpoint:
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   
   # Extract other saved states
   epoch = checkpoint.get('epoch', None)
   metrics = checkpoint.get('metrics', None)
   config = checkpoint.get('config', None)
   saved_at = checkpoint.get('saved_at', 'Unknown')
   
   print(f"Model loaded from {filepath} (saved at: {saved_at})")
   if epoch is not None:
       print(f"Saved at epoch: {epoch}")
   if metrics is not None:
       print(f"Final metrics: {metrics}")
   
   return model, optimizer, epoch, metrics, config


