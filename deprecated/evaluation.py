# evaluation.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
)

from .losses import beta_vae_loss

def evaluate_binary_vae(model, test_loader, beta=1.0, device='cpu'):
    """
    Evaluate BinaryVariationalAutoencoder on test set
    
    Returns:
        dict: Metrics including loss, recon_loss, kl_loss
    """
    
    model.eval()
    model = model.to(device)
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = len(test_loader)
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            p_x, q_z, z = model(x)
            loss, recon, kl = beta_vae_loss(x, p_x, q_z, beta)
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches
    }


def evaluate_binary_classifier(model, test_loader, device='cpu'):
    """
    Evaluate BinaryMLPClassifier on test set
    
    Returns:
        dict: Metrics including loss, accuracy, precision, recall, f1
    """
    
    model.eval()
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    n_batches = len(test_loader)
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            
            if y.ndim == 1:
                y = y.unsqueeze(1).float()
            else:
                y = y.float()
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    return {
        'loss': total_loss / n_batches,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }