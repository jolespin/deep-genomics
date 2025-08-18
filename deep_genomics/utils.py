# utils.py
import os
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
import torch
import hashlib

def convert_sequence_to_md5(sequence, case_sensitive=False):
    """
    Convert a nucleotide sequence to its MD5 hash.
    
    Parameters:
    sequence (str): The nucleotide sequence to convert.
    
    Returns:
    str: The MD5 hash of the sequence.
    """
    sequence = sequence.replace(" ", "")
    if not case_sensitive:
        sequence = sequence.upper()
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()

# Utility functions
def set_seed(seed=42):
    """Set random seeds for reproducible results"""
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    
    # PyTorch CUDA random (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # PyTorch MPS random (if using MPS)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make PyTorch operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")

# Device detection
def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")