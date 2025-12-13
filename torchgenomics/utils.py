# utils.py
import os
import random
import numpy as np
from loguru import logger
import torch
import torch.nn as nn
import hashlib

def get_sequence_md5(sequence, case_sensitive=False):
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
def set_seed(seed=42, deterministic=False):
    """
    Set random seeds for reproducible results across multiple libraries.
    
    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, make PyTorch operations deterministic. This ensures
                      full reproducibility but may reduce performance (default: False)
    
    Note:
        For complete reproducibility, call this function before any model creation,
        data loading, or random operations. Some operations (like certain CUDA ops)
        may still be non-deterministic even with this function.
    """
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
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Warn about performance impact
        logger.warning(
            "Deterministic mode enabled. This ensures reproducibility but may reduce performance. "
            "Set deterministic=False if reproducibility is not critical."
        )
    
    # Set environment variable for complete reproducibility
    # Note: This only affects the current process if set before Python starts
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")

# Device detection
def get_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device
    
# Activation functions
ACTIVATION_MAP = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
    'GELU': nn.GELU
}

def get_activation_fn(name):
    """
    Get activation function class from string name
    
    Args:
        name: Activation function name (e.g., 'ReLU', 'GELU')
    
    Returns:
        Activation function class
    
    Raises:
        ValueError: If activation function name is not recognized
    """
    if name not in ACTIVATION_MAP:
        supported = ', '.join(ACTIVATION_MAP.keys())
        raise ValueError(
            f"Unknown activation function: '{name}'. "
            f"Supported activations: {supported}"
        )
    return ACTIVATION_MAP[name]