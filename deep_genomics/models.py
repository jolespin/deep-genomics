from pathlib import Path
import json
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions import (
    Normal, 
    Bernoulli,
    kl_divergence,
)
from .utils import get_activation_fn



class BinaryMLPClassifier(nn.Module):
    def __init__(
        self, 
        input_dim: int,  
        hidden_dims: list = [512, 256, 128], 
        dropout_rates: list = [0.3, 0.3, 0.2], 
        activation_fn = nn.ReLU,
    ):
        """
        Binary MLP classifier with configurable architecture
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes (default: [512, 256, 128])
            dropout_rates: List of dropout rates for each hidden layer (default: [0.3, 0.3, 0.2])
            activation_fn: Activation function class (default: nn.ReLU)
        """
        super().__init__()
        
        
        # Store config
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.activation_fn = activation_fn
        
        # Validate dimensions match
        assert len(hidden_dims) == len(dropout_rates), \
            f"hidden_dims and dropout_rates must have same length, got {len(hidden_dims)} and {len(dropout_rates)}"
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (logits)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass returning logits
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        
        Returns:
            Logits of shape [batch_size, 1]
        """
        return self.model(x)
    
    def predict_proba(self, x):
        """
        Predict probabilities (deterministic)
        
        Args:
            x: Input tensor of shape [input_dim] or [batch_size, input_dim]
        
        Returns:
            Probabilities of shape [1] or [batch_size, 1]
        """
        self.eval()
        
        # Check if single sample (1D) - add batch dimension
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        
        # Remove batch dimension if input was single sample
        if squeeze_output:
            probs = probs.squeeze(0)
        
        return probs
    
    def predict(self, x, threshold=0.5):
        """
        Predict binary class labels (deterministic)
        
        Args:
            x: Input tensor of shape [input_dim] or [batch_size, input_dim]
            threshold: Classification threshold (default: 0.5)
        
        Returns:
            Binary predictions of shape [1] or [batch_size, 1]
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()
    
    def save_pretrained(self, save_directory):
        """Save model weights and config in HuggingFace format"""
        
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rates': self.dropout_rates,
            'activation_fn': self.activation_fn.__name__,
            'model_type': self.__class__.__name__
        }
        
        config_path = save_directory / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save weights
        weights_path = save_directory / 'pytorch_model.bin'
        torch.save(self.state_dict(), weights_path)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, map_location=None):
        """
        Load model from HuggingFace format
        
        Args:
            pretrained_model_path: Path to directory containing config.json and pytorch_model.bin
            map_location: Device to load model weights (e.g., 'cpu', 'cuda'). 
                         If None, uses default torch loading behavior
        
        Returns:
            Loaded model instance
        """
        pretrained_model_path = Path(pretrained_model_path)
        
        # Validate paths exist
        config_path = pretrained_model_path / 'config.json'
        weights_path = pretrained_model_path / 'pytorch_model.bin'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate model type
        if config.get('model_type') != cls.__name__:
            logger.warning(
                f"Loading {config.get('model_type')} config into {cls.__name__}. "
                "This may fail if architectures don't match."
            )
        
        # Map activation function name back to class
        config['activation_fn'] = get_activation_fn(config['activation_fn'])
        
        # Create model
        model = cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            dropout_rates=config['dropout_rates'],
            activation_fn=config['activation_fn']
        )
        
        # Load weights
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {pretrained_model_path}")
        
        return model

class BinaryVariationalAutoencoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: list,
            latent_dim: int,
            activation_fn = nn.ReLU
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        
        # Build encoder with progressive compression
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Encoder heads
        last_hidden_dim = hidden_dims[-1]
        self.fc_mu = nn.Linear(last_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_hidden_dim, latent_dim)

        # Build decoder - mirror encoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn()
            ])
            prev_dim = hidden_dim
        
        # Final layer outputs logits (not probabilities)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        logits = self.decoder(z)
        return logits
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encode(x)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        # q_z is the approximate posterior - q(z|x)
        q_z = Normal(mu, std)
        z = q_z.rsample()

        # Decode
        logits = self.decode(z)
        # p_x is the likelihood - p(x|z)
        p_x = Bernoulli(logits=logits)

        return p_x, q_z, z
    
    def sample(self, n_samples:int, device=None):
        """
        Generate samples from the prior p(z) = N(0, I)
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            # Sample from distribution
            z = torch.randn(n_samples, self.latent_dim, device=device)
        
            # Decode to get reconstructions
            logits = self.decode(z)
            samples = torch.sigmoid(logits)
            
        return samples
    
    # Reconstruct
    def reconstruct(self, x, device=None, return_cpu=True):
        """
        Reconstruct input using posterior mean (deterministic)

        Args:
            x: DataLoader, or tensor of shape [n_features] or [batch_size, n_features]
            device: Device to move tensors to (e.g., 'cpu', 'mps', 'cuda'). If None, uses model's current device
            return_cpu: Return output on CPU
        Returns:
            Reconstruction(s)
        """
        if device is None:
            device = next(self.parameters()).device

        if isinstance(x, DataLoader):
            return self._reconstruct_dataloader(x, device=device, return_cpu=return_cpu)
        else:
            return self._reconstruct_tensor(x, device=device, return_cpu=return_cpu)

    def _reconstruct_dataloader(self, dataloader, device, return_cpu):
        """Reconstruct via DataLoader (memory efficient)"""
        self.eval()
        
        reconstructions = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                mu, _ = self.encode(x)
                logits = self.decode(mu)
                x_recon = torch.sigmoid(logits)
                reconstructions.append(x_recon.cpu() if return_cpu else x_recon)
        
        return torch.cat(reconstructions, dim=0)

    def _reconstruct_tensor(self, x, device, return_cpu):
        """Reconstruct single tensor (convenience method)"""
        self.eval()
        
        squeeze_output = x.ndim == 1
        if squeeze_output:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            x = x.to(device)
            mu, _ = self.encode(x)
            logits = self.decode(mu)
            x_recon = torch.sigmoid(logits)
        
        if squeeze_output:
            x_recon = x_recon.squeeze(0)
        
        if return_cpu:
            x_recon = x_recon.cpu()
        
        return x_recon
    
    # def reconstruct(self, x):
    #     """
    #     Reconstruct input using posterior mean (deterministic)

    #     Args:
    #         x: Input tensor of shape [n_features] or [batch_size, n_features]

    #     Returns:
    #         Reconstruction of shape [n_features] or [batch_size, n_features]
    #     """
    #     self.eval()

    #     # Check if single sample (1D) - add batch dimension
    #     if x.ndim == 1:
    #         x = x.unsqueeze(0)
    #         squeeze_output = True
    #     else:
    #         squeeze_output = False

    #     with torch.no_grad():
    #         # Encode: get posterior mean (deterministic latent representation)
    #         mu, logvar = self.encode(x)
    #         # Decode: get logits
    #         logits = self.decode(mu)
    #         # Get reconstruction mean
    #         x_recon = torch.sigmoid(logits)
        
    #     # Remove batch dimension if input was single sample
    #     if squeeze_output:
    #         x_recon = x_recon.squeeze(0)

    #     return x_recon
        
    # Transform
    def transform(self, x, device=None, return_cpu=True):
        """
        Transform input to latent representation (deterministic)
        
        Args:
            x: DataLoader, or tensor of shape [n_features] or [batch_size, n_features]
            device: Device to move tensors to (e.g., 'cpu', 'mps', 'cuda'). If None, uses model's current device
            return_cpu: Return output on CPU
        
        Returns:
            Latent representation(s) of shape [latent_dim] or [batch_size, latent_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        if isinstance(x, DataLoader):
            return self._transform_dataloader(x, device=device, return_cpu=return_cpu)
        else:
            return self._transform_tensor(x, device=device, return_cpu=return_cpu)

    def _transform_dataloader(self, dataloader, device, return_cpu):
        """Transform via DataLoader (memory efficient)"""
        self.eval()
        
        latent_codes = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                mu, _ = self.encode(x)
                latent_codes.append(mu.cpu() if return_cpu else mu)
        
        return torch.cat(latent_codes, dim=0)

    def _transform_tensor(self, x, device, return_cpu):
        """Transform single tensor (convenience method)"""
        self.eval()
        
        squeeze_output = x.ndim == 1
        if squeeze_output:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            x = x.to(device)
            mu, _ = self.encode(x)
        
        if squeeze_output:
            mu = mu.squeeze(0)
        
        if return_cpu:
            mu = mu.cpu()
        
        return mu
    
# def transform(self, x):
    #     """
    #     Transform input to latent representation (deterministic)
        
    #     Args:
    #         x: Input tensor of shape [n_features] or [batch_size, n_features]
        
    #     Returns:
    #         Latent representation of shape [latent_dim] or [batch_size, latent_dim]
    #     """
    #     self.eval()
        
    #     # Check if single sample (1D) - add batch dimension
    #     if x.ndim == 1:
    #         x = x.unsqueeze(0)
    #         squeeze_output = True
    #     else:
    #         squeeze_output = False
        
    #     with torch.no_grad():
    #         mu, _ = self.encode(x)
        
    #     # Remove batch dimension if input was single sample
    #     if squeeze_output:
    #         mu = mu.squeeze(0)
        
    #     return mu

    # HuggingFace Support
    def save_pretrained(self, save_directory):
        """Save model weights and config in HuggingFace format"""
        
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'latent_dim': self.latent_dim,
            'activation_fn': self.activation_fn.__name__,
            'model_type': self.__class__.__name__
        }
        
        config_path = save_directory / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save weights
        weights_path = save_directory / 'pytorch_model.bin'
        torch.save(self.state_dict(), weights_path)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, map_location=None):
        """
        Load model from HuggingFace format
        
        Args:
            pretrained_model_path: Path to directory containing config.json and pytorch_model.bin
            map_location: Device to load model weights (e.g., 'cpu', 'cuda'). 
                         If None, uses default torch loading behavior
        
        Returns:
            Loaded model instance
        """
        pretrained_model_path = Path(pretrained_model_path)
        
        # Validate paths exist
        config_path = pretrained_model_path / 'config.json'
        weights_path = pretrained_model_path / 'pytorch_model.bin'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate model type
        if config.get('model_type') != cls.__name__:
            logger.warning(
                f"Loading {config.get('model_type')} config into {cls.__name__}. "
                "This may fail if architectures don't match."
            )
        
        # Map activation function name back to class
        config['activation_fn'] = get_activation_fn(config['activation_fn'])
        
        # Create model
        model = cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim'],
            activation_fn=config['activation_fn']
        )
        
        # Load weights
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {pretrained_model_path}")
        
        return model
    
