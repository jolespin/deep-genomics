import numpy as np
import torch
import torch.nn as nn
from torch.distributions import (
    Normal, 
    Bernoulli,
)
import lightning as L

from ..utils import (
    get_activation_fn,
)
from ..training.losses import (
    beta_vae_loss, 
    beta_tcvae_loss,
)
from ..metrics import (
    binary_confusion_matrix,
)
from .base import BaseVAE

class VariationalEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: list,
            latent_dim: int,
            activation_fn = nn.ReLU,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn)
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

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
class VariationalDecoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: list,
            latent_dim: int,
            activation_fn = nn.ReLU,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn)
        self.activation_fn = activation_fn


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

    def forward(self, z):
        logits = self.decoder(z)
        return logits
    
class BinaryBetaVAE(BaseVAE):
    def __init__(
            self,
            # Architecture
            input_dim: int,
            hidden_dims: list,
            latent_dim: int,
            activation_fn = "ReLU",
            # Optimizer
            learning_rate:float = 1e-3,
            # Loss
            beta:float = 1.0,
            # Sub-models
            encoder: nn.Module = None,
            decoder: nn.Module = None,
            # Metadata
            experiment_name:str = None
            ) -> None:
        # Get activation function
        if not isinstance(activation_fn, str):
            activation_fn = activation_fn.__name__
        # Build encoder/decoder if not provided
        if all([
            encoder is None,
            decoder is None,
            ]):
            # Neither provided - build both
            if any([
                input_dim is None, 
                hidden_dims is None, 
                latent_dim is None,
                ]):
                raise ValueError(
                    "Must provide either (encoder, decoder) or "
                    "(input_dim, hidden_dims, latent_dim)"
                )
            encoder = VariationalEncoder(input_dim, hidden_dims, latent_dim, activation_fn)
            decoder = VariationalDecoder(input_dim, hidden_dims, latent_dim, activation_fn)
            
        elif encoder is None or decoder is None:
            # Only one provided - error
            raise ValueError("Must provide both encoder and decoder, or neither")
            
        else:
            # Both provided - validate they match
            if encoder.input_dim != decoder.input_dim:
                raise ValueError("Encoder and decoder input_dim must match")
            if list(encoder.hidden_dims) != list(decoder.hidden_dims):
                raise ValueError("Encoder and decoder hidden_dims must match")
            if encoder.latent_dim != decoder.latent_dim:
                raise ValueError("Encoder and decoder latent_dim must match")
            
            # Infer architecture from encoder
            input_dim = encoder.input_dim
            hidden_dims = encoder.hidden_dims
            latent_dim = encoder.latent_dim
        super().__init__(encoder, decoder, learning_rate)
        # Store archtecture metadata
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.beta = beta
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        if experiment_name is None:
            h = "-".join(map(str,hidden_dims))
            experiment_name = f"{self.__class__.__name__}__h{h}_z{latent_dim}_lr{learning_rate}_beta{beta}"
        self.experiment_name = experiment_name

    def encode(self, x: torch.Tensor):
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
    
    def forward(self, x):
        """Return (p_x, q_z, z) for loss computation"""
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
    
    def reconstruct(self, x, batch_size=2048, device=None, return_cpu=True):
        """
        Reconstruct input using posterior mean (deterministic)

        Args:
            x: Tensor or array of shape [n_samples, n_features] or [n_features]
            batch_size: Process in batches of this size for memory efficiency
            device: Device to move tensors to (e.g., 'cpu', 'mps', 'cuda'). If None, uses model's current device
            return_cpu: Return output on CPU
        
        Returns:
            Reconstruction(s) of shape [n_samples, n_features] or [n_features]
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Handle single sample
        squeeze_output = False
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True
        
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        
        n_samples = x.shape[0]
        reconstructions = []
        
        with torch.no_grad():
            # Process in batches for memory efficiency
            for i in range(0, n_samples, batch_size):
                batch = x[i:i+batch_size].to(device)
                mu, _ = self.encode(batch)
                logits = self.decode(mu)
                x_recon = torch.sigmoid(logits)
                reconstructions.append(x_recon.cpu() if return_cpu else x_recon)
        
        result = torch.cat(reconstructions, dim=0)
        
        # Remove batch dimension if input was single sample
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # Get data
        x = batch[0]

        # Forward pass
        p_x, q_z, z = self.forward(x)

        # Compute loss
        losses = beta_vae_loss(
            x=x,
            p_x=p_x,
            q_z=q_z,
            beta=self.beta,
        )

        # Compute reconstruction metrics
        x_recon = torch.sigmoid(p_x.logits)
        confusion_matrix = binary_confusion_matrix(x_recon, x, threshold=0.5)

        # Log loss
        self.log("train_loss", losses["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", losses["recon_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_kl_loss", losses["kl_loss"], on_step=False, on_epoch=True, prog_bar=True)
        # Log reconstruction metrics
        self.log('train_precision', confusion_matrix['precision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', confusion_matrix['recall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', confusion_matrix['f1'], on_step=False, on_epoch=True)

        # Return total loss for backpropagation
        return losses["total_loss"]
    
    def validation_step(self, batch, batch_idx):
        # Get data
        x = batch[0]

        # Forward pass
        p_x, q_z, z = self.forward(x)

        # Compute loss
        losses = beta_vae_loss(
            x=x,
            p_x=p_x,
            q_z=q_z,
            beta=self.beta,
        )

        # Compute reconstruction metrics
        x_recon = torch.sigmoid(p_x.logits)
        confusion_matrix = binary_confusion_matrix(x_recon, x, threshold=0.5)

        # Log metrics
        self.log("val_loss", losses["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", losses["recon_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kl_loss", losses["kl_loss"], on_step=False, on_epoch=True, prog_bar=True)
        # Log reconstruction metrics
        self.log('train_precision', confusion_matrix['precision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', confusion_matrix['recall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', confusion_matrix['f1'], on_step=False, on_epoch=True)
        
        # Return total loss for backpropagation
        return losses["total_loss"]
    
class BinaryBetaTCVAE(BaseVAE):
    def __init__(
            self,
            # Architecture
            input_dim: int,
            hidden_dims: list,
            latent_dim: int,
            activation_fn = "ReLU",
            # Optimizer
            learning_rate:float = 1e-3,
            # Loss
            beta:float = 1.0,
            alpha:float = 1.0,
            gamma:float = 1.0,
            # Sub-models
            encoder: nn.Module = None,
            decoder: nn.Module = None,
            # Metadata
            experiment_name:str = None
            ) -> None:
        # Get activation function
        if not isinstance(activation_fn, str):
            activation_fn = activation_fn.__name__
        # Build encoder/decoder if not provided
        if all([
            encoder is None,
            decoder is None,
            ]):
            # Neither provided - build both
            if any([
                input_dim is None, 
                hidden_dims is None, 
                latent_dim is None,
                ]):
                raise ValueError(
                    "Must provide either (encoder, decoder) or "
                    "(input_dim, hidden_dims, latent_dim)"
                )
            encoder = VariationalEncoder(input_dim, hidden_dims, latent_dim, activation_fn)
            decoder = VariationalDecoder(input_dim, hidden_dims, latent_dim, activation_fn)
            
        elif encoder is None or decoder is None:
            # Only one provided - error
            raise ValueError("Must provide both encoder and decoder, or neither")
            
        else:
            # Both provided - validate they match
            if encoder.input_dim != decoder.input_dim:
                raise ValueError("Encoder and decoder input_dim must match")
            if list(encoder.hidden_dims) != list(decoder.hidden_dims):
                raise ValueError("Encoder and decoder hidden_dims must match")
            if encoder.latent_dim != decoder.latent_dim:
                raise ValueError("Encoder and decoder latent_dim must match")
            
            # Infer architecture from encoder
            input_dim = encoder.input_dim
            hidden_dims = encoder.hidden_dims
            latent_dim = encoder.latent_dim
        super().__init__(encoder, decoder, learning_rate)
        # Store archtecture metadata
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        if experiment_name is None:
            h = "-".join(map(str,hidden_dims))
            experiment_name = f"{self.__class__.__name__}__h{h}_z{latent_dim}_lr{learning_rate}_beta{beta}_alpha{alpha}_gamma{gamma}"
        self.experiment_name = experiment_name

    def encode(self, x: torch.Tensor):
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
    
    def forward(self, x):
        """Return (p_x, q_z, z) for loss computation"""
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
    
    def reconstruct(self, x, batch_size=2048, device=None, return_cpu=True):
        """
        Reconstruct input using posterior mean (deterministic)

        Args:
            x: Tensor or array of shape [n_samples, n_features] or [n_features]
            batch_size: Process in batches of this size for memory efficiency
            device: Device to move tensors to (e.g., 'cpu', 'mps', 'cuda'). If None, uses model's current device
            return_cpu: Return output on CPU
        
        Returns:
            Reconstruction(s) of shape [n_samples, n_features] or [n_features]
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Handle single sample
        squeeze_output = False
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze_output = True
        
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        
        n_samples = x.shape[0]
        reconstructions = []
        
        with torch.no_grad():
            # Process in batches for memory efficiency
            for i in range(0, n_samples, batch_size):
                batch = x[i:i+batch_size].to(device)
                mu, _ = self.encode(batch)
                logits = self.decode(mu)
                x_recon = torch.sigmoid(logits)
                reconstructions.append(x_recon.cpu() if return_cpu else x_recon)
        
        result = torch.cat(reconstructions, dim=0)
        
        # Remove batch dimension if input was single sample
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # Get data
        x = batch[0]

        # Forward pass
        p_x, q_z, z = self.forward(x)

        # Compute loss
        losses = beta_tcvae_loss(
            x=x,
            p_x=p_x,
            q_z=q_z,
            z=z,
            beta=self.beta,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        # Compute reconstruction metrics
        x_recon = torch.sigmoid(p_x.logits)
        confusion_matrix = binary_confusion_matrix(x_recon, x, threshold=0.5)

        # Log loss
        self.log("train_loss", losses["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", losses["recon_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("mi_loss", losses["mi_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tc_loss", losses["tc_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("dw_kl_loss", losses["dw_kl_loss"], on_step=False, on_epoch=True, prog_bar=True)

        # Log reconstruction metrics
        self.log('train_precision', confusion_matrix['precision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', confusion_matrix['recall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', confusion_matrix['f1'], on_step=False, on_epoch=True)

        # Return total loss for backpropagation
        return losses["total_loss"]
    
    def validation_step(self, batch, batch_idx):
        # Get data
        x = batch[0]

        # Forward pass
        p_x, q_z, z = self.forward(x)

        # Compute loss
        losses = beta_tcvae_loss(
            x=x,
            p_x=p_x,
            q_z=q_z,
            z=z,
            beta=self.beta,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        # Compute reconstruction metrics
        x_recon = torch.sigmoid(p_x.logits)
        confusion_matrix = binary_confusion_matrix(x_recon, x, threshold=0.5)

        # Log metrics
        self.log("val_loss", losses["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", losses["recon_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("mi_loss", losses["mi_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tc_loss", losses["tc_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("dw_kl_loss", losses["dw_kl_loss"], on_step=False, on_epoch=True, prog_bar=True)

        # Log reconstruction metrics
        self.log('val_precision', confusion_matrix['precision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', confusion_matrix['recall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', confusion_matrix['f1'], on_step=False, on_epoch=True)
        
        # Return total loss for backpropagation
        return losses["total_loss"]

