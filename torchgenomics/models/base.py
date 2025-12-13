from pathlib import Path
import json
from abc import (
    ABC,
    abstractmethod,
)
import numpy as np
import torch
import torch.nn as nn
import lightning as L

class BaseVAE(L.LightningModule, ABC):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            learning_rate:float,
            ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        # Ignore encoder/decoder save the rest
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
    # Default abstract methods
    @abstractmethod
    def encode(self, x: torch.Tensor):
        pass
    @abstractmethod
    def decode(self, z: torch.Tensor):
        pass
    @abstractmethod
    def forward(self, x):
        """Return (p_x, q_z, z) for loss computation"""
        pass
    @abstractmethod
    def sample(self, n_samples:int, device=None):
        pass
    @abstractmethod
    def reconstruct(self, x, batch_size=2048, device=None, return_cpu=True):
        pass
    def transform(self, x, batch_size=2048, device=None, return_cpu=True):
        """
        Transform input to latent representation (deterministic)
        
        Args:
            x: Tensor or array of shape [n_samples, n_features]
            batch_size: Process in batches of this size for memory efficiency
            device: Device to move tensors to (e.g., 'cpu', 'mps', 'cuda'). If None, uses model's current device
            return_cpu: Return output on CPU
        
        Returns:
            Latent representations of shape [n_samples, latent_dim]
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        
        n_samples = x.shape[0]
        latent_codes = []
        
        with torch.no_grad():
            # Process in batches for memory efficiency
            for i in range(0, n_samples, batch_size):
                batch = x[i:i+batch_size].to(device)
                mu, _ = self.encode(batch)
                latent_codes.append(mu.cpu() if return_cpu else mu)
        
        return torch.cat(latent_codes, dim=0)

    # Lightning Methods
    @abstractmethod
    def configure_optimizers(self):
        pass
    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass
    # @abstractmethod
    # # def test_step(self, batch, batch_idx):
    #     pass
    # @abstractmethod
    # # def predict_step(self, batch, batch_idx):
    #     pass

    # HuggingFace
    def save_pretrained(self, save_directory):
        """
        Save model weights and config in HuggingFace format.
        
        Args:
            save_directory: Path to directory where model will be saved
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model weights (PyTorch state_dict)
        weights_path = save_directory / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)
        
        # Save hyperparameters as config
        config_path = save_directory / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.hparams, f, indent=2, default=str)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, map_location=None):
        """
        Load model from HuggingFace format or local directory.
        
        Args:
            pretrained_model_path: Local path or HuggingFace Hub model ID
            map_location: Device to load model weights (e.g., 'cpu', 'cuda', 'mps')
        
        Returns:
            Loaded model instance
        """
        pretrained_model_path = Path(pretrained_model_path)
        
        # Load config
        config_path = pretrained_model_path / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance (subclass must handle this)
        # This is why it's a classmethod - subclass implements construction logic
        model = cls(**config)
        
        # Load weights
        weights_path = pretrained_model_path / 'pytorch_model.bin'
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {pretrained_model_path}")
        return model
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = False,
        token: str = None,
    ):
        """
        Upload model to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., "username/model-name")
            commit_message: Commit message for the upload
            private: Whether to make the repository private
            token: HuggingFace API token (or set HF_TOKEN environment variable)
        
        Example:
            model.push_to_hub("myusername/binary-vae-mnist")
        """
        from huggingface_hub import HfApi
        
        # Save to temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            
            # Upload to Hub
            api = HfApi()
            api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
                token=token,
            )
            
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
            )
        
        print(f"Model uploaded to https://huggingface.co/{repo_id}")