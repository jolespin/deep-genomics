from tqdm import tqdm
from loguru import logger
import torch

from .losses import binary_vae_loss

def train_binary_classifier(
    model,
    train_loader,
    num_epochs,
    learning_rate=1e-3,
    device='cpu',
    verbose=True,
    early_stopping_patience=None,
    early_stopping_min_delta=1e-4,
    wandb_run=None
    ):
    """
    Train a BinaryMLPClassifier.
    
    Args:
        model: BinaryMLPClassifier instance
        train_loader: DataLoader with training data (features, labels)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer (default: 1e-3)
        device: Device to train on ('cpu', 'cuda', 'mps')
        verbose: Log training progress (default: True)
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
                                 None to disable early stopping (default: None)
        early_stopping_min_delta: Minimum change in loss to qualify as improvement (default: 1e-4)
        wandb_run: Optional wandb run object for logging. If provided, training config will be logged to wandb.config
    
    Returns:
        dict: Training history with keys 'loss', 'accuracy'
              Each value is a list of per-epoch averages

    Usage with WanDB:
        import wandb

        # User initializes wandb with basic info
        with wandb.init(
            project="genomics-classifier",
            name="experiment-1",
            tags=["binary", "mlp"]
            ) as wandb_run:

            # Training parameters are automatically logged to wandb.config
            history = train_classifier(
                model,
                train_loader,
                num_epochs=20,
                learning_rate=1e-3,
                wandb_run=wandb_run
            )

    """
    
    # Log training config to wandb
    if wandb_run is not None:
        wandb_run.config.update({
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'device': device,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'batch_size': train_loader.batch_size,
            'dataset_size': len(train_loader.dataset),
            # Model architecture
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'dropout_rates': model.dropout_rates,
            'activation_fn': model.activation_fn.__name__
        })
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Track history
    history = {
        'loss': [],
        'accuracy': []
    }
    
    # Early stopping state
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        n_batches = len(train_loader)
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}") if verbose else train_loader
        
        for batch in iterator:
            x, y = batch[0].to(device), batch[1].to(device)
            
            # Ensure y is float and correct shape for BCEWithLogitsLoss
            if y.ndim == 1:
                y = y.unsqueeze(1).float()
            else:
                y = y.float()
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x)
            
            # Compute loss
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        # Calculate epoch averages
        avg_loss = total_loss / n_batches
        accuracy = correct / total
        
        # Store history
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy
            })
        
        # Log progress
        if verbose:
            logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Early stopping check
        if early_stopping_patience is not None:
            if avg_loss < best_loss - early_stopping_min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.warning(f"Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
                break
    
    return history

def train_binary_vae(
    model, 
    train_loader, 
    num_epochs, 
    learning_rate=1e-3, 
    beta=1.0, 
    device='cpu', 
    verbose=True,
    early_stopping_patience=None,
    early_stopping_min_delta=1e-4,
    wandb_run=None
):
    """
    Train a BinaryVariationalAutoencoder.
    
    Args:
        model: BinaryVariationalAutoencoder instance
        train_loader: DataLoader with training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer (default: 1e-3)
        beta: Weight for KL divergence term (default: 1.0)
        device: Device to train on ('cpu', 'cuda', 'mps')
        verbose: Log training progress (default: True)
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
                                 None to disable early stopping (default: None)
        early_stopping_min_delta: Minimum change in loss to qualify as improvement (default: 1e-4)
        wandb_run: Optional wandb run object for logging. If provided, training config will be logged to wandb.config
    
    Returns:
        dict: Training history with keys 'loss', 'recon_loss', 'kl_loss'
              Each value is a list of per-epoch averages


    Usage with WanDB:
    import wandb

    # User initializes wandb with basic info
    with wandb.init(
        project="genomics-vae",
        name="experiment-1",
        tags=["binary", "kegg"]
        ) as wandb_run:

        # Training parameters are automatically logged to wandb.config
        history = train_vae(
            model, 
            train_loader, 
            num_epochs=20,
            learning_rate=1e-3,
            beta=1.0,
            wandb_run=wandb_run
        )

    """
    
    # Log training config to wandb
    if wandb_run is not None:
        wandb_run.config.update({
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'beta': beta,
            'device': device,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'batch_size': train_loader.batch_size,
            'dataset_size': len(train_loader.dataset),
            # Model architecture
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'latent_dim': model.latent_dim,
            'activation_fn': model.activation_fn.__name__
        })

    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track history
    history = {
        'loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    # Early stopping state
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        n_batches = len(train_loader)
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        # Progress bar
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}") if verbose else train_loader
        
        for batch in iterator:
            x = batch[0].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            p_x, q_z, z = model(x)
            
            # Compute loss
            loss, recon, kl = binary_vae_loss(x, p_x, q_z, beta)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
        
        # Calculate epoch averages
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        
        # Store history
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch,
                'loss': avg_loss,
                'recon_loss': avg_recon,
                'kl_loss': avg_kl
            })
        
        # Log progress
        if verbose:
            logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
        
        # Early stopping check
        if early_stopping_patience is not None:
            if avg_loss < best_loss - early_stopping_min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.warning(f"Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
                break
    
    return history