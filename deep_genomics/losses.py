import torch
from torch.distributions import (
    Normal, 
    kl_divergence,
)

def binary_vae_loss(x, p_x, q_z, beta=1.0):
    """
    Compute the VAE loss for binary data (ELBO with Bernoulli likelihood).
    
    Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        x: Input data, shape [batch_size, input_dim]
        p_x: Bernoulli distribution over reconstructions, p(x|z)
        q_z: Posterior distribution over latents, q(z|x)
        beta: Weight for KL divergence term (default=1.0). 
              Use beta < 1.0 to prevent posterior collapse.
    
    Returns:
        tuple: (total_loss, recon_loss, kl_loss)
            - total_loss: Combined loss for backpropagation
            - recon_loss: Negative log-likelihood (reconstruction error)
            - kl_loss: KL divergence between posterior and prior
    """
    # Reconstruction loss: -log p(x|z)
    # Negative log-likelihood of data under the model
    recon_loss = -p_x.log_prob(x).sum(dim=1).mean()

    # KL divergence: KL(q(z|x) || p(z))
    # Regularization term - keeps posterior close to standard normal prior
    p_z = Normal(torch.zeros_like(q_z.mean), torch.ones_like(q_z.stddev))
    kl_loss = kl_divergence(q_z, p_z).sum(dim=1).mean()

    # Total loss (negative ELBO)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss