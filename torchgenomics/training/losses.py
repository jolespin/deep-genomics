import torch
from torch.distributions import (
    Normal, 
    kl_divergence,
)

def beta_vae_loss(x, p_x, q_z, beta=1.0):
    """
    Compute the β-VAE loss (ELBO with KL annealing).
    
    Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        x: Input data, shape [batch_size, input_dim]
        p_x: Reconstruction distribution p(x|z), e.g., Bernoulli for binary data,
             Normal for continuous data, or any PyTorch distribution with log_prob method
        q_z: Posterior distribution over latents, q(z|x)
        beta: Weight for KL divergence term (default=1.0). 
              Use beta < 1.0 to prevent posterior collapse, beta > 1.0 for stronger disentanglement.
    
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

    return {
        "total_loss":total_loss, 
        "recon_loss":recon_loss, 
        "kl_loss":kl_loss,
        }

def beta_tcvae_loss(x, p_x, q_z, z, beta=1.0, alpha=1.0, gamma=1.0):
    """
    Compute the β-TCVAE loss.
    
    Loss = Reconstruction + α*Index-Code MI + β*Total Correlation + γ*Dimension-wise KL
    
    Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
    (Chen et al., 2018): https://arxiv.org/abs/1802.04942
    
    Args:
        x: Input data, shape [batch_size, input_dim]
        p_x: Reconstruction distribution p(x|z), e.g., Bernoulli for binary data,
             Normal for continuous data, or any PyTorch distribution with log_prob method
        q_z: Posterior distribution over latents, q(z|x)
        z: Sampled latent codes, shape [batch_size, latent_dim]
        beta: Weight for total correlation term (default: 1.0)
        alpha: Weight for index-code MI term (default: 1.0)
        gamma: Weight for dimension-wise KL term (default: 1.0)
    
    Returns:
        tuple: (total_loss, recon_loss, mi_loss, tc_loss, dw_kl_loss)
    """
    batch_size = x.shape[0]
    latent_dim = z.shape[1]
    device = z.device
    
    # Minibatch weight for q(z) estimation
    log_batch_size = torch.log(torch.tensor(batch_size, dtype=torch.float32, device=device))
    
    # Reconstruction loss: -log p(x|z)
    recon_loss = -p_x.log_prob(x).sum(dim=1).mean()
    
    # Compute log q(z|x) for the batch
    log_qz_given_x = q_z.log_prob(z).sum(dim=1)  # [batch_size]
    
    # Expand for broadcasting to compute all pairs
    z_expand = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
    mu_expand = q_z.mean.unsqueeze(0)  # [1, batch_size, latent_dim]
    std_expand = q_z.stddev.unsqueeze(0)  # [1, batch_size, latent_dim]
    
    # Compute log q(z_i[d] | x_j) for all pairs (i,j) and dimensions d
    # Shape: [batch_size_i, batch_size_j, latent_dim]
    # Entry [i,j,d] represents log q(z_i[d] | x_j), keeping dimensions separate
    all_log_qz_given_x_per_dim = Normal(mu_expand, std_expand).log_prob(z_expand)
    
    # Minibatch weighted log q(z): estimate using all pairs, then average over j
    log_qz = torch.logsumexp(all_log_qz_given_x_per_dim.sum(dim=2), dim=1) - log_batch_size  # [batch_size]
    
    # Compute log q(z) product of marginals: log prod_d q(z_d)
    # For each dimension d, estimate log q(z_d) independently, then sum over d
    log_qz_product = (
        torch.logsumexp(all_log_qz_given_x_per_dim, dim=1) - log_batch_size
    ).sum(dim=1)  # [batch_size]
    
    # Compute log p(z) = log N(0, I)
    log_pz = Normal(
        torch.zeros(latent_dim, device=device), 
        torch.ones(latent_dim, device=device)
    ).log_prob(z).sum(dim=1)  # [batch_size]
    
    # Decompose KL into three terms:
    # Index-code mutual information: I(z;x) = E[log q(z|x) - log q(z)]
    mi_loss = (log_qz_given_x - log_qz).mean()
    
    # Total correlation: TC(z) = E[log q(z) - log prod_d q(z_d)]
    tc_loss = (log_qz - log_qz_product).mean()
    
    # Dimension-wise KL: sum_d KL(q(z_d)||p(z_d)) = E[log prod_d q(z_d) - log p(z)]
    dw_kl_loss = (log_qz_product - log_pz).mean()
    
    # Total loss
    total_loss = recon_loss + alpha * mi_loss + beta * tc_loss + gamma * dw_kl_loss
    
    return {
        "total_loss":total_loss, 
        "recon_loss":recon_loss, 
        "mi_loss":mi_loss, 
        "tc_loss":tc_loss, 
        "dw_kl_loss":dw_kl_loss,
    }