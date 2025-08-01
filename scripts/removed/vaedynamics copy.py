import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from torchdiffeq import odeint

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=256, image_channels=3):
        super().__init__()
        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute feature size dynamically
        with torch.no_grad():
            test_input = torch.zeros(1, image_channels, 64, 64)
            feature_size = self.feature_extractor(test_input).shape[1]
        
        # Layers to map features to latent space
        self.fc_mu = nn.Linear(feature_size, latent_dim)
        self.fc_logvar = nn.Linear(feature_size, latent_dim)
    
    def forward(self, x, prev_latent=None, prev_action=None):
        """
        Encode image to latent representation
        
        Args:
            x (torch.Tensor): Input image
            prev_latent (torch.Tensor, optional): Previous latent state
            prev_action (torch.Tensor, optional): Previous action
        
        Returns:
            torch.Tensor: Latent representation
        """
        features = self.feature_extractor(x)
        
        # Compute mean and log variance
        mu = self.fc_mu(features)
        log_var = self.fc_logvar(features)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, log_var

class NeuralODE(nn.Module):
    def __init__(self, latent_dim, control_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        # Dynamics network for ODE integration
        layers = []
        input_dim = latent_dim + control_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.ReLU()
            ])
            input_dim = h_dim
        
        layers.append(nn.Linear(input_dim, latent_dim))
        
        self.dynamics_net = nn.Sequential(*layers)
    
    def forward(self, t, x, u):
        """
        Compute latent state dynamics
        
        Args:
            t (torch.Tensor): Time (not used, but required by odeint)
            x (torch.Tensor): Current latent state
            u (torch.Tensor): Control input
        
        Returns:
            torch.Tensor: Latent state derivative
        """
        xu = torch.cat([x, u], dim=-1)
        dx_dt = self.dynamics_net(xu)
        return dx_dt

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=256, image_channels=3):
        super().__init__()
        
        # Initial dense layer
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # Transposed convolutional layers to reconstruct image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensure pixel values between 0 and 1
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        reconstructed_img = self.decoder(x)
        return reconstructed_img

class DynamicsVAE(nn.Module):
    def __init__(
        self, 
        latent_dim=256, 
        control_dim=2, 
        image_channels=3, 
        dt=0.1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.dt = dt
        
        # Encoder, Neural ODE, Decoder
        self.encoder = ImageEncoder(latent_dim, image_channels)
        self.neural_ode = NeuralODE(latent_dim, control_dim)
        self.decoder = ImageDecoder(latent_dim, image_channels)
    
    def simulate(self, images, actions, dt=0.1):
        """
        Simulate latent dynamics through a sequence of images and actions
        
        Args:
            images (torch.Tensor): Sequence of images 
            actions (torch.Tensor): Sequence of actions
        
        Returns:
            tuple: Simulated latent states, reconstructed images
        """
        batch_size, seq_len = images.shape[:2]
        
        # Initial latent state
        z_init, _, _ = self.encoder(images[:, 0])
        
        # Simulate sequence
        zs = [z_init]
        reconstructed_images = [self.decoder(z_init)]
        
        for t in range(1, seq_len):
            # Integrate ODE for next latent state
            t_span = torch.tensor([0, dt], device=images.device, dtype=torch.float32)
            z_next = odeint(
                self.neural_ode, 
                zs[-1], 
                t_span, 
                args=(actions[:, t-1],)
            )[-1]  # Take last time step
            
            # Encode next image
            next_z, _, _ = self.encoder(images[:, t], prev_latent=z_next)
            
            # Reconstruct image
            reconstructed_img = self.decoder(next_z)
            
            zs.append(next_z)
            reconstructed_images.append(reconstructed_img)
        
        return torch.stack(zs, dim=1), torch.stack(reconstructed_images, dim=1)
    
    def loss_function(self, images, actions, reconstructed_images, zs, original_zs, original_mu, original_log_var):
        """
        Compute VAE and dynamics losses
        
        Args:
            images (torch.Tensor): Original images
            actions (torch.Tensor): Control actions
            reconstructed_images (torch.Tensor): Reconstructed images
            zs (torch.Tensor): Simulated latent states
            original_zs (torch.Tensor): Original encoded latent states
            original_mu (torch.Tensor): Original latent mean
            original_log_var (torch.Tensor): Original latent log variance
        
        Returns:
            dict: Computed losses
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_images, images)
        
        # Dynamics loss (latent state matching)
        dynamics_loss = F.mse_loss(zs, original_zs)
        
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + original_log_var 
            - original_mu**2 
            - torch.exp(original_log_var)
        )
        
        return {
            'recon_loss': recon_loss,
            'dynamics_loss': dynamics_loss,
            'kl_loss': kl_loss
        }

class DynamicsVAETrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        device='cpu', 
        lr=1e-4,
        kl_weight=0.001,
        recon_weight=1.0,
        dynamics_weight=1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss weights
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.dynamics_weight = dynamics_weight
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, actions) in enumerate(self.train_loader):
            images = images.to(self.device)
            actions = actions.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Encode images
            original_zs, original_mu, original_log_var = self.model.encoder(images)
            
            # Simulate dynamics
            zs, reconstructed_images = self.model.simulate(images, actions)
            
            # Compute losses
            losses = self.model.loss_function(
                images, actions, 
                reconstructed_images, zs, 
                original_zs, original_mu, original_log_var
            )
            
            # Weighted total loss
            total_batch_loss = (
                losses['recon_loss'] * self.recon_weight + 
                losses['dynamics_loss'] * self.dynamics_weight + 
                losses['kl_loss'] * self.kl_weight
            )
            
            # Backpropagate
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # Log metrics
            wandb.log({
                'train_batch_loss': total_batch_loss.item(),
                **{f'train_{k}': v.item() for k, v in losses.items()}
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, actions in self.val_loader:
                images = images.to(self.device)
                actions = actions.to(self.device)
                
                # Encode images
                original_zs, original_mu, original_log_var = self.model.encoder(images)
                
                # Simulate dynamics
                zs, reconstructed_images = self.model.simulate(images, actions)
                
                # Compute losses
                losses = self.model.loss_function(
                    images, actions, 
                    reconstructed_images, zs, 
                    original_zs, original_mu, original_log_var
                )
                
                # Weighted total loss
                total_batch_loss = (
                    losses['recon_loss'] * self.recon_weight + 
                    losses['dynamics_loss'] * self.dynamics_weight + 
                    losses['kl_loss'] * self.kl_weight
                )
                
                total_loss += total_batch_loss.item()
                
                # Log metrics
                wandb.log({
                    'val_batch_loss': total_batch_loss.item(),
                    **{f'val_{k}': v.item() for k, v in losses.items()}
                })
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        wandb.init(project="dynamics-vae")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        wandb.finish()

# Utility function for data loading
def create_data_loaders(data, train_split=0.8, batch_size=32):
    """
    Create train and validation data loaders from numpy arrays
    """
    # Convert to PyTorch tensors
    images = torch.FloatTensor(data['states_rgb']) / 255.0
    actions = torch.FloatTensor(data['actions'])
    
    # Determine split index
    split_idx = int(len(images) * train_split)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        images[:split_idx], 
        actions[:split_idx]
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        images[split_idx:], 
        actions[split_idx:]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader
def main():
    # Load your data
    data = np.load('safe_rl_dataset_images.npz')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(data)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DynamicsVAE(
        state_dim=4,  # x_car, y_car, x_obs, y_obs
        action_dim=2, 
        latent_dim=4, 
        image_channels=3,
        hidden_dim=64,
        dt=0.1
    ).to(device)
    
    # Initialize trainer
    trainer = DynamicsVAETrainer(
        model, 
        train_loader, 
        val_loader, 
        device=device,
        lr=1e-4,
        wrec1=1.0,  # Image reconstruction loss weight
        wrec2=1.0,  # Encoded-decoded image reconstruction loss weight
        wstate=1.0,  # State prediction loss weight
        kl_weight=0.001  # Weight for KL divergence loss
    )
    
    # Train the model
    trainer.train(num_epochs=100)

if __name__ == "__main__":
    main()
