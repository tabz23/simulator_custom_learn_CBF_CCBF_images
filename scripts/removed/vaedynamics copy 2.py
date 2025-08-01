import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

def mlp(layer_sizes, activation=nn.ReLU(), output_activation=None):
    """
    Create a Multi-Layer Perceptron
    
    Args:
        layer_sizes (list): List of layer sizes including input and output
        activation (nn.Module): Activation function for hidden layers
        output_activation (nn.Module): Activation function for output layer
    
    Returns:
        nn.Sequential: Multi-layer perceptron
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
        layers.append(layer)
        
        # Add activation function for hidden layers
        if i < len(layer_sizes) - 2:
            layers.append(activation)
    
    # Add output activation if specified
    if output_activation is not None:
        layers.append(output_activation)
    
    return nn.Sequential(*layers)

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=32):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute the size after convolutions
        with torch.no_grad():
            sample_input = torch.zeros(1, image_channels, 64, 64)
            feature_size = self.encoder(sample_input).shape[1]
        
        # Latent space layers
        self.fc_mu = nn.Linear(feature_size, latent_dim)
        self.fc_var = nn.Linear(feature_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, feature_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_size // (16*16), 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """
        Encode input image to latent space
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from latent space
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
        
        Returns:
            z (torch.Tensor): Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to reconstructed image
        
        Args:
            z (torch.Tensor): Latent vector
        
        Returns:
            reconstructed_image (torch.Tensor): Reconstructed image
        """
        h = self.decoder_input(z)
        h = h.view(h.size(0), h.size(1) // (16*16), 16, 16)
        return self.decoder(h)
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Args:
            x (torch.Tensor): Input image
        
        Returns:
            reconstruction (torch.Tensor): Reconstructed image
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class AffineDynamics(nn.Module):
    def __init__(self, state_dim=32, action_dim=2, num_action=1, 
                 hidden_dim=64, num_layers=3, dt=0.1):
        """
        Initialize Affine Dynamics Model
        
        Args:
            state_dim (int): Dimensionality of latent state
            action_dim (int): Dimensionality of action
            num_action (int): Number of action components
            hidden_dim (int): Dimensionality of hidden layers
            num_layers (int): Number of hidden layers
            dt (float): Time step for dynamics
        """
        super().__init__()
        
        self.num_action = num_action
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dt = dt
        
        # Drift term (f) network
        self.f = mlp(
            [self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim], 
            activation=nn.ReLU()
        )
        
        # Diffusion term (g) network
        self.g = mlp(
            [self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim*self.num_action], 
            activation=nn.ReLU()
        )
    
    def forward(self, state):
        """
        Forward pass to compute drift and diffusion terms
        
        Args:
            state (torch.Tensor): Current state
        
        Returns:
            f (torch.Tensor): Drift term
            g (torch.Tensor): Diffusion term
        """
        return self.f(state), self.g(state)
    
    def forward_x_dot(self, state, action):
        """
        Compute state derivative
        
        Args:
            state (torch.Tensor): Current state
            action (torch.Tensor): Current action
        
        Returns:
            x_dot (torch.Tensor): State derivative
        """
        f, g = self.forward(state)
        
        # Compute control input contribution
        gu = torch.einsum(
            'bsa,ba->bs', 
            g.view(g.shape[0], self.state_dim, self.num_action), 
            action
        )
        
        # Compute state derivative
        x_dot = f + gu
        return x_dot
    
    def forward_next_state(self, state, action):
        """
        Compute next state using Euler integration
        
        Args:
            state (torch.Tensor): Current state
            action (torch.Tensor): Current action
        
        Returns:
            next_state (torch.Tensor): Next state
        """
        return self.forward_x_dot(state, action) * self.dt + state

class DynamicsDataset(Dataset):
    def __init__(self, data_path):
        """
        Initialize the dataset from a numpy .npz file path
        
        Args:
            data_path (str): Path to the .npz file
        """
        try:
            # Load the data
            data = np.load(data_path)
            
            # Print available keys for debugging
            print("Available keys in the dataset:", list(data.keys()))
            
            # Check for required keys and load data
            if 'image_sequence' in data:
                self.images = data['image_sequence']
            elif 'images' in data:
                self.images = data['images']
            else:
                raise KeyError("No image data found. Expected 'image_sequence' or 'images' key.")
            
            if 'action_sequence' in data:
                self.actions = data['action_sequence']
            elif 'actions' in data:
                self.actions = data['actions']
            else:
                raise KeyError("No action data found. Expected 'action_sequence' or 'actions' key.")
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        # Assertions to ensure correct data shape
        assert self.images.shape[0] == self.actions.shape[0], "Images and actions must have same batch dimension"
        assert self.images.shape[1] == self.actions.shape[1], "Images and actions must have same time dimension"
        
        # Convert to torch tensors
        self.images = torch.from_numpy(self.images).float() / 255.0  # Normalize images
        self.actions = torch.from_numpy(self.actions).float()
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        """
        Returns a sequence of images and corresponding actions for a given index
        
        Returns:
            images (torch.Tensor): Sequence of images, shape (T, 64, 64, 3)
            actions (torch.Tensor): Sequence of actions, shape (T, action_dim)
        """
        return self.images[idx], self.actions[idx]

class Trainer:
    def __init__(self, vae, dynamics, dataset, batch_size=128, lr=1e-4, device="mps"):
        """
        Initialize trainer for VAE and Dynamics model
        
        Args:
            vae (VAE): Variational Autoencoder model
            dynamics (AffineDynamics): Dynamics model
            dataset (DynamicsDataset): Training dataset
            batch_size (int): Batch size for training
            lr (float): Learning rate
            device (str): Training device
        """
        self.device = torch.device(device)
        
        # Models
        self.vae = vae.to(self.device)
        self.dynamics = dynamics.to(self.device)
        
        # Dataset and DataLoader
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Optimizers
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        self.dynamics_optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.kl_loss = self._kl_divergence
    
    def _kl_divergence(self, mu, logvar):
        """
        Compute KL Divergence loss
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
        
        Returns:
            kl_loss (torch.Tensor): KL Divergence loss
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def train_step(self):
        """
        Perform one training step
        
        Returns:
            vae_loss (float): VAE loss for the step
            dynamics_loss (float): Dynamics model loss for the step
        """
        # Set models to training mode
        self.vae.train()
        self.dynamics.train()
        
        total_vae_loss = 0
        total_dynamics_loss = 0
        
        for images, actions in self.dataloader:
            images = images.to(self.device)
            actions = actions.to(self.device)
            
            # Reset gradients
            self.vae_optimizer.zero_grad()
            self.dynamics_optimizer.zero_grad()
            
            # VAE Forward Pass
            batch_size, seq_len, C, H, W = images.shape
            images_flat = images.view(-1, C, H, W)
            
            reconstructed_images, mu, logvar = self.vae(images_flat)
            
            # VAE Losses
            recon_loss = self.reconstruction_loss(reconstructed_images, images_flat)
            kl_loss = self.kl_loss(mu, logvar)
            vae_loss = recon_loss + 0.001 * kl_loss  # KL divergence weight
            
            # Encode sequence to latent space
            with torch.no_grad():
                latent_states = []
                for t in range(seq_len):
                    lat_mu, _ = self.vae.encode(images[:, t])
                    latent_states.append(lat_mu)
                latent_states = torch.stack(latent_states, dim=1)
            
            # Dynamics Model Loss
            dynamics_pred_loss = 0
            for t in range(1, seq_len):
                # Predict next latent state
                pred_state = self.dynamics.forward_next_state(
                    latent_states[:, t-1], 
                    actions[:, t-1]
                )
                
                # Prediction loss
                dynamics_pred_loss += F.mse_loss(pred_state, latent_states[:, t])
            
            # Total loss
            total_loss = vae_loss + dynamics_pred_loss
            total_loss.backward()
            
            # Update parameters
            self.vae_optimizer.step()
            self.dynamics_optimizer.step()
            
            # Log metrics
            total_vae_loss += vae_loss.item()
            total_dynamics_loss += dynamics_pred_loss.item()
        
        # Average losses
        avg_vae_loss = total_vae_loss / len(self.dataloader)
        avg_dynamics_loss = total_dynamics_loss / len(self.dataloader)
        
        # Log to wandb
        wandb.log({
            'vae_loss': avg_vae_loss, 
            'dynamics_loss': avg_dynamics_loss
        })
        
        return avg_vae_loss, avg_dynamics_loss

def main():
    # Initialize wandb
    wandb.init(project="vae-dynamics-car-navigation", name="training_run")
    print("initialized wandb\n\n")
    
    # Path to your dataset
    dataset_path = "safe_rl_dataset_images.npz"
    
    # Load dataset
    try:
        dataset = DynamicsDataset(dataset_path)
        print("Dataset loaded successfully\n\n")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please check the dataset path and content.")
        return
    
    # Initialize models
    vae = VAE(image_channels=3, latent_dim=32)
    dynamics = AffineDynamics(state_dim=32, action_dim=2)
    
    # Setup trainer
    trainer = Trainer(vae, dynamics, dataset, batch_size=128, lr=1e-4, device="mps")
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        try:
            vae_loss, dynamics_loss = trainer.train_step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, VAE Loss: {vae_loss:.4f}, Dynamics Loss: {dynamics_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'vae_state_dict': trainer.vae.state_dict(),
                    'dynamics_state_dict': trainer.dynamics.state_dict(),
                    'vae_optimizer_state_dict': trainer.vae_optimizer.state_dict(),
                    'dynamics_optimizer_state_dict': trainer.dynamics_optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        except Exception as e:
            print(f"Error during training in epoch {epoch+1}: {e}")
            break
    
    # Finish wandb run
    wandb.finish()
    print("Training complete")

if __name__ == "__main__":
    main()