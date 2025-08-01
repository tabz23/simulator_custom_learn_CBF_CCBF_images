import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

# Network Utility Functions
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Create a multi-layer perceptron with given sizes and activations."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class DynamicsDataset(Dataset):
    """Dataset for dynamics learning with sequential data."""
    def __init__(self, num_trajectories=100, trajectory_length=10, 
                 image_size=64, num_channels=3):
        # Generate synthetic data for demonstration
        self.actions = np.random.uniform(-1, 1, (num_trajectories, trajectory_length, 2))
        self.states_rgb = np.random.uniform(0, 1, (num_trajectories, trajectory_length, image_size, image_size, num_channels))
        self.dones = np.zeros((num_trajectories, trajectory_length))
        self.dones[:, -1] = 1  # Mark last timestep of each trajectory
        
        # Convert to tensors
        self.actions = torch.FloatTensor(self.actions)
        self.states_rgb = torch.FloatTensor(self.states_rgb)
        self.states_rgb = self.states_rgb.permute(0, 1, 4, 2, 3)  # Change to (N, T, C, H, W)
        self.dones = torch.FloatTensor(self.dones)
        
        # Make sure the dataset size is large enough for the sequences
        self.num_samples = len(self.states_rgb)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'action': self.actions[idx],
            'state_rgb': self.states_rgb[idx],
            'done': self.dones[idx]
        }

class AffineDynamics(nn.Module):
    """Affine dynamics model for learning state transitions."""
    def __init__(
        self,
        num_action,
        state_dim,
        hidden_dim=64,
        num_layers=3,
        dt=0.1):
        super().__init__()
        
        self.num_action = num_action
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dt = dt
        
        self.f = mlp([self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim], activation=nn.ReLU)
        self.g = mlp([self.state_dim] + num_layers*[self.hidden_dim] + [self.state_dim*self.num_action], activation=nn.ReLU)
        
    def forward(self, state):
        return self.f(state), self.g(state)
    
    def forward_x_dot(self, state, action):
        f, g = self.forward(state)
        gu = torch.einsum('bsa,ba->bs', g.view(g.shape[0], self.state_dim, self.num_action), action)
        x_dot = f + gu
        return x_dot
    
    def forward_next_state(self, state, action):
        return self.forward_x_dot(state, action) * self.dt + state

class VAEDynamicsModel(nn.Module):
    """Variational Autoencoder with Dynamics Model"""
    def __init__(self, input_channels, latent_dim, action_dim, 
                 hidden_dim=64, num_layers=3):
        super().__init__()
        
        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Dynamics model
        self.dynamics = AffineDynamics(
            num_action=action_dim, 
            state_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers
        )
        
        # Decoder Network 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # To constrain output to [0, 1]
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, previous_z, action):
        # Encode current observation
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Predict next latent state using dynamics model
        next_z_pred = self.dynamics.forward_next_state(previous_z, action)
        
        # Reconstruct observation
        x_reconstructed = self.decode(z)
        
        return z, mu, logvar, next_z_pred, x_reconstructed

def compute_loss(model, batch, device, beta=1.0):
    states = batch['state_rgb'].to(device)
    actions = batch['action'].to(device)
    dones = batch['done'].to(device)
    
    batch_size, seq_length, channels, height, width = states.shape
    
    # Initialize variables
    loss_dyn = torch.zeros(1, device=device)
    loss_vae = torch.zeros(1, device=device)
    
    # Initial hidden state
    z_prev = torch.zeros(batch_size, model.dynamics.state_dim, device=device)
    
    for t in range(1, seq_length):
        # Current states and actions
        x_curr = states[:, t]
        action_curr = actions[:, t-1]
        
        # Forward pass
        z, mu, logvar, z_pred, x_recon = model(x_curr, z_prev, action_curr)
        
        # Dynamics loss
        loss_dyn += F.mse_loss(z_pred, z)
        
        # VAE loss components
        recon_loss = F.mse_loss(x_recon, x_curr)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss_vae += recon_loss + beta * kl_loss
        
        # Update previous hidden state
        z_prev = z
    
    # Average losses
    loss_dyn /= (seq_length - 1)
    loss_vae /= (seq_length - 1)
    
    total_loss = loss_dyn + loss_vae
    wandb.log({
                'total_loss': total_loss.item(),
                'dynamics_loss': loss_dyn.item(),
                'vae_loss': loss_vae.item()
            })
    
    return total_loss, loss_dyn, loss_vae

def train(model, dataloader, optimizer, device, epochs=100):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            loss, loss_dyn, loss_vae = compute_loss(model, batch, device)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
      
        
        print(f"Epoch {epoch+1}, Total Loss: {total_loss/len(dataloader)}")
    
    return model

def main():
    # Device configuration
    device = "mps"
    
    # Initialize wandb
    wandb.init(project="vae-dynamics-learning", name="synthetic_data_run")
    
    # Hyperparameters
    input_channels = 3
    latent_dim = 32
    action_dim = 2
    hidden_dim = 64
    seq_length = 10
    batch_size = 32
    learning_rate = 1e-4
    epochs = 50
    
    # Create synthetic dataset
    dataset = DynamicsDataset(
        num_trajectories=1000, 
        trajectory_length=seq_length, 
        image_size=64, 
        num_channels=input_channels
    )
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = VAEDynamicsModel(
        input_channels=input_channels, 
        latent_dim=latent_dim, 
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    trained_model = train(model, dataloader, optimizer, device, epochs)
    
    # Save the model
    torch.save(trained_model.state_dict(), 'vae_dynamics_model.pth')
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()