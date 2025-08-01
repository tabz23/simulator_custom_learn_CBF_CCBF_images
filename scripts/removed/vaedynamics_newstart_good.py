import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import trange
# Network Utility Functions
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Create a multi-layer perceptron with given sizes and activations."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class DynamicsDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        # Load data
        self.actions = torch.FloatTensor(data['actions'])
        self.states_rgb = torch.FloatTensor(data['states_rgb']).permute(0, 3, 1, 2) / 255.0
        self.dones = torch.FloatTensor(data['dones'])
        
        # Prepare sequences
        self.sequences = self._create_sequences(sequence_length)

    def _create_sequences(self, sequence_length):
        """
        Create sequences respecting trajectory boundaries
        
        Args:
            sequence_length (int): Length of sequences to extract
        
        Returns:
            List of dictionaries, each containing a sequence
        """
        sequences = []
        
        # Iterate through the entire dataset
        i = 0
        while i < len(self.dones) - sequence_length:
            # Check if we can create a full sequence without crossing trajectory boundary
            sequence_actions = self.actions[i:i+sequence_length]
            sequence_states = self.states_rgb[i:i+sequence_length]
            sequence_dones = self.dones[i:i+sequence_length]
            
            # If any point in the sequence is a done, skip this sequence
            if torch.any(sequence_dones == 1):
                # Find the index of the first done
                done_idx = torch.where(sequence_dones == 1)[0]
                
                # Move to the next trajectory start
                i += done_idx[0] + 1
                continue
            
            # Create sequence dictionary
            sequence = {
                'action': sequence_actions,
                'state_rgb': sequence_states,
                'done': sequence_dones
            }
            
            sequences.append(sequence)
            
            # Move to next possible sequence
            i += 1
        
        return sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

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
    # print(states.shape)torch.Size([32, 10, 3, 64, 64])
    batch_size, seq_length, channels, height, width = states.shape
    
    # Initialize variables
    loss_dyn = torch.zeros(1, device=device)
    loss_vae = torch.zeros(1, device=device)
    
    # Initial hidden state
    z_prev = torch.zeros(batch_size, model.dynamics.state_dim, device=device)
    
    # Track number of valid steps
    valid_steps = 0
    
    for t in range(1, seq_length):
        # Check if this is a valid step (not after a done)
        valid_mask = (dones[:, t-1] == 0).float()
        
        if valid_mask.sum() == 0:
            continue
        
        # valid_indices = (valid_mask == 1).nonzero(as_tuple=True)
        
        # Current states and actions for valid steps
        x_curr = states[:, t][valid_mask.bool()]
        action_curr = actions[:, t-1][valid_mask.bool()]
        z_prev_valid = z_prev[valid_mask.bool()]
        
        # Forward pass
        z, mu, logvar, z_pred, x_recon = model(x_curr, z_prev_valid, action_curr)
        
        # Dynamics loss
        loss_dyn += F.mse_loss(z_pred, z)
        
        # VAE loss components
        recon_loss = F.mse_loss(x_recon, x_curr)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss_vae += recon_loss + beta * kl_loss
        
        # Update previous hidden state
        z_prev[valid_mask == 1] = z
        
        valid_steps += 1
    
    # Avoid division by zero
    if valid_steps > 0:
        loss_dyn /= valid_steps
        loss_vae /= valid_steps
    
    total_loss = loss_dyn + loss_vae
    wandb.log({
                'total_loss': total_loss.item(),
                'dynamics_loss': loss_dyn.item(),
                'vae_loss': loss_vae.item()
            })
    
    return total_loss, loss_dyn, loss_vae

def train(model, dataloader, optimizer, device, epochs=100):
    model.to(device)
    
    for epoch in trange(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            loss, loss_dyn, loss_vae = compute_loss(model, batch, device)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log metrics
            wandb.log({
                'total_loss': loss.item(),
                'dynamics_loss': loss_dyn.item(),
                'vae_loss': loss_vae.item()
            })
        
        print(f"Epoch {epoch+1}, Total Loss: {total_loss/len(dataloader)}")
    
    return model

def main():
    # Initialize wandb
    wandb.init(project="vae-dynamics-car-navigation", name="training_run")
    print("Initialized wandb\n")
    
    # Load dataset
    data = np.load("safe_rl_dataset_images.npz")
    print("Initialized data variable\n")
    
    # Device configuration
    device = "mps"
    
    # Hyperparameters
    input_channels = 3
    latent_dim = 32
    action_dim = 2
    hidden_dim = 64
    batch_size = 32
    learning_rate = 1e-4
    epochs = 100
    
    # Create dataset and dataloader
    dataset = DynamicsDataset(data, sequence_length=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("loaded data\n")
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