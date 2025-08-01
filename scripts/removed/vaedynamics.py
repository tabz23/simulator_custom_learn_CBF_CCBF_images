import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
# Assuming you have these imports from your original file
from modules.dataset import *
from modules.network import *
from envs.car import *

# VAE Implementation
class VAE(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Control-Affine Dynamics Model
class AffineDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=3, dt=0.1):
        super(AffineDynamics, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt
        
        self.f = mlp([state_dim] + [hidden_dim]*num_layers + [state_dim], nn.ReLU)
        self.g = mlp([state_dim] + [hidden_dim]*num_layers + [state_dim*action_dim], nn.ReLU)
    
    def forward(self, state):
        return self.f(state), self.g(state).view(-1, self.state_dim, self.action_dim)
    
    def forward_next_state(self, state, action):
        f, g = self.forward(state)
        x_dot = f + torch.bmm(g, action.unsqueeze(2)).squeeze(2)
        return state + x_dot * self.dt

# Custom Dataset
class DynamicsDataset(Dataset):
    def __init__(self, data):
        # self.states = torch.FloatTensor(data['states'])
        self.actions = torch.FloatTensor(data['actions'])
        # self.next_states = torch.FloatTensor(data['next_states'])
        self.states_rgb = torch.FloatTensor(data['states_rgb']).permute(0, 3, 1, 2) / 255.0
        self.dones = torch.FloatTensor(data['dones'])
        # self.states_rgb_next = torch.FloatTensor(data['states_rgb_next']).permute(0, 3, 1, 2) / 255.0

    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'next_state': self.next_states[idx],
            'state_rgb': self.states_rgb[idx],
            'state_rgb_next': self.states_rgb_next[idx],
            'dones':self.dones[idx]
        }

# Trainer Class
class Trainer:
    def __init__(self, vae, dynamics, dataset, batch_size=128, lr=1e-4, device='cpu'):
        self.vae = vae.to(device)
        self.dynamics = dynamics.to(device)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.device = device
        
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        self.dynamics_optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)
    
    def vae_loss(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def dynamics_loss(self, pred_next_state, true_next_state):
        return nn.functional.mse_loss(pred_next_state, true_next_state)
    
    def train_step(self):
        self.vae.train()
        self.dynamics.train()
        
        total_vae_loss = 0
        total_dynamics_loss = 0
        
        for batch in self.dataloader:
            state = batch['state'].to(self.device)
            action = batch['action'].to(self.device)
            next_state = batch['next_state'].to(self.device)
            state_rgb = batch['state_rgb'].to(self.device)
            state_rgb_next = batch['state_rgb_next'].to(self.device)
            
            # VAE forward pass
            recon_state, mu, logvar = self.vae(state_rgb)
            vae_loss = self.vae_loss(recon_state, state_rgb, mu, logvar)
            
            # Dynamics forward pass
            pred_next_state = self.dynamics.forward_next_state(state, action)
            dynamics_loss = self.dynamics_loss(pred_next_state, next_state)
            
            # Backpropagation
            self.vae_optimizer.zero_grad()
            self.dynamics_optimizer.zero_grad()
            
            vae_loss.backward()
            dynamics_loss.backward()
            
            self.vae_optimizer.step()
            self.dynamics_optimizer.step()
            
            total_vae_loss += vae_loss.item()
            total_dynamics_loss += dynamics_loss.item()
            wandb.log({

            "vae_loss": vae_loss,
            "dynamics_loss": dynamics_loss
        })
        
        return total_vae_loss / len(self.dataloader), total_dynamics_loss / len(self.dataloader)

def main():
    # Initialize wandb
    wandb.init(project="vae-dynamics-car-navigation", name="training_run")
    print("initialized wandb\n\n")
    # Load dataset
    data = np.load("safe_rl_dataset_images.npz")
    print("initialized data variable\n\n")
    dataset = DynamicsDataset(data)
    print("created dataloader")
    # Initialize models
    vae = VAE(image_channels=3, latent_dim=32)
    dynamics = AffineDynamics(state_dim=4, action_dim=2)
    
    # Setup trainer
    trainer = Trainer(vae, dynamics, dataset, batch_size=128, lr=1e-4, device="mps")
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        vae_loss, dynamics_loss = trainer.train_step()
        
        # Log to wandb

        
        print(f"Epoch {epoch+1}/{num_epochs}, VAE Loss: {vae_loss:.4f}, Dynamics Loss: {dynamics_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'vae_state_dict': trainer.vae.state_dict(),
                'dynamics_state_dict': trainer.dynamics.state_dict(),
                'vae_optimizer_state_dict': trainer.vae_optimizer.state_dict(),
                'dynamics_optimizer_state_dict': trainer.dynamics_optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pt')
    
    print("Training complete")

if __name__ == "__main__":
    main()
