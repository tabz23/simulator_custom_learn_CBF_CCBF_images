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

class DynamicsDataset(Dataset):
    def __init__(self, data, seq_length):
        self.seq_length = seq_length
        self.actions = torch.FloatTensor(data['actions'])
        self.states_rgb = torch.FloatTensor(data['states_rgb']).permute(0, 3, 1, 2) / 255.0
        self.dones = torch.FloatTensor(data['dones'])
        
        # Make sure the dataset size is large enough for the sequences
        self.num_samples = len(self.states_rgb) - self.seq_length + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create sequences of actions, states, and dones
        seq_actions = self.actions[idx:idx + self.seq_length]
        seq_states_rgb = self.states_rgb[idx:idx + self.seq_length]
        seq_dones = self.dones[idx:idx + self.seq_length]
        
        return {
            'action': seq_actions,
            'state_rgb': seq_states_rgb,
            'done': seq_dones##note that done determines that this sample is last in the current trajectory, so next sample is a new one.
        }
class AffineDynamics(nn.Module):##this should take in the latent state produced by the encoder
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
    
    
def main():
    # Initialize wandb
    wandb.init(project="vae-dynamics-car-navigation", name="training_run")
    print("initialized wandb\n\n")
    
    # Load dataset
    data = np.load("safe_rl_dataset_images.npz")
    print("initialized data variable\n\n")
    
    # Set the sequence length (e.g., 10)
    seq_length = 10
    dataset = DynamicsDataset(data, seq_length)
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Iterate through DataLoader and print shapes of batch elements
    for batch in dataloader:
        print("Batch shapes:")
        print(f"Action shape: {batch['action'].shape}")
        print(f"State shape: {batch['state_rgb'].shape}")
        print(f"Done shape: {batch['done'].shape}")
'''
Batch shapes:
Action shape: torch.Size([32, 10, 2])
State shape: torch.Size([32, 10, 3, 64, 64])
Done shape: torch.Size([32, 10])
'''

if __name__ == "__main__":
    main()
