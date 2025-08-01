import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from tqdm import trange, tqdm

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Create a multi-layer perceptron with given sizes and activations."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class DynamicsDataset(Dataset):
    def __init__(self, data, sequence_length=10, prediction_horizon=5):
        """
        Initialize dataset with sequence creation and prediction horizon
        
        Args:
            data (dict): Dictionary containing actions, states, and done flags
            sequence_length (int): Total sequence length
            prediction_horizon (int): Number of steps to predict ahead
        """
        self.actions = torch.FloatTensor(data['actions'])
        self.states_rgb = torch.FloatTensor(data['states_rgb']).permute(0, 3, 1, 2) / 255.0
        self.dones = torch.FloatTensor(data['dones'])
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Prepare sequences respecting trajectory boundaries
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        """
        Create sequences that can be used for multiple shooting
        
        Returns:
            List of trajectory segments suitable for training
        """
        sequences = []
        
        i = 0
        while i < len(self.dones) - self.sequence_length:
            # Check trajectory boundary conditions
            sequence_actions = self.actions[i:i+self.sequence_length]
            sequence_states = self.states_rgb[i:i+self.sequence_length]
            sequence_dones = self.dones[i:i+self.sequence_length]
            
            # Skip sequences with done flags in the first prediction horizon
            if torch.any(sequence_dones[:self.prediction_horizon] == 1):
                i += 1
                continue
            
            sequence = {
                'actions': sequence_actions,
                'states_rgb': sequence_states,
                'dones': sequence_dones
            }
            
            sequences.append(sequence)
            i += 1
        
        return sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class RecursiveEncoder(nn.Module):
    """
    Recursive encoder that takes current observation, 
    previous latent state, and previous action
    """
    def __init__(self, input_channels, latent_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Image encoder
        self.image_encoder = nn.Sequential(
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
        
        # Fusion layer for image, previous state, and action
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
    
    def forward(self, current_obs, prev_latent_state, prev_action):
        """
        Encode current observation with context of previous state and action
        
        Args:
            current_obs (torch.Tensor): Current image observation
            prev_latent_state (torch.Tensor): Previous latent state
            prev_action (torch.Tensor): Previous action
        
        Returns:
            torch.Tensor: New latent state
        """
        # Encode current observation
        image_features = self.image_encoder(current_obs)
        
        # Concatenate image features, previous latent state, and previous action
        fusion_input = torch.cat([image_features, prev_latent_state, prev_action], dim=1)
        
        # Generate new latent state
        new_latent_state = self.fusion_layer(fusion_input)
        
        return new_latent_state

class DeterministicDecoder(nn.Module):
    """Decoder to reconstruct observations from latent states"""
    def __init__(self, latent_dim, input_channels, hidden_dim=64):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
    
    def forward(self, latent_state):
        """
        Reconstruct observation from latent state
        
        Args:
            latent_state (torch.Tensor): Latent representation
        
        Returns:
            torch.Tensor: Reconstructed observation
        """
        return self.decoder(latent_state)

class DeterministicDynamicsModel(nn.Module):
    """Deterministic Dynamics Model with Recursive Encoder and Decoder"""
    def __init__(self, input_channels, latent_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Recursive encoder
        self.encoder = RecursiveEncoder(
            input_channels, latent_dim, action_dim, hidden_dim
        )
        
        # Decoder for observation reconstruction
        self.decoder = DeterministicDecoder(
            latent_dim, input_channels, hidden_dim
        )
        
        # Dynamics prediction network
        self.dynamics_predictor = mlp(
            [latent_dim + action_dim, hidden_dim, hidden_dim, latent_dim], 
            activation=nn.ReLU
        )
    
    def forward(self, current_obs, prev_latent_state, prev_action):
        """
        Forward pass for dynamics prediction and observation reconstruction
        
        Args:
            current_obs (torch.Tensor): Current observation
            prev_latent_state (torch.Tensor): Previous latent state
            prev_action (torch.Tensor): Previous action
        
        Returns:
            tuple: New latent state, reconstructed observation, predicted next latent state
        """
        # Generate new latent state
        new_latent_state = self.encoder(current_obs, prev_latent_state, prev_action)
        
        # Reconstruct observation
        reconstructed_obs = self.decoder(new_latent_state)
        
        # Predict next latent state
        dynamics_input = torch.cat([new_latent_state, prev_action], dim=1)
        predicted_next_latent_state = self.dynamics_predictor(dynamics_input)
        
        return new_latent_state, reconstructed_obs, predicted_next_latent_state

def compute_multiple_shooting_loss(model, batch, device, 
                                    prediction_horizon=5, 
                                    wstate=1.0, 
                                    wrec1=1.0, 
                                    wrec2=1.0):
    """
    Compute multiple shooting loss as described in the paper
    
    Args:
        model (DeterministicDynamicsModel): Dynamics model
        batch (dict): Batch of trajectory data
        device (torch.device): Computation device
        prediction_horizon (int): Number of steps to predict ahead
        wstate, wrec1, wrec2 (float): Loss weights
    
    Returns:
        torch.Tensor: Total loss
    """
    states = batch['states_rgb'].to(device)
    actions = batch['actions'].to(device)
    
    batch_size, seq_length, channels, height, width = states.shape
    latent_dim = model.encoder.fusion_layer[-1].out_features
    
    total_loss = torch.zeros(1, device=device)
    
    # Initial zero latent state and zero action
    z_prev = torch.zeros(batch_size, latent_dim, device=device)
    prev_action = torch.zeros(batch_size, actions.shape[-1], device=device)
    
    for t in range(1, seq_length):
        # Current states and actions
        current_state = states[:, t]
        current_action = actions[:, t-1]
        
        # Forward pass
        z_current, x_recon_current, z_predicted = model(
            current_state, z_prev, prev_action
        )
        
        # Multiple shooting loss components
        # Prediction error for future states
        prediction_loss = F.mse_loss(z_predicted, z_current)
        
        # Reconstruction losses
        recon_loss1 = F.mse_loss(
            model.decoder(z_predicted),  # Predicted state reconstruction
            current_state
        )
        
        recon_loss2 = F.mse_loss(
            x_recon_current,  # Direct observation reconstruction
            current_state
        )
        
        # Accumulate losses with weights
        total_loss += (
            wstate * prediction_loss + 
            wrec1 * recon_loss1 + 
            wrec2 * recon_loss2
        )
        wandb.log({
            'train_wstate_loss': wstate,
            'train_wrec1_loss': wrec1,
            'train_wrec2_loss':wrec2
        })
        
        
        # Update previous state and action for next iteration
        z_prev = z_current
        prev_action = current_action
    
    # Normalize loss
    total_loss /= (seq_length - 1)
    
    return total_loss

def train(model, train_loader, val_loader, optimizer, device, epochs=100):
    """Training function with validation"""
    model.to(device)
    best_val_loss = float('inf')
    
    wandb.init(project="deterministic-dynamics-model", name="training_run")
    
    for epoch in trange(epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            loss = compute_multiple_shooting_loss(model, batch, device)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss = compute_multiple_shooting_loss(model, batch, device)
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Logging

        # Model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_dynamics_model.pth')
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
    
    wandb.finish()
    return model

def main():
    # Load dataset
    data = np.load("safe_rl_dataset_images.npz")
    
    # Hyperparameters
    input_channels = 3
    latent_dim = 32
    action_dim = 2
    hidden_dim = 64
    batch_size = 32
    learning_rate = 1e-4
    epochs = 100
    validation_split = 0.2
    
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create dataset
    full_dataset = DynamicsDataset(data, sequence_length=10, prediction_horizon=5)
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = DeterministicDynamicsModel(
        input_channels=input_channels, 
        latent_dim=latent_dim, 
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    trained_model = train(model, train_loader, val_loader, optimizer, device, epochs)
    
    # Save final model
    torch.save(trained_model.state_dict(), 'final_dynamics_model.pth')

if __name__ == "__main__":
    main()