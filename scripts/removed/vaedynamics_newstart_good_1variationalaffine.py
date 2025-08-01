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
import random
# Network Utility Functions
rng = random.Random()  # This uses a new random state
random_value = rng.randint(100, 999)
def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
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



# [Previous utility functions and classes remain the same]

def compute_loss(model, batch, device, beta=1.0, wrec1=1.0, wrec2=1.0):
    states = batch['state_rgb'].to(device)
    actions = batch['action'].to(device)
    dones = batch['done'].to(device)
    
    batch_size, seq_length, channels, height, width = states.shape
    
    # Initialize variables
    loss_dyn = torch.zeros(1, device=device)
    loss_vae = torch.zeros(1, device=device)
    loss_rec1 = torch.zeros(1, device=device)  # Reconstruction of predicted observation
    loss_rec2 = torch.zeros(1, device=device)  # Direct reconstruction
    
    # Initial hidden state
    z_prev = torch.zeros(batch_size, model.dynamics.state_dim, device=device)
    
    # Track number of valid steps
    valid_steps = 0
    
    for t in range(1, seq_length):
        # Check if this is a valid step (not after a done)
        valid_mask = (dones[:, t-1] == 0).float()
        
        if valid_mask.sum() == 0:
            continue
        
        # Current states and actions for valid steps
        x_curr = states[:, t][valid_mask.bool()]
        action_curr = actions[:, t-1][valid_mask.bool()]
        z_prev_valid = z_prev[valid_mask.bool()]
        
        # Forward pass
        z, mu, logvar, z_pred, x_recon = model(x_curr, z_prev_valid, action_curr)
        
        # Dynamics loss (state prediction error)
        loss_dyn += F.mse_loss(z_pred, z)
        
        # VAE loss components
        recon_loss = F.mse_loss(x_recon, x_curr)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # New reconstruction losses
        # Reconstruct observation from predicted state
        predicted_state = z_pred
        x_predicted_recon = model.decode(predicted_state)
        loss_rec1 += F.mse_loss(x_predicted_recon, x_curr)
        
        # Direct reconstruction loss
        loss_rec2 += F.mse_loss(x_recon, x_curr)
        
        loss_vae += recon_loss + beta * kl_loss
        
        # Update previous hidden state
        z_prev[valid_mask == 1] = z
        
        valid_steps += 1
    
    # Avoid division by zero
    if valid_steps > 0:
        loss_dyn /= valid_steps
        loss_vae /= valid_steps
        loss_rec1 /= valid_steps
        loss_rec2 /= valid_steps
    
    # Compute total loss with weighted reconstruction terms
    total_loss = loss_dyn + loss_vae + wrec1 * loss_rec1 + wrec2 * loss_rec2
    
    # Log all loss components
    wandb.log({
        'total_loss': total_loss.item(),
        'dynamics_loss': loss_dyn.item(),
        'vae_loss': loss_vae.item(),
        'reconstruction_loss1': loss_rec1.item(),
        'reconstruction_loss2': loss_rec2.item()
    })
    
    return total_loss, loss_dyn, loss_vae, loss_rec1, loss_rec2

def validate(model, dataloader, device):
    """Validation function to compute validation loss."""
    model.eval()
    total_val_loss = 0.0
    val_loss_dyn = 0.0
    val_loss_vae = 0.0
    val_loss_rec1 = 0.0
    val_loss_rec2 = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            loss, loss_dyn, loss_vae, loss_rec1, loss_rec2 = compute_loss(
                model, batch, device
            )
            
            total_val_loss += loss.item()
            val_loss_dyn += loss_dyn.item()
            val_loss_vae += loss_vae.item()
            val_loss_rec1 += loss_rec1.item()
            val_loss_rec2 += loss_rec2.item()
    
    # Compute average validation losses
    num_batches = len(dataloader)
    avg_val_loss = total_val_loss / num_batches
    avg_val_loss_dyn = val_loss_dyn / num_batches
    avg_val_loss_vae = val_loss_vae / num_batches
    avg_val_loss_rec1 = val_loss_rec1 / num_batches
    avg_val_loss_rec2 = val_loss_rec2 / num_batches
    
    # Log validation metrics
    wandb.log({
        'val_total_loss': avg_val_loss,
        'val_dynamics_loss': avg_val_loss_dyn,
        'val_vae_loss': avg_val_loss_vae,
        'val_reconstruction_loss1': avg_val_loss_rec1,
        'val_reconstruction_loss2': avg_val_loss_rec2
    })
    
    return avg_val_loss

def train(model, train_loader, val_loader, optimizer, device, epochs=100):
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in trange(epochs):
        model.train()
        total_train_loss = 0.0
        
        # Training phase
        for batch in train_loader:
            optimizer.zero_grad()
            
            loss, loss_dyn, loss_vae, loss_rec1, loss_rec2 = compute_loss(model, batch, device)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        avg_val_loss = validate(model, val_loader, device)
        
        # Learning rate scheduling and early stopping could be added here
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {total_train_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_vae_dynamics_model.pth')
    
    return model

def main():
    # Initialize wandb
    seed_everything(1)
    wandb.init(project="vae-dynamics-car-navigation", name="training_run_with_validation")
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
    validation_split = 0.2
    
    # Create dataset
    full_dataset = DynamicsDataset(data, sequence_length=10)
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Loaded data\n")
    
    # Initialize model
    model = VAEDynamicsModel(
        input_channels=input_channels, 
        latent_dim=latent_dim, 
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model with validation
    trained_model = train(model, train_loader, val_loader, optimizer, device, epochs)
    
    # Save the final model
    torch.save(trained_model.state_dict(), 'final_vae_dynamics_model.pth')
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()