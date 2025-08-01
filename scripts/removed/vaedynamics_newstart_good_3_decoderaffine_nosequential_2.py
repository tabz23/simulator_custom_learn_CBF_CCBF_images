import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from modules.network import CBF  # Assuming this is still relevant
import random
from tqdm import trange, tqdm
import torchmetrics

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
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
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
        self.ground_truth_states = torch.FloatTensor(data['states'])

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
            sequence_actions = self.actions[i:i + self.sequence_length]
            sequence_states = self.states_rgb[i:i + self.sequence_length]
            sequence_dones = self.dones[i:i + self.sequence_length]

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


class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, hidden_dim=64):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Output: 64x64
            nn.Sigmoid()  # Constrain output to [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


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

        self.f = mlp([self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim], activation=nn.ReLU)
        self.g = mlp(
            [self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim * self.num_action],
            activation=nn.ReLU)

    def forward(self, state):
        return self.f(state), self.g(state)

    def forward_x_dot(self, state, action):
        f, g = self.forward(state)
        gu = torch.einsum('bsa,ba->bs', g.view(g.shape[0], self.state_dim, self.num_action), action)
        x_dot = f + gu
        return x_dot

    def forward_next_state(self, state, action):
        return self.forward_x_dot(state, action) * self.dt + state


class DynamicsModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=64):
        super(DynamicsModel, self).__init__()
        self.dynamics = AffineDynamics(
            num_action=action_dim,
            state_dim=latent_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        next_state = self.dynamics.forward_next_state(state, action)
        return next_state


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence


def train(vae, dynamics_model, train_loader, val_loader, vae_optimizer, dynamics_optimizer, device, epochs=100,
        dynamics_weight=0.1):
    """Training function with separate VAE and Dynamics training"""
    vae.to(device)
    dynamics_model.to(device)
    best_val_loss = float('inf')

    wandb.init(project="vae-dynamics-model", name="training_run")

    for epoch in trange(epochs):
        vae.train()
        dynamics_model.train()
        total_vae_loss = 0.0
        total_dynamics_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            states = batch['states_rgb'].to(device)  # Shape: [batch_size, sequence_length, channels, height, width]
            actions = batch['actions'].to(device)    # Shape: [batch_size, sequence_length, action_dim]
            '''states  torch.Size([32, 6, 3, 64, 64])
            actions torch.Size([32, 6, 2])
'''
            # VAE Training: Train on EACH image in the sequence independently
            vae_loss_seq = 0.0
            for t in range(states.shape[1]):  # Iterate through the sequence length
                current_state = states[:, t, :, :, :]  # Shape: [batch_size, channels, height, width]
                vae_optimizer.zero_grad()
                recon_x, mu, logvar, _ = vae(current_state)
                loss_vae = vae_loss(recon_x, current_state, mu, logvar)
                loss_vae.backward()
                vae_optimizer.step()
                vae_loss_seq += loss_vae.item()

            total_vae_loss += vae_loss_seq / states.shape[1]  # Average loss over the sequence

            # Dynamics Training - Properly iterate through the sequence
            dynamics_optimizer.zero_grad()
            dynamics_loss_seq = 0.0
            '''
            Batch shapes:
  states shape: torch.Size([32, 6, 3, 64, 64])
  actions shape: torch.Size([32, 6, 2])
  current_state shape: torch.Size([32, 3, 64, 64])
  recon_x shape: torch.Size([32, 3, 64, 64])
  mu shape: torch.Size([32, 4])
  logvar shape: torch.Size([32, 4])
  current_state shape: torch.Size([32, 3, 64, 64])
  recon_x shape: torch.Size([32, 3, 64, 64])
  mu shape: torch.Size([32, 4])
  logvar shape: torch.Size([32, 4])
  current_state shape: torch.Size([32, 3, 64, 64])
  recon_x shape: torch.Size([32, 3, 64, 64])
  mu shape: torch.Size([32, 4])
  logvar shape: torch.Size([32, 4])
  current_state shape: torch.Size([32, 3, 64, 64])
  recon_x shape: torch.Size([32, 3, 64, 64])
  mu shape: torch.Size([32, 4])
  logvar shape: torch.Size([32, 4])
  current_state shape: torch.Size([32, 3, 64, 64])
  recon_x shape: torch.Size([32, 3, 64, 64])
  mu shape: torch.Size([32, 4])
  logvar shape: torch.Size([32, 4])
  current_state shape: torch.Size([32, 3, 64, 64])
  recon_x shape: torch.Size([32, 3, 64, 64])
  mu shape: torch.Size([32, 4])
  logvar shape: torch.Size([32, 4])
  '''
            '''
            Dynamics step 0:
  current_state shape: torch.Size([32, 3, 64, 64])
  current_action shape: torch.Size([32, 2])
  next_state shape: torch.Size([32, 3, 64, 64])
  current_latent shape: torch.Size([32, 4])
  next_latent_gt shape: torch.Size([32, 4])
  predicted_next_latent shape: torch.Size([32, 4])
  '''
            # We need sequence_length - 1 transitions (current -> next)
            for t in range(states.shape[1] - 1):
                # Current state at time t
                current_state = states[:, t, :, :, :]
                # The action taken at time t
                current_action = actions[:, t]
                
                # The next state at time t+1
                next_state = states[:, t+1, :, :, :]
                
                # Encode both states to get their latent representations
                _, _, _, current_latent = vae(current_state)
                _, _, _, next_latent_gt = vae(next_state)
                
                # Predict the next latent state using the dynamics model
                predicted_next_latent = dynamics_model(current_latent, current_action)
                
                # Calculate loss for this step in the sequence
                step_dynamics_loss = F.mse_loss(predicted_next_latent, next_latent_gt)
                dynamics_loss_seq += step_dynamics_loss
            
            # Average the dynamics loss across the sequence steps
            avg_dynamics_loss = dynamics_loss_seq / (states.shape[1] - 1)
            # Apply weight and backpropagate
            weighted_dynamics_loss = dynamics_weight * avg_dynamics_loss
            weighted_dynamics_loss.backward()
            dynamics_optimizer.step()
            
            total_dynamics_loss += weighted_dynamics_loss.item()

        # Validation (Now including dynamics validation)
        vae.eval()
        dynamics_model.eval()
        total_val_vae_loss = 0.0
        total_val_dynamics_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                states = batch['states_rgb'].to(device)
                actions = batch['actions'].to(device)

                val_vae_loss_seq = 0.0
                for t in range(states.shape[1]):  # Iterate through the sequence length
                    current_state = states[:, t, :, :, :]
                    recon_x, mu, logvar, _ = vae(current_state)
                    loss_vae = vae_loss(recon_x, current_state, mu, logvar)
                    val_vae_loss_seq += loss_vae.item()

                total_val_vae_loss += val_vae_loss_seq / states.shape[1]
                
                # Validate dynamics model
                val_dynamics_loss_seq = 0.0
                for t in range(states.shape[1] - 1):
                    current_state = states[:, t, :, :, :]
                    current_action = actions[:, t]
                    next_state = states[:, t+1, :, :, :]
                    
                    _, _, _, current_latent = vae(current_state)
                    _, _, _, next_latent_gt = vae(next_state)
                    
                    predicted_next_latent = dynamics_model(current_latent, current_action)
                    step_dynamics_loss = F.mse_loss(predicted_next_latent, next_latent_gt)
                    val_dynamics_loss_seq += step_dynamics_loss
                
                avg_val_dynamics_loss = val_dynamics_loss_seq / (states.shape[1] - 1)
                total_val_dynamics_loss += avg_val_dynamics_loss.item()

        avg_train_vae_loss = total_vae_loss / len(train_loader)
        avg_train_dynamics_loss = total_dynamics_loss / len(train_loader)
        avg_val_vae_loss = total_val_vae_loss / len(val_loader)
        avg_val_dynamics_loss = total_val_dynamics_loss / len(val_loader)

        wandb.log({
            'epoch': epoch + 1,
            'train_vae_loss': avg_train_vae_loss,
            'train_dynamics_loss': avg_train_dynamics_loss,
            'val_vae_loss': avg_val_vae_loss,
            'val_dynamics_loss': avg_val_dynamics_loss
        })

        # Model checkpoint based on combined validation loss
        combined_val_loss = avg_val_vae_loss + dynamics_weight * avg_val_dynamics_loss
        if combined_val_loss < best_val_loss:
            best_val_loss = combined_val_loss
            torch.save(vae.state_dict(), 'vae_model_final_2.pth')
            torch.save(dynamics_model.state_dict(), 'dynamics_model_final_2.pth')
            print("saved VAE and Dynamics model")

        print(
            f"Epoch {epoch + 1}: Train VAE Loss {avg_train_vae_loss:.4f}, Train Dynamics Loss {avg_train_dynamics_loss:.4f}, "
            f"Val VAE Loss {avg_val_vae_loss:.4f}, Val Dynamics Loss {avg_val_dynamics_loss:.4f}")

    wandb.finish()
    return vae, dynamics_model


def main():
    # Load dataset
    data = np.load("safe_rl_dataset_images_ALLUNSAFE_big_obstacle_3000traj.npz.npz")
    seed_everything(1)

    # Hyperparameters
    input_channels = 3
    latent_dim = 4
    action_dim = 2
    hidden_dim = 400
    batch_size = 32
    learning_rate_vae = 1e-4
    learning_rate_dynamics = 1e-4
    epochs = 50
    validation_split = 0.2
    dynamics_weight = 0.1  # Weight for the dynamics loss

    # Device configuration
    device = "mps"

    # Create dataset
    full_dataset = DynamicsDataset(data, sequence_length=6, prediction_horizon=6)

    # Split into train and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    vae = VAE(input_channels=input_channels, latent_dim=latent_dim, hidden_dim=hidden_dim)
    dynamics_model = DynamicsModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    # Optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=learning_rate_vae)
    dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=learning_rate_dynamics)

    # Train the model
    trained_vae, trained_dynamics = train(vae, dynamics_model, train_loader, val_loader, vae_optimizer,
                                          dynamics_optimizer, device, epochs, dynamics_weight)

    # Save final model
    torch.save(trained_vae.state_dict(), 'final_vae_model_2.pth')
    torch.save(trained_dynamics.state_dict(), 'final_dynamics_model_2.pth')


if __name__ == "__main__":
    main()