import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
# Reuse these classes from your original code

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

        # Direct mapping to latent space (no distribution)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

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
        z = self.fc_latent(h)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AffineDynamics(torch.nn.Module):
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

        self.f = self.mlp([self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim], 
                          activation=torch.nn.ReLU)
        self.g = self.mlp(
            [self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim * self.num_action],
            activation=torch.nn.ReLU)

    def mlp(self, sizes, activation=torch.nn.ReLU, output_activation=torch.nn.Identity):
        """Create a multi-layer perceptron with given sizes and activations."""
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
        return torch.nn.Sequential(*layers)

    def forward(self, state):
        return self.f(state), self.g(state)

    def forward_x_dot(self, state, action):
        f, g = self.forward(state)
        gu = torch.einsum('bsa,ba->bs', g.view(g.shape[0], self.state_dim, self.num_action), action)
        x_dot = f + gu
        return x_dot

    def forward_next_state(self, state, action):
        return self.forward_x_dot(state, action) * self.dt + state

class DynamicsModel(torch.nn.Module):
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

class DynamicsDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length=10, prediction_horizon=5):
        self.actions = torch.FloatTensor(data['actions'])
        self.states_rgb = torch.FloatTensor(data['states_rgb']).permute(0, 3, 1, 2) / 255.0
        self.dones = torch.FloatTensor(data['dones'])
        self.ground_truth_states = torch.FloatTensor(data['states'])

        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Prepare sequences respecting trajectory boundaries
        self.sequences = self._create_sequences()

    def _create_sequences(self):
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

def visualize_vae_and_dynamics():
    # Load dataset
    data = np.load("safe_rl_dataset_images_ALLUNSAFE_big_obstacle.npz")
    
    # Hyperparameters (same as in your original code)
    input_channels = 3
    latent_dim = 4
    action_dim = 2
    hidden_dim = 400
    batch_size = 1  # Set to 1 for easy visualization
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    full_dataset = DynamicsDataset(data, sequence_length=6, prediction_horizon=6)
    
    # Create data loader - just use a few samples for visualization
    vis_size = min(3, len(full_dataset))
    vis_indices = np.random.choice(len(full_dataset), vis_size, replace=False)
    vis_dataset = torch.utils.data.Subset(full_dataset, vis_indices)
    vis_loader = DataLoader(vis_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize VAE model and load weights
    vae = VAE(input_channels=input_channels, latent_dim=latent_dim, hidden_dim=hidden_dim)
    
    # Initialize Dynamics model and load weights
    dynamics_model = DynamicsModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    
    # Try different checkpoint filenames
    # vae_checkpoint_files = ['vae_model_final.pth', 'final_vae_model.pth']
    # dynamics_checkpoint_files = ['dynamics_model_final.pth', 'final_dynamics_model.pth']
    
    # vae_checkpoint_files = ['vae_model_final_2.pth', 'final_vae_model_2.pth']
    # dynamics_checkpoint_files = ['dynamics_model_final_2.pth', 'final_dynamics_model_2.pth']
    
    vae_checkpoint_files = ['vae_model_final_4.pth', 'final_vae_model_4.pth']
    dynamics_checkpoint_files = ['dynamics_model_final_4.pth', 'final_dynamics_model_4.pth']
    
    
    vae_loaded = False
    for ckpt_file in vae_checkpoint_files:
        if os.path.exists(ckpt_file):
            print(f"Loading VAE checkpoint from {ckpt_file}")
            vae.load_state_dict(torch.load(ckpt_file, map_location=device))
            vae_loaded = True
            break
    
    dynamics_loaded = False
    for ckpt_file in dynamics_checkpoint_files:
        if os.path.exists(ckpt_file):
            print(f"Loading Dynamics checkpoint from {ckpt_file}")
            dynamics_model.load_state_dict(torch.load(ckpt_file, map_location=device))
            dynamics_loaded = True
            break
    
    if not vae_loaded:
        raise FileNotFoundError(f"Could not find any VAE checkpoint files: {vae_checkpoint_files}")
    if not dynamics_loaded:
        raise FileNotFoundError(f"Could not find any Dynamics checkpoint files: {dynamics_checkpoint_files}")
    
    vae.to(device)
    dynamics_model.to(device)
    vae.eval()
    dynamics_model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(vis_loader):
            # Get the sequence of states and actions
            states_seq = batch['states_rgb'].to(device)  # [batch_size, sequence_length, channels, height, width]
            actions_seq = batch['actions'].to(device)    # [batch_size, sequence_length, action_dim]
            
            batch_size, seq_length = states_seq.shape[0], states_seq.shape[1]
            
            # Create figure for this batch
            fig, axs = plt.subplots(2, seq_length, figsize=(18, 6))
            
            # Store latent states for comparison
            actual_latents = []
            predicted_latents = []
            
            # Process each sequence in the batch
            for b in range(batch_size):
                print(f"\n===== Batch {batch_idx+1}, Sequence {b+1} =====")
                
                # First pass: encode all frames to get actual latent states
                for t in range(seq_length):
                    current_state = states_seq[b, t].unsqueeze(0)  # [1, channels, height, width]
                    _, z = vae(current_state)
                    actual_latents.append(z.cpu().numpy())
                
                # Second pass: predict latent states using dynamics model and visualize results
                predicted_z = None
                
                for t in range(seq_length):
                    # Get current state
                    current_state = states_seq[b, t].unsqueeze(0)  # [1, channels, height, width]
                    
                    # Encode the current state
                    _, z = vae(current_state)
                    
                    # If t > 0, we should have a prediction to compare
                    if t > 0:
                        print(f"Timestep {t}:")
                        print(f"  Actual latent:    {actual_latents[t][0]}")
                        print(f"  Predicted latent: {predicted_z[0]}")
                        
                        # Calculate prediction error
                        error = np.mean(np.abs(actual_latents[t][0] - predicted_z[0]))
                        print(f"  Mean prediction error: {error:.6f}")
                    
                    # Save the current prediction for the next timestep
                    if t < seq_length - 1:
                        # Get the action
                        current_action = actions_seq[b, t].unsqueeze(0)  # [1, action_dim]
                        
                        # Predict next latent state
                        predicted_z = dynamics_model(z, current_action).cpu().numpy()
                        predicted_latents.append(predicted_z)
                    
                    # Get reconstruction
                    recon, _= vae(current_state)
                    
                    # Convert tensors to numpy for plotting
                    input_img = current_state.cpu().squeeze().permute(1, 2, 0).numpy()  # [height, width, channels]
                    recon_img = recon.cpu().squeeze().permute(1, 2, 0).numpy()  # [height, width, channels]
                    
                    # Plot original (top row)
                    axs[0, t].imshow(input_img)
                    axs[0, t].set_title(f"Original t={t}")
                    axs[0, t].axis('off')
                    
                    # Plot reconstruction (bottom row)
                    axs[1, t].imshow(recon_img)
                    axs[1, t].set_title(f"Reconstructed t={t}")
                    axs[1, t].axis('off')
            
            # Add super titles
            plt.suptitle(f"VAE Reconstruction and Dynamics Prediction (Batch {batch_idx+1})", fontsize=16)
            fig.text(0.5, 0.01, "Top row: Original images, Bottom row: VAE reconstructions", 
                     ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.1)
            plt.savefig(f"vae_dynamics_visualization_batch_{batch_idx+1}.png", dpi=150, bbox_inches='tight')
            plt.show()

if __name__ == "__main__":
    visualize_vae_and_dynamics()