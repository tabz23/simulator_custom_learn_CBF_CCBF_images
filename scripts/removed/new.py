import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision.utils import make_grid

# Import your model architecture
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Recreate the model architecture (copy from your training script)
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
        
        def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
            """Create a multi-layer perceptron with given sizes and activations."""
            layers = []
            for j in range(len(sizes)-1):
                act = activation if j < len(sizes)-2 else output_activation
                layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
            return nn.Sequential(*layers)
        
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
        
        # Dynamics prediction network (using AffineDynamics)
        self.dynamics_predictor = AffineDynamics(
            num_action=action_dim,
            state_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(self, current_obs, prev_latent_state, prev_action):
        # Generate new latent state
        new_latent_state = self.encoder(current_obs, prev_latent_state, prev_action)
        
        # Reconstruct observation
        reconstructed_obs = self.decoder(new_latent_state)
        
        # Predict next latent state from previous latent state
        predicted_latent_state = self.dynamics_predictor.forward_next_state(prev_latent_state, prev_action)
        
        return new_latent_state, reconstructed_obs, predicted_latent_state

def compute_losses(model, current_state, z_prev, prev_action, device):
    """
    Compute the same losses used during training for a single step
    
    Args:
        model: The dynamics model
        current_state: Current observation
        z_prev: Previous latent state
        prev_action: Previous action
        device: Computation device
    
    Returns:
        dict: Dictionary containing individual loss components
    """
    # Forward pass
    z_current, x_recon_current, z_predicted = model(current_state, z_prev, prev_action)
    
    # Same loss components as in training
    # Prediction error (z_predicted vs z_current)
    prediction_loss = F.mse_loss(z_predicted, z_current)
    
    # Reconstruction loss 1 (decoded prediction vs actual)
    predicted_recon = model.decoder(z_predicted)
    recon_loss1 = F.mse_loss(predicted_recon, current_state)
    
    # Reconstruction loss 2 (direct reconstruction vs actual)
    recon_loss2 = F.mse_loss(x_recon_current, current_state)
    print(current_state.shape)
    
    
    # Loss weights (same as in training function)
    wstate = 100.0
    wrec1 = 1.0
    wrec2 = 1.0
    
    # Total loss
    total_loss = wstate * prediction_loss + wrec1 * recon_loss1 + wrec2 * recon_loss2
    
    return {
        'prediction_loss': prediction_loss.item(),
        'recon_loss1': recon_loss1.item(),
        'recon_loss2': recon_loss2.item(),
        'weighted_prediction_loss': wstate * prediction_loss.item(),
        'weighted_recon_loss1': wrec1 * recon_loss1.item(),
        'weighted_recon_loss2': wrec2 * recon_loss2.item(),
        'total_loss': total_loss.item(),
        'z_current': z_current.detach(),
        'x_recon': x_recon_current.detach()
    }

def visualize_sequence_with_losses(model, sequence_data, device, save_path="sequence_visualization.png"):
    """
    Visualize a sequence of original images and their reconstructions with losses
    
    Args:
        model: Trained dynamics model
        sequence_data: Dictionary containing a sequence of states and actions
        device: Computation device
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Get sequence data
    states = sequence_data['states_rgb'].to(device)
    actions = sequence_data['actions'].to(device)
    
    # Get sequence length and latent dimension
    seq_length = states.shape[1]
    latent_dim = model.encoder.fusion_layer[-1].out_features
    
    # Initialize latent state and action
    z_prev = torch.zeros(1, latent_dim, device=device)
    prev_action = torch.zeros(1, actions.shape[-1], device=device)
    
    # Storage for reconstructed images and losses
    reconstructed_states = []
    all_losses = []
    
    # Process each step in the sequence
    for t in range(seq_length):
        current_state = states[:, t]
        current_action = actions[:, t] if t < actions.shape[1] else prev_action
        
        # Compute losses and get reconstructed state
        losses = compute_losses(model, current_state, z_prev, prev_action, device)
        all_losses.append(losses)
        
        # Store reconstructed state
        reconstructed_states.append(losses['x_recon'].cpu())
        
        # Update previous state and action
        z_prev = losses['z_current']
        prev_action = current_action
    
    # Convert lists to tensors
    original_states = [states[:, t].cpu() for t in range(seq_length)]
    
    # Print losses for each step
    print("\n===== Loss Details for Each Step =====")
    total_seq_loss = 0
    for t, loss_dict in enumerate(all_losses):
        print(f"\nStep {t+1}:")
        print(f"  Prediction Loss: {loss_dict['prediction_loss']:.6f} (weighted: {loss_dict['weighted_prediction_loss']:.6f})")
        print(f"  Reconstruction Loss 1: {loss_dict['recon_loss1']:.6f} (weighted: {loss_dict['weighted_recon_loss1']:.6f})")
        print(f"  Reconstruction Loss 2: {loss_dict['recon_loss2']:.6f} (weighted: {loss_dict['weighted_recon_loss2']:.6f})")
        print(f"  Total Step Loss: {loss_dict['total_loss']:.6f}")
        total_seq_loss += loss_dict['total_loss']
    
    # Print average loss across sequence
    avg_loss = total_seq_loss / seq_length
    print("\n===== Sequence Summary =====")
    print(f"Average Loss per Step: {avg_loss:.6f}")
    print(f"Total Sequence Loss: {total_seq_loss:.6f}")
    
    # Create the figure for visualization
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(2, seq_length, figure=fig)
    
    # Helper function to convert tensor to displayable image
    def tensor_to_img(tensor):
        # Convert from BCHW to HWC and scale to [0, 255]
        img = tensor.squeeze().permute(1, 2, 0).numpy() * 255
        return img.astype(np.uint8)
    
    # Plot original and reconstructed states
    for t in range(seq_length):
        # Original image in top row
        ax1 = fig.add_subplot(gs[0, t])
        ax1.imshow(tensor_to_img(original_states[t]))
        ax1.set_title(f"Original {t+1}")
        ax1.axis('off')
        
        # Reconstructed image in bottom row
        ax2 = fig.add_subplot(gs[1, t])
        ax2.imshow(tensor_to_img(reconstructed_states[t]))
        ax2.set_title(f"Recon {t+1}\nLoss: {all_losses[t]['total_loss']:.4f}")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {save_path}")
    
    return fig

def main():
    # Hyperparameters (same as in your training script)
    input_channels = 3
    latent_dim = 4
    action_dim = 2
    hidden_dim = 64
    
    # Device configuration - adapt based on availability
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    data_path = "safe_rl_dataset_images_ALLUNSAFE.npz"
    data = np.load(data_path)
    
    # Create a single sequence for visualization
    seq_length = 10
    
    # Find a good starting point for a sequence (avoiding done flags)
    start_idx = 0
    found = False
    
    # Look for a sequence without done flags
    while not found and start_idx < len(data['dones']) - seq_length:
        if not any(data['dones'][start_idx:start_idx+seq_length]):
            found = True
        else:
            start_idx += 1
    
    if not found:
        print("Warning: Could not find a clean sequence. Using first available.")
        start_idx = 0
    
    # Extract sequence data
    sequence_data = {
        'states_rgb': torch.FloatTensor(data['states_rgb'][start_idx:start_idx+seq_length]).permute(0, 3, 1, 2) / 255.0,
        'actions': torch.FloatTensor(data['actions'][start_idx:start_idx+seq_length]),
        'dones': torch.FloatTensor(data['dones'][start_idx:start_idx+seq_length])
    }
    
    # Add batch dimension
    sequence_data['states_rgb'] = sequence_data['states_rgb'].unsqueeze(0)
    sequence_data['actions'] = sequence_data['actions'].unsqueeze(0)
    sequence_data['dones'] = sequence_data['dones'].unsqueeze(0)
    
    # Initialize model
    model = DeterministicDynamicsModel(
        input_channels=input_channels, 
        latent_dim=latent_dim, 
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    # Load the trained model weights
    model_path = "CCBF_images_model_final_fixedaction.pth"  # Adjust if needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print(f"Loaded model from {model_path}")
    print(f"Processing sequence starting at index {start_idx} in the dataset")
    
    # Visualize sequence with losses
    visualization = visualize_sequence_with_losses(model, sequence_data, device)
    
    print("Sequence visualization and loss calculation complete!")

if __name__ == "__main__":
    main()