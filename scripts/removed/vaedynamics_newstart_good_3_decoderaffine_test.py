import torch
import numpy as np
from torch.utils.data import DataLoader
from vaedynamics_newstart_good_3_decoderaffine import DeterministicDynamicsModel, DynamicsDataset  # Import your model definition
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

def simulate_trajectory(model, initial_state, initial_action, sequence_length, device):
    """
    Simulates a trajectory using the learned dynamics model and outputs latent vectors.

    Args:
        model (DeterministicDynamicsModel): Trained dynamics model.
        initial_state (torch.Tensor): Initial state (image).
        initial_action (torch.Tensor): Initial action.
        sequence_length (int): Length of the trajectory to simulate.
        device (torch.device): Device to run the simulation on.

    Returns:
        tuple:
            - list: List of simulated states (images).
            - list: List of corresponding latent vectors.
    """
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    simulated_states = [initial_state.cpu().numpy()]  # Store initial state
    latent_vectors = []
    current_state = initial_state.unsqueeze(0).to(device)  # Add batch dimension
    prev_action = initial_action.unsqueeze(0).to(device)  # Add batch dimension
    latent_dim = model.encoder.fusion_layer[-1].out_features
    z_prev = torch.zeros(1, latent_dim, device=device) # Initial latent state

    with torch.no_grad():
        for _ in range(sequence_length - 1):
            # Encode, predict, and decode
            z_current, _, z_predicted = model(current_state, z_prev, prev_action)
            predicted_state = model.decoder(z_predicted)

            # Store the simulated state and latent vector
            simulated_states.append(predicted_state.squeeze(0).cpu().numpy())  # Remove batch dimension
            latent_vectors.append(z_current.squeeze().cpu().numpy())

            # Update for the next step
            current_state = predicted_state
            prev_action = initial_action.unsqueeze(0).to(device)  # Use same action for simplicity
            z_prev = z_current

    return simulated_states, latent_vectors


def main():
    # Load dataset (or create a dummy dataset)
    data = np.load("safe_rl_dataset_images.npz")

    # Hyperparameters (must match training)
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
    model.load_state_dict(torch.load('CCBF_images_model.pth', map_location=torch.device('cpu'))) # Load the best weights
    model.eval()  # Set the model to evaluation mode

    # Get an initial state and action from the validation dataset
    initial_batch = next(iter(val_loader))
    initial_state = initial_batch['states_rgb'][0]  # Get the first state from the batch
    initial_action = initial_batch['actions'][0]    # Get the first action from the batch

    # Simulation parameters
    sequence_length = 5  # Length of the trajectory to simulate (reduced for testing)

    # Simulate trajectory
    simulated_trajectory, latent_vectors = simulate_trajectory(model, initial_state, initial_action, sequence_length, device)

    # Visualize the trajectory (example)
    num_plots = len(simulated_trajectory)
    fig, axes = plt.subplots(2, num_plots, figsize=(5 * num_plots, 10))  # Increased height for two rows
    fig.suptitle("Simulated Trajectory: Input vs. Output", fontsize=16)

    # Plot initial input image
    initial_img = initial_state.cpu().numpy()
    initial_img = np.transpose(initial_img, (1, 2, 0))  # Convert to (H, W, C) for display
    axes[0, 0].imshow(initial_img)
    axes[0, 0].set_title("Initial Input")
    axes[0, 0].axis('off')

    for i, state in enumerate(simulated_trajectory):
        # Assuming the state is an image (C, H, W)
        state_img = np.transpose(state, (1, 2, 0))  # Convert to (H, W, C) for display
        ax_col = i + 1  # Start from the second column

        if i == 0:
            ax_col = 0
            axes[1, ax_col].imshow(state_img)
            axes[1, ax_col].set_title(f"Reconstructed State {i+1}")
            axes[1, ax_col].axis('off')
        else:
            axes[1, i].imshow(state_img)
            axes[1, i].set_title(f"Reconstructed State {i+1}")
            axes[1, i].axis('off')

            # Display images in the top row
            axes[0, i].imshow(state_img)
            axes[0, i].set_title(f"Input State {i+1}")
            axes[0, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
    plt.show()

    # Output Latent Vectors
    print("\nLatent Vectors:")
    for i, latent_vector in enumerate(latent_vectors):
        print(f"State {i+1}: {latent_vector}")

if __name__ == "__main__":
    main()
