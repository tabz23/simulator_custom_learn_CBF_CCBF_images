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
import seaborn as sns
from itertools import islice
# Import the classes and functions from the original script


def visualize_latent_space_and_reconstructions(model, val_loader, device, num_samples=5):
    """
    Visualize latent space representations and image reconstructions
    
    Args:
        model (DeterministicDynamicsModel): Trained dynamics model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Computation device
        num_samples (int): Number of samples to visualize
    """
    model.eval()
    
    # Create figure for visualization
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(islice(val_loader, 1)):
            if batch_idx >= num_samples:
                break
            
            # Get data from the batch
            states = batch['states_rgb'].to(device)
            actions = batch['actions'].to(device)
            
            # Initial zero latent state and zero action
            batch_size = states.shape[0]
            latent_dim = model.encoder.fusion_layer[-1].out_features
            z_prev = torch.zeros(batch_size, latent_dim, device=device)
            prev_action = torch.zeros(batch_size, actions.shape[-1], device=device)
            
            # Process the first sequence
            for t in range(1, states.shape[1]):
                current_state = states[:, t]
                current_action = actions[:, t]###changed this
                
                # Forward pass
                z_current, x_recon_current, z_predicted = model(
                    current_state, z_prev, prev_action
                )
                
                # Visualization for the first sample in the batch
                sample_idx = 0
                
                # Original image
                axs[batch_idx, 0].imshow(states[sample_idx, t].cpu().permute(1, 2, 0))
                axs[batch_idx, 0].set_title(f'Original Image (t={t})')
                axs[batch_idx, 0].axis('off')
                
                # Reconstructed image
                reconstructed_img = x_recon_current[sample_idx].cpu().permute(1, 2, 0)
                axs[batch_idx, 1].imshow(reconstructed_img)
                axs[batch_idx, 1].set_title('Reconstructed Image')
                axs[batch_idx, 1].axis('off')
                
                # # Latent space visualization
                # latent_heatmap = z_current[sample_idx].cpu().numpy()
                # sns.heatmap(latent_heatmap.reshape(4, 1), 
                #             cmap='viridis', 
                #             ax=axs[batch_idx, 2], 
                #             cbar=False)
                # axs[batch_idx, 2].set_title('Latent Space Representation')
                
                # # Predicted vs Actual Latent Space
                # latent_diff = torch.abs(z_current[sample_idx] - z_predicted[sample_idx]).cpu().numpy()
                # sns.heatmap(latent_diff.reshape(4, 8), 
                #             cmap='coolwarm', 
                #             ax=axs[batch_idx, 3], 
                #             cbar=False)
                # axs[batch_idx, 3].set_title('Latent Space Difference')
                
                # Update for next iteration
                z_prev = z_current
                prev_action = current_action
    
    plt.tight_layout()
    plt.savefig('latent_space_visualization.png')
    plt.close()

def analyze_latent_state_prediction(model, val_loader, device):
    """
    Analyze the accuracy of latent state predictions
    
    Args:
        model (DeterministicDynamicsModel): Trained dynamics model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Computation device
    
    Returns:
        dict: Metrics about latent state prediction
    """
    model.eval()
    
    prediction_metrics = {
        'mse_loss': [],
        'cosine_similarity': [],
        'l1_loss': []
    }
    
    with torch.no_grad():
        for batch in val_loader:
            states = batch['states_rgb'].to(device)
            actions = batch['actions'].to(device)
            
            # Initial zero latent state and zero action
            batch_size = states.shape[0]
            latent_dim = model.encoder.fusion_layer[-1].out_features
            z_prev = torch.zeros(batch_size, latent_dim, device=device)
            prev_action = torch.zeros(batch_size, actions.shape[-1], device=device)
            
            for t in range(1, states.shape[1]):
                current_state = states[:, t]
                current_action = actions[:, t-1]
                
                # Forward pass
                z_current, _, z_predicted = model(
                    current_state, z_prev, prev_action
                )
                
                # Compute metrics
                mse_loss = torch.nn.functional.mse_loss(z_current, z_predicted, reduction='none')
                l1_loss = torch.nn.functional.l1_loss(z_current, z_predicted, reduction='none')
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(z_current, z_predicted)
                
                prediction_metrics['mse_loss'].append(mse_loss.mean().item())
                prediction_metrics['cosine_similarity'].append(cos_sim.mean().item())
                prediction_metrics['l1_loss'].append(l1_loss.mean().item())
                
                # Update for next iteration
                z_prev = z_current
                prev_action = current_action
    
    # Aggregate metrics
    for key in prediction_metrics:
        prediction_metrics[key] = np.mean(prediction_metrics[key])
    
    return prediction_metrics

def main():
    # Load dataset
    data = np.load("safe_rl_dataset_images.npz")
    
    # Hyperparameters
    input_channels = 3
    latent_dim = 4
    action_dim = 2
    hidden_dim = 64
    batch_size = 32
    validation_split = 0.2
    
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create dataset
    full_dataset = DynamicsDataset(data, sequence_length=10, prediction_horizon=5)
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load trained model
    model = DeterministicDynamicsModel(
        input_channels=input_channels, 
        latent_dim=latent_dim, 
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    model.load_state_dict(torch.load('/Users/i.k.tabbara/Documents/python directory/CCBF_images_model_final_fixedaction.pth'))
    model.to(device)
    model.eval()
    
    # Visualize latent space and reconstructions
    visualize_latent_space_and_reconstructions(model, val_loader, device)
    
    # Analyze latent state predictions
    prediction_metrics = analyze_latent_state_prediction(model, val_loader, device)
    
    # Print and save metrics
    print("Latent State Prediction Metrics:")
    for metric, value in prediction_metrics.items():
        print(f"{metric}: {value}")
    
    # Save metrics to file
    with open('latent_state_prediction_metrics.txt', 'w') as f:
        for metric, value in prediction_metrics.items():
            f.write(f"{metric}: {value}\n")

if __name__ == "__main__":
    main()