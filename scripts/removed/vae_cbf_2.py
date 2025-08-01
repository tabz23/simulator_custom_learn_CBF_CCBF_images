import sys
import os
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
from modules.network import CBF
from vaedynamics_newstart_good_3_decoderaffine import DeterministicDynamicsModel,AffineDynamics,RecursiveEncoder,DeterministicDecoder,mlp

from tqdm import trange, tqdm

class CombinedDataset(Dataset):
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
            sequence_ground_truth_states = self.ground_truth_states[i:i+self.sequence_length]

            # Skip sequences with done flags in the first prediction horizon
            if torch.any(sequence_dones[:self.prediction_horizon] == 1):
                i += 1
                continue

            sequence = {
                'actions': sequence_actions,
                'states_rgb': sequence_states,
                'dones': sequence_dones,
                'ground_truth_states':sequence_ground_truth_states
            }

            sequences.append(sequence)
            i += 1

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def compute_multiple_shooting_loss(model, batch, device,
                                    prediction_horizon=5,
                                    wstate=100.0,
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
        current_action = actions[:, t - 1]

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
            'prediction_loss': wstate * prediction_loss.item(),
            'train_recon_loss1': wrec1 * recon_loss1.item(),
            'train_recon_loss2': wrec2 * recon_loss2.item()  # for x axis
        })

        # Update previous state and action for next iteration
        z_prev = z_current.detach()
        prev_action = current_action

    # Normalize loss
    total_loss /= (seq_length - 1)
    wandb.log({
        'train_loss_total': total_loss.item(),
    })

    return total_loss
def train_dynamics(model, train_loader, optimizer, device):
    """Training function for the dynamics model."""
    model.train()
    total_train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):  # Keep track of batch index
        optimizer.zero_grad()

        loss = compute_multiple_shooting_loss(model, batch, device)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    wandb.log({'dynamics_train_loss': avg_train_loss})
    return avg_train_loss

def compute_cbf_loss(cbf_model, dynamics_model, batch, device,
                     safe_distance=4, eps_safe=0.1, eps_unsafe=0.15,
                     safe_loss_weight=1, unsafe_loss_weight=1.3,
                     action_loss_weight=1, dt=0.1, eps_grad=0):
    """Compute the CBF loss."""
    ground_truth_states = batch['ground_truth_states'].to(device)
    actions = batch['actions'].to(device)
    states_rgb = batch['states_rgb'].to(device)
    batch_size, seq_length, channels, height, width = states_rgb.shape

    latent_dim = dynamics_model.encoder.fusion_layer[-1].out_features
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    # Initial zero latent state and zero action
    z_prev = torch.zeros(batch_size, latent_dim, device=device)
    prev_action = torch.zeros(batch_size, actions.shape[-1], device=device)

    # Initialize accumulators for metrics
    safe_correct = 0
    unsafe_correct = 0
    safe_h_sum = 0
    unsafe_h_sum = 0
    safe_count = 0
    unsafe_count = 0

    for t in range(1, seq_length):
        current_state_rgb = states_rgb[:, t]
        current_action = actions[:, t - 1]
        ground_truth_state = ground_truth_states[:,t]

        # Encode the current state into the latent space
        z_current, _, _ = dynamics_model(current_state_rgb, z_prev, prev_action)

        # Define safe and unsafe sets based on ground truth states
        is_safe = torch.norm(ground_truth_state[:, :2] - ground_truth_state[:, 2:], dim=1) > safe_distance
        safe_mask = is_safe.float().to(device).reshape(-1, 1)
        unsafe_mask = (~is_safe).float().to(device).reshape(-1, 1)

        # Random actions for CQL
        random_actions = torch.randn(z_current.shape[0], actions.shape[-1]).to(device)
        B = cbf_model(z_current).reshape(-1, 1)  # CBF value for current state

        # Loss for safe states
        loss_safe_vector = safe_loss_weight * F.relu(-B + eps_safe) * safe_mask
        num_safe_elements = safe_mask.sum()
        loss_safe = loss_safe_vector.sum() / (num_safe_elements + 1e-8)
        total_loss = total_loss + loss_safe

        # Loss for unsafe states
        loss_unsafe_vector = unsafe_loss_weight * F.relu(B + eps_unsafe) * unsafe_mask
        num_unsafe_elements = unsafe_mask.sum()
        loss_unsafe = loss_unsafe_vector.sum() / (num_unsafe_elements + 1e-8)
        total_loss = total_loss + loss_unsafe

        # Gradient loss calculation
        z_current = z_current.detach().clone().requires_grad_(True)####imp note: z_current is not a leaf tensor, meaning it has been derived from another tensor via some computation. In PyTorch, only leaf tensors (those created directly, not as the result of an operation) can have requires_grad set manually.
        B = cbf_model(z_current).reshape(-1, 1)
        grad_b = torch.autograd.grad(B, z_current, grad_outputs=torch.ones_like(B), retain_graph=True)[0]

        with torch.no_grad():
            f, g = dynamics_model.dynamics_predictor(z_current)
            x_dot = f + torch.einsum('bsa,ba->bs', g.view(g.shape[0], dynamics_model.dynamics_predictor.state_dim, dynamics_model.dynamics_predictor.num_action), current_action)
            b_dot = torch.einsum('bo,bo->b', grad_b, x_dot).reshape(-1, 1)
        gradient = b_dot + 1 * B
        loss_grad_vector = 1 * F.relu(eps_grad - gradient) * safe_mask
        num_grad_elements = safe_mask.sum()
        loss_grad = loss_grad_vector.sum() / (num_grad_elements + 1e-8)
        total_loss = total_loss + loss_grad

        # Compute metrics
        safe_correct += ((B > 0) == is_safe.reshape(-1,1)).sum().item()
        unsafe_correct += ((B <= 0) == (~is_safe).reshape(-1,1)).sum().item()
        safe_h_sum += (B * safe_mask).sum().item()
        unsafe_h_sum += (B * unsafe_mask).sum().item()
        safe_count += safe_mask.sum().item()
        unsafe_count += unsafe_mask.sum().item()


        wandb.log({
            'loss_safe': loss_safe.item(),
            'loss_unsafe': loss_unsafe.item(),
            'loss_grad': loss_grad.item(),
        })

        # Update previous state and action
        z_prev = z_current.detach()
        prev_action = current_action

    total_loss /= (seq_length - 1)

    # Calculate and log metrics
    total_samples = safe_count + unsafe_count
    safe_accuracy = safe_correct / total_samples if total_samples > 0 else 0
    unsafe_accuracy = unsafe_correct / total_samples if total_samples > 0 else 0
    avg_safe_h = safe_h_sum / safe_count if safe_count > 0 else 0
    avg_unsafe_h = unsafe_h_sum / unsafe_count if unsafe_count > 0 else 0

    wandb.log({
        'safe_accuracy': safe_accuracy,
        'unsafe_accuracy': unsafe_accuracy,
        'avg_safe_h': avg_safe_h,
        'avg_unsafe_h': avg_unsafe_h
    })
    return total_loss

def train_cbf(cbf_model, dynamics_model, train_loader, optimizer, device,
              safe_distance=4, eps_safe=0.02, eps_unsafe=0.02,
              safe_loss_weight=1, unsafe_loss_weight=1.5,
              action_loss_weight=1, dt=0.1, eps_grad=0.1):
    """Training function for the CBF model."""
    cbf_model.train()
    total_train_loss = 0.0

    # Initialize accumulators for metrics
    total_safe_accuracy = 0.0
    total_unsafe_accuracy = 0.0
    total_avg_safe_h = 0.0
    total_avg_unsafe_h = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        loss = compute_cbf_loss(cbf_model, dynamics_model, batch, device,
                                 safe_distance, eps_safe, eps_unsafe,
                                 safe_loss_weight, unsafe_loss_weight,
                                 action_loss_weight, dt, eps_grad)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Accumulate metrics (assuming compute_cbf_loss logs them)
        total_safe_accuracy += wandb.summary.get('safe_accuracy', 0)  # type: ignore
        total_unsafe_accuracy += wandb.summary.get('unsafe_accuracy', 0)  # type: ignore
        total_avg_safe_h += wandb.summary.get('avg_safe_h', 0)  # type: ignore
        total_avg_unsafe_h += wandb.summary.get('avg_unsafe_h', 0)  # type: ignore
        num_batches += 1

    avg_train_loss = total_train_loss / len(train_loader)
    avg_safe_accuracy = total_safe_accuracy / num_batches if num_batches > 0 else 0
    avg_unsafe_accuracy = total_unsafe_accuracy / num_batches if num_batches > 0 else 0
    avg_avg_safe_h = total_avg_safe_h / num_batches if num_batches > 0 else 0
    avg_avg_unsafe_h = total_avg_unsafe_h / num_batches if num_batches > 0 else 0


    wandb.log({
        'cbf_train_loss': avg_train_loss,
        'train_safe_accuracy': avg_safe_accuracy,
        'train_unsafe_accuracy': avg_unsafe_accuracy,
        'train_avg_safe_h': avg_avg_safe_h,
        'train_avg_unsafe_h': avg_avg_unsafe_h
    })

    return avg_train_loss

def main():
    # Load dataset
    data = np.load("safe_rl_dataset_images.npz")

    # Hyperparameters
    input_channels = 3
    latent_dim = 4
    action_dim = 2
    hidden_dim = 64
    batch_size = 32
    learning_rate_dynamics = 1e-4
    learning_rate_cbf = 1e-3
    epochs = 100
    validation_split = 0.2
    device = "mps"

    # CBF parameters
    safe_distance = 4
    eps_safe = 0.02
    eps_unsafe = 0.02
    safe_loss_weight = 1
    unsafe_loss_weight = 1.5
    action_loss_weight = 1
    dt = 0.1
    eps_grad = 0.1

    # Create dataset
    full_dataset = CombinedDataset(data, sequence_length=10, prediction_horizon=5)

    # Split into train and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    dynamics_model = DeterministicDynamicsModel(
        input_channels=input_channels,
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    ).to(device)

    num_hidden_dim=3
    dim_hidden=32
    cbf_model = CBF(
        # state_dim=latent_dim,##by default it is 4 in the class
        num_hidden_dim=num_hidden_dim,dim_hidden=dim_hidden###I TOOK 3 LAYERS 32 NEURONS EACH, MIGHT BE TOO SMALL DOUBLE CHECK THIS WHEN POSIBLE
    ).to(device)

    # Optimizers
    optimizer_dynamics = optim.Adam(dynamics_model.parameters(), lr=learning_rate_dynamics)
    optimizer_cbf = optim.Adam(cbf_model.parameters(), lr=learning_rate_cbf)

    # Initialize WandB
    wandb.init(project="combined-dynamics-cbf-training", name="run_1")

    best_val_loss_dynamics = float('inf')
    best_val_loss_cbf = float('inf')

    # Training loop
    for epoch in trange(epochs):

        ## Train dynamics model
        # avg_train_loss_dynamics = train_dynamics(dynamics_model, train_loader, optimizer_dynamics, device)

        dynamics_model.load_state_dict(torch.load('/Users/i.k.tabbara/Documents/python directory/CCBF_images_model_final.pth'))
        dynamics_model.to(device)
        dynamics_model.eval()

        # Train CBF model
        avg_train_loss_cbf = train_cbf(cbf_model, dynamics_model, train_loader, optimizer_cbf, device,
                                         safe_distance, eps_safe, eps_unsafe,
                                         safe_loss_weight, unsafe_loss_weight,
                                         action_loss_weight, dt, eps_grad)

        # Validation (Dynamics)
        dynamics_model.eval()
        total_val_loss_dynamics = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                val_loss_dynamics = compute_multiple_shooting_loss(dynamics_model, batch, device)
                total_val_loss_dynamics += val_loss_dynamics.item()
        avg_val_loss_dynamics = total_val_loss_dynamics / len(val_loader)
        wandb.log({'dynamics_val_loss': avg_val_loss_dynamics, 'epoch': epoch})

        # Validation (CBF)
        cbf_model.eval()
        total_val_loss_cbf = 0.0
        # Initialize accumulators for metrics
        total_safe_accuracy_val = 0.0
        total_unsafe_accuracy_val = 0.0
        total_avg_safe_h_val = 0.0
        total_avg_unsafe_h_val = 0.0
        num_batches_val = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                val_loss_cbf = compute_cbf_loss(cbf_model, dynamics_model, batch, device,
                                                  safe_distance, eps_safe, eps_unsafe,
                                                  safe_loss_weight, unsafe_loss_weight,
                                                  action_loss_weight, dt, eps_grad)
                total_val_loss_cbf += val_loss_cbf.item()
                # Accumulate metrics (assuming compute_cbf_loss logs them)
                total_safe_accuracy_val += wandb.summary.get('safe_accuracy', 0)  # type: ignore
                total_unsafe_accuracy_val += wandb.summary.get('unsafe_accuracy', 0)  # type: ignore
                total_avg_safe_h_val += wandb.summary.get('avg_safe_h', 0)  # type: ignore
                total_avg_unsafe_h_val += wandb.summary.get('avg_unsafe_h', 0)  # type: ignore
                num_batches_val += 1
        avg_val_loss_cbf = total_val_loss_cbf / len(val_loader)
        avg_safe_accuracy_val = total_safe_accuracy_val / num_batches_val if num_batches_val > 0 else 0
        avg_unsafe_accuracy_val = total_unsafe_accuracy_val / num_batches_val if num_batches_val > 0 else 0
        avg_avg_safe_h_val = total_avg_safe_h_val / num_batches_val if num_batches_val > 0 else 0
        avg_avg_unsafe_h_val = total_avg_unsafe_h_val / num_batches_val if num_batches_val > 0 else 0
        wandb.log({
            'cbf_val_loss': avg_val_loss_cbf,
            'val_safe_accuracy': avg_safe_accuracy_val,
            'val_unsafe_accuracy': avg_unsafe_accuracy_val,
            'val_avg_safe_h': avg_avg_safe_h_val,
            'val_avg_unsafe_h': avg_avg_unsafe_h_val,
            'epoch': epoch
        })

        # Save best models
        if avg_val_loss_dynamics < best_val_loss_dynamics:
            best_val_loss_dynamics = avg_val_loss_dynamics
            torch.save(dynamics_model.state_dict(), 'best_dynamics_model.pth')

        if avg_val_loss_cbf < best_val_loss_cbf:
            best_val_loss_cbf = avg_val_loss_cbf
            torch.save(cbf_model.state_dict(), 'best_cbf_model.pth')

        print(f"Epoch {epoch+1}: Dynamics Train Loss {avg_train_loss_dynamics:.4f}, Dynamics Val Loss {avg_val_loss_dynamics:.4f}")
        print(f"Epoch {epoch+1}: CBF Train Loss {avg_train_loss_cbf:.4f}, CBF Val Loss {avg_val_loss_cbf:.4f}")

    # Finish WandB
    wandb.finish()

if __name__ == "__main__":
    main()
