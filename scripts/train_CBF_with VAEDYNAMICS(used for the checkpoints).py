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
import argparse
# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Import modules
from modules.network import CBF
from vaedynamics_newstart_good_3_decoderaffine_nosequential_4_deterministic import AffineDynamics, VAE
import random
rng = random.Random()
randomnb=rng.randint(100, 999)
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


class CombinedVAEDynamics(nn.Module):
    """A wrapper class that combines VAE encoder with dynamics model."""
    def __init__(self, vae, dynamics_model):
        super(CombinedVAEDynamics, self).__init__()
        self.vae = vae
        self.dynamics_model = dynamics_model
        
    def encode(self, x):
        """Encode images to latent space."""
        z = self.vae(x)
        return z
        
    def forward_dynamics(self, z, action):
        """Predict next state in latent space."""
        return self.dynamics_model(z, action)


def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class CombinedDataset(Dataset):
    def __init__(self, data, sequence_length=10, prediction_horizon=10):
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
        self.ground_truth_next_states = torch.FloatTensor(data['next_states'])

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
            

    
            sequence_ground_truth_next_states = self.ground_truth_next_states[i:i+self.sequence_length]##added this
            
            # print(sequence_ground_truth_states.shape)
            # print(sequence_ground_truth_next_states.shape)
            # Skip sequences with done flags in the first prediction horizon
            if torch.any(sequence_dones[:self.prediction_horizon] == 1):
                i += 1
                continue

            sequence = {
                'actions': sequence_actions,
                'states_rgb': sequence_states,
                'dones': sequence_dones,
                'ground_truth_states': sequence_ground_truth_states,
                'ground_truth_next_states':sequence_ground_truth_next_states###CHECK SHAPE OF THIS
            }

            sequences.append(sequence)
            i += 1

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def sample_random_actions(batch_size, action_dim=2, device="cpu"):
    """
    Sample random actions similar to the method in the original Trainer class
    """
    actions = []
    for i in range(action_dim):
        dim = 6 * torch.rand(batch_size, 1, device=device) - 3  # Uniform in [-3,3]
        actions.append(dim)
    return torch.cat(actions, dim=1)


def compute_cbf_loss(cbf_model, combined_model, batch, device,
                     safe_distance=4, eps_safe=0.05, eps_unsafe=0.2,
                     safe_loss_weight=1.0, unsafe_loss_weight=1.1,
                     gradient_loss_weight=1.0, dt=0.1, eps_grad=0.04,
                     use_cql_actions=True, cql_actions_weight=0.1,
                     num_action_samples=10, temp=0.7, detach=False):
    """Compute the CBF loss with added CQL-inspired loss."""
    ground_truth_states = batch['ground_truth_states'].to(device)
    ground_truth_next_states = batch['ground_truth_next_states'].to(device)
    actions = batch['actions'].to(device)
    states_rgb = batch['states_rgb'].to(device)
    # print("ground_truth_states",ground_truth_states.shape)
    # print("ground_truth_next_states",ground_truth_next_states.shape)
    
    batch_size, seq_length, channels, height, width = states_rgb.shape

    # Make sure all models are in the right mode
    cbf_model.train()
    combined_model.vae.eval()
    combined_model.dynamics_model.eval()
    
    # Get dynamics model for gradient calculations
    dynamics = combined_model.dynamics_model.dynamics

    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    # Initialize metrics
    safe_correct = 0
    unsafe_correct = 0
    safe_h_sum = 0
    unsafe_h_sum = 0
    safe_count = 0
    unsafe_count = 0
    gradient_loss_sum = 0
    safe_loss_sum = 0
    unsafe_loss_sum = 0
    cql_actions_loss_sum = 0
    logsumexp_term=0

    for t in range(seq_length):
        # Process current timestep
        current_state_rgb = states_rgb[:, t]
        current_action = actions[:, t] #if t < seq_length - 1 else torch.zeros_like(actions[:, 0])
        ground_truth_state = ground_truth_states[:, t]
        ground_truth_next_state = ground_truth_next_states[:, t]####ADDED THIS
        # print("ground_truth_state",ground_truth_state.shape)
        # print("ground_truth_next_states", ground_truth_next_states)
        
        # Encode current state to latent representation (without gradients for VAE)
        with torch.no_grad():
            _, z_current_detached = combined_model.vae(current_state_rgb)
        
        # Create a copy that requires gradients for CBF calculations
        z_current = z_current_detached.clone().detach().requires_grad_(True)

        # Define safe and unsafe sets based on ground truth states
        # Note: This uses the ground truth states, not latent states
        
        # print("ground_truth_state",ground_truth_state.shape)
        is_safe = torch.norm(ground_truth_state[:, :2] - ground_truth_state[:, 2:], dim=1) > safe_distance##ADD 3 
        
        # print("ground_truth_next_states",ground_truth_next_states.shape)
        is_safe_action_grad = torch.norm(ground_truth_next_state[:, :2] - ground_truth_next_state[:, 2:], dim=1) > safe_distance####ADDED THIS
        
        safe_mask = is_safe.float().to(device).reshape(-1, 1).detach()
        unsafe_mask = (~is_safe).float().to(device).reshape(-1, 1).detach()
        

        # Evaluate CBF on current latent state
        B = cbf_model(z_current).reshape(-1, 1)

        # Calculate safe loss (h(x) > 0 for safe states)
        loss_safe_vector = safe_loss_weight * F.relu(-B + eps_safe) * safe_mask
        num_safe_elements = safe_mask.sum() + 1e-8
        loss_safe = loss_safe_vector.sum() / num_safe_elements
        safe_loss_sum += loss_safe.item()
        total_loss = total_loss + loss_safe

        # Calculate unsafe loss (h(x) < 0 for unsafe states)
        loss_unsafe_vector = unsafe_loss_weight * F.relu(B + eps_unsafe) * unsafe_mask
        num_unsafe_elements = unsafe_mask.sum() + 1e-8
        loss_unsafe = loss_unsafe_vector.sum() / num_unsafe_elements
        unsafe_loss_sum += loss_unsafe.item()
        total_loss = total_loss + loss_unsafe

        # Calculate gradient loss (Lie derivative constraint) for safe states only
        if t < seq_length - 1 and safe_mask.sum() > 0:
            # Get CBF gradient with respect to latent state
            # grad_B = torch.autograd.grad(B.sum(), z_current, create_graph=True)[0]
            grad_B =  torch.autograd.grad(B, z_current,grad_outputs=torch.ones_like(B),retain_graph=True)[0] ###RETAIN GRAPH KEEP TRUE OR WONT PROPERLY USE GRADIENT
            # print(grad_B)
            # print(grad_B_2)
            
            # Use affine dynamics to calculate x_dot
            f, g = dynamics(z_current)
            gu = torch.einsum('bsa,ba->bs', 
                              g.view(g.shape[0], dynamics.state_dim, dynamics.num_action), 
                              current_action)
            x_dot = f + gu
            
            # Calculate Lie derivative: L_f h(x) = ∇h(x)ᵀf(x)
            b_dot = torch.sum(grad_B * x_dot, dim=1, keepdim=True)
            # print(grad_B.shape) 32,4 
            # print(x_dot.shape) 32,4 
            
            # CBF condition: ḣ(x) + αh(x) ≥ 0 for safe states
            # Equivalent to: L_f h(x) + αh(x) ≥ 0
            alpha = 1.0  # CBF parameter
            cbf_condition = b_dot + alpha * B
            
            # print(b_dot.shape)        torch.Size([32, 1])
            # print((alpha * B).shape)  torch.Size([32, 1])
            
            # Loss when condition is violated
            # is_safe_action_grad = torch.norm(ground_truth_next_states[:, :2] - ground_truth_next_states[:, 2:], dim=1) > safe_distance####ADDED THIS
            safe_mask_grad = is_safe_action_grad.float().to(device).reshape(-1, 1).detach()
            unsafe_mask_grad = (~is_safe_action_grad).float().to(device).reshape(-1, 1).detach()
            ##eps_safe like 0.1 but eps_unsafe 0.05
            loss_grad_vector_safe_action = gradient_loss_weight * F.relu(eps_grad - cbf_condition) * safe_mask_grad#like 0.
            loss_grad_vector_unsafe_action = (gradient_loss_weight/2) * F.relu(eps_grad + cbf_condition) * unsafe_mask_grad #for unsafe actions
            
            num_grad_safe_elements = safe_mask_grad.sum() + 1e-8
            num_grad_unsafe_elements = unsafe_mask_grad.sum() + 1e-8
            loss_grad_safe = loss_grad_vector_safe_action.sum() / num_grad_safe_elements
            loss_grad_unsafe = loss_grad_vector_unsafe_action.sum() / num_grad_unsafe_elements
            gradient_loss_sum += loss_grad_safe + loss_grad_unsafe
            total_loss = total_loss + gradient_loss_sum

            # CQL-inspired action loss for safe states
            if use_cql_actions and safe_mask.sum() > 0:
                # Get states that are safe for CQL loss
                safe_indices = torch.where(is_safe)[0]
                safe_z_current = z_current[safe_indices]
                safe_actions = current_action[safe_indices]
                
                # Get actual next states for safe states
                with torch.no_grad():
                    actual_next_z = combined_model.dynamics_model(safe_z_current, safe_actions)
                
                # Evaluate CBF on actual next states
                actual_next_B = cbf_model(actual_next_z)
                
                # Sample random actions and compute their next states
                all_random_next_B = []
                for _ in range(num_action_samples):
                    # Sample random actions for each safe state
                    random_actions = sample_random_actions(safe_z_current.shape[0], action_dim=safe_actions.shape[1], device=device)
                    
                    # Get next states using the learned dynamics model
                    random_next_z = combined_model.dynamics_model(safe_z_current, random_actions)
                    
                    # Evaluate CBF on these next states
                    random_next_B = cbf_model(random_next_z)
                    all_random_next_B.append(random_next_B)
                
                if all_random_next_B:
                    # Stack all CBF values for random actions
                    stacked_B_values = torch.stack(all_random_next_B, dim=1)
                    
                    # Combine with the actual next state CBF value
                    # print(stacked_B_values.shape)
                    # print(actual_next_B.shape)
                    # print(actual_next_B.squeeze().unsqueeze(1).shape)
                    # print(actual_next_B.unsqueeze(1).shape)
# torch.Size([30, 10, 1])|                                                  | 0/10227 [00:00<?, ?it/s]
# torch.Size([30, 1])
# torch.Size([30, 1])
# torch.Size([30, 1, 1])


                    
                    combined_B_values = torch.cat([stacked_B_values, actual_next_B.unsqueeze(1)], dim=1)
                    ###DOUBLE CHECK THE ABOVEx
                    # Compute logsumexp term (expectation over random actions)
                    logsumexp_B = temp * torch.logsumexp(combined_B_values/temp, dim=1, keepdim=True)
                    
                    # CQL term: make the action taken in the dataset have higher CBF value
                    # than the average of random actions
                    if detach:
                        cql_actions_term = logsumexp_B - actual_next_B.detach()
                    else:
                        cql_actions_term = logsumexp_B - actual_next_B
                    
                    # Add to total loss with appropriate weight
                    loss_cql_actions = cql_actions_weight * torch.mean(cql_actions_term)
                    cql_actions_loss_sum += loss_cql_actions.item()
                    total_loss = total_loss + loss_cql_actions
                    logsumexp_term+=torch.mean(logsumexp_B).item()
                    

        # Compute metrics
        safe_correct += ((B > 0) & is_safe.reshape(-1, 1)).sum().item()
        unsafe_correct += ((B <= 0) & (~is_safe).reshape(-1, 1)).sum().item()
        safe_h_sum += (B * safe_mask).sum().item()
        unsafe_h_sum += (B * unsafe_mask).sum().item()
        safe_count += safe_mask.sum().item()
        unsafe_count += unsafe_mask.sum().item()

    # Normalize total loss by sequence length
    total_loss = total_loss / seq_length
    
    # Calculate average metrics
    metrics = {
        'safe_accuracy': safe_correct / (safe_count + 1e-8),
        'unsafe_accuracy': unsafe_correct / (unsafe_count + 1e-8),
        'avg_safe_h': safe_h_sum / (safe_count + 1e-8),
        'avg_unsafe_h': unsafe_h_sum / (unsafe_count + 1e-8),
        'safe_loss': safe_loss_sum / seq_length,
        'unsafe_loss': unsafe_loss_sum / seq_length,
        'gradient_loss': gradient_loss_sum / (seq_length - 1) if seq_length > 1 else 0,
        'cql_actions_loss': cql_actions_loss_sum / seq_length,
        'logsumexp':logsumexp_term / seq_length
    }
    
    return total_loss, metrics


def train_cbf(combined_model, cbf_model, train_loader, val_loader, optimizer, device, 
              epochs, safe_distance=4, eps_safe=0.05, eps_unsafe=0.2, 
              gradient_loss_weight=1.0, checkpoint_interval=5,
              use_cql_actions=True, cql_actions_weight=0.1,
              num_action_samples=10, temp=0.7, detach=False):
    """
    Train CBF model using pre-trained VAE and dynamics models
    """
    # Make sure VAE and dynamics models are in eval mode
    combined_model.vae.eval()
    combined_model.dynamics_model.eval()
    
    # Freeze VAE and dynamics model parameters
    for param in combined_model.parameters():
        param.requires_grad = False
    
    best_val_loss = float('inf')
    global_step = 0

    for epoch in trange(epochs, desc="Training CBF"):
        # Training phase
        cbf_model.train()
        epoch_train_loss = 0.0
        epoch_train_metrics = {
            'safe_accuracy': 0.0,
            'unsafe_accuracy': 0.0,
            'avg_safe_h': 0.0,
            'avg_unsafe_h': 0.0,
            'safe_loss': 0.0,
            'unsafe_loss': 0.0,
            'gradient_loss': 0.0,
            'cql_actions_loss': 0.0
        }
        
        # Training loop
        train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            optimizer.zero_grad()
            
            loss, metrics = compute_cbf_loss(
                cbf_model, combined_model, batch, device,
                safe_distance=safe_distance, eps_safe=eps_safe, eps_unsafe=eps_unsafe,
                gradient_loss_weight=gradient_loss_weight,
                use_cql_actions=use_cql_actions, cql_actions_weight=cql_actions_weight,
                num_action_samples=num_action_samples, temp=temp, detach=detach
            )
            
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            epoch_train_loss += loss.item()
            for key in epoch_train_metrics:
                epoch_train_metrics[key] += metrics[key]
            
            # Log metrics for each batch
            wandb.log({
                'train_loss': loss.item(),
                'train_safe_accuracy': metrics['safe_accuracy'],
                'train_unsafe_accuracy': metrics['unsafe_accuracy'],
                'train_avg_safe_h': metrics['avg_safe_h'],
                'train_avg_unsafe_h': metrics['avg_unsafe_h'],
                'train_safe_loss': metrics['safe_loss'],
                'train_unsafe_loss': metrics['unsafe_loss'],
                'train_gradient_loss': metrics['gradient_loss'],
                'train_cql_actions_loss': metrics['cql_actions_loss'],
                'global_step': global_step,
                'logsumexp':metrics['logsumexp'],
                'epoch': epoch
            })
            
            global_step += 1
            train_batches += 1
        
        # Calculate epoch averages
        epoch_train_loss /= train_batches
        for key in epoch_train_metrics:
            epoch_train_metrics[key] /= train_batches

        # Validation phase
        cbf_model.eval()
        epoch_val_loss = 0.0
        epoch_val_metrics = {
            'safe_accuracy': 0.0,
            'unsafe_accuracy': 0.0,
            'avg_safe_h': 0.0,
            'avg_unsafe_h': 0.0,
            'safe_loss': 0.0,
            'unsafe_loss': 0.0,
            'gradient_loss': 0.0,
            'cql_actions_loss': 0.0
        }
        
        val_batches = 0
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
            # For validation, we need to recreate the computation graph for gradient calculation
            cbf_model.requires_grad_(True)
            
            val_loss, val_metrics = compute_cbf_loss(
                cbf_model, combined_model, batch, device,
                safe_distance=safe_distance, eps_safe=eps_safe, eps_unsafe=eps_unsafe,
                gradient_loss_weight=gradient_loss_weight,
                use_cql_actions=use_cql_actions, cql_actions_weight=cql_actions_weight,
                num_action_samples=num_action_samples, temp=temp, detach=detach
            )
            
            # Accumulate metrics
            epoch_val_loss += val_loss.item()
            for key in epoch_val_metrics:
                epoch_val_metrics[key] += val_metrics[key]
            
            val_batches += 1
        
        # Calculate epoch validation averages
        epoch_val_loss /= val_batches
        for key in epoch_val_metrics:
            epoch_val_metrics[key] /= val_batches
            
        # Log validation metrics
        wandb.log({
            'val_loss': epoch_val_loss,
            'val_safe_accuracy': epoch_val_metrics['safe_accuracy'],
            'val_unsafe_accuracy': epoch_val_metrics['unsafe_accuracy'], 
            'val_avg_safe_h': epoch_val_metrics['avg_safe_h'],
            'val_avg_unsafe_h': epoch_val_metrics['avg_unsafe_h'],
            'val_safe_loss': epoch_val_metrics['safe_loss'],
            'val_unsafe_loss': epoch_val_metrics['unsafe_loss'],
            'val_gradient_loss': epoch_val_metrics['gradient_loss'],
            'val_cql_actions_loss': epoch_val_metrics['cql_actions_loss'],
            'epoch': epoch
        })
        
        # Check if this is the best model so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(cbf_model.state_dict(), f'best_cbf_VAE_Deterministic_4_cql_{randomnb}.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint at specified intervals
        # if (epoch + 1) % checkpoint_interval == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'cbf_model_state_dict': cbf_model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': epoch_val_loss,
        #     }, f'cbf_checkpoint_epoch_{epoch+1}.pth')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}")
        print(f"  Train Safe Accuracy: {epoch_train_metrics['safe_accuracy']:.4f}")
        print(f"  Val Safe Accuracy: {epoch_val_metrics['safe_accuracy']:.4f}")
        print(f"  Train Unsafe Accuracy: {epoch_train_metrics['unsafe_accuracy']:.4f}")
        print(f"  Val Unsafe Accuracy: {epoch_val_metrics['unsafe_accuracy']:.4f}")
        print(f"  Train CQL Actions Loss: {epoch_train_metrics['cql_actions_loss']:.4f}")
        print(f"  Val CQL Actions Loss: {epoch_val_metrics['cql_actions_loss']:.4f}")
        # print(f"  val_gradient_loss': {epoch_val_metrics['gradient_loss']:.4f}")
    
    # Save final model
    torch.save(cbf_model.state_dict(), f"best_cbf_VAE_Deterministic_4_cql_final{randomnb}.pth")
    print("Training completed. Final model saved.")
    
    return cbf_model




import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
def parse_args():
    parser = argparse.ArgumentParser(description="CBF Training Config")

    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="safe_rl_dataset_images_ALLUNSAFE_big_obstacle_3000traj.npz", help="Path to dataset")
    parser.add_argument("--sequence_length", type=int, default=6, help="Sequence length")
    parser.add_argument("--prediction_horizon", type=int, default=6, help="Prediction horizon")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split")

    # Model parameters
    parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--latent_dim", type=int, default=4, help="Latent dimension size")
    parser.add_argument("--action_dim", type=int, default=2, help="Action dimension size")
    parser.add_argument("--hidden_dim", type=int, default=400, help="Hidden layer dimension")
    parser.add_argument("--cbf_hidden_dim", type=int, default=128, help="CBF hidden layer dimension")
    parser.add_argument("--cbf_layers", type=int, default=3, help="Number of CBF layers")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")

    # CBF parameters
    parser.add_argument("--safe_distance", type=float, default=6.0, help="Safe distance threshold")
    parser.add_argument("--eps_safe", type=float, default=0.05, help="Epsilon safe threshold")
    parser.add_argument("--eps_unsafe", type=float, default=0.1, help="Epsilon unsafe threshold")
    parser.add_argument("--gradient_loss_weight", type=float, default=3.0, help="Gradient loss weight")

    # Checkpointing
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Checkpoint saving interval")

    # CQL parameters
    parser.add_argument("--use_cql_actions", type=str2bool, default=False, help="Use CQL actions (True/False)")
    parser.add_argument("--cql_actions_weight", type=float, default=1, help="CQL actions weight")
    parser.add_argument("--num_action_samples", type=int, default=10, help="Number of action samples")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature parameter")
    parser.add_argument("--detach", type=str2bool, default=False, help="Detach flag (True/False)")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    

    args = parser.parse_args()
    return vars(args)  # Convert argparse Namespace to dictionary

def main():
    config = parse_args()
    # Log configuration to wandb
    # Set random seed for reproducibility
    seed_everything(config['seed'])

    # Initialize wandb
    wandb.init(project="cbf-training-latent-dynamics", name=f"cbf-training-latent-space_deterministic_with_cql_{randomnb}")
    
    # Parameters and configuration


    # Override config with provided command-line arguments

    wandb.config.update(config)
    
    # Device configuration
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else 
        "cpu"
    )
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {config['dataset_path']}...")
    try:
        data = np.load(config['dataset_path'])
    except FileNotFoundError:
        print(f"Dataset not found at {config['dataset_path']}. Please check the path.")
        return
    
    # Create dataset and data loaders
    print("Preparing dataset...")
    full_dataset = CombinedDataset(
        data, 
        sequence_length=config['sequence_length'], 
        prediction_horizon=config['prediction_horizon']
    )
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * config['validation_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    print(f"Dataset prepared. Train size: {train_size}, Validation size: {val_size}")
    
    # Load pre-trained VAE and dynamics models
    print("Loading pre-trained VAE and dynamics models...")
    try:
        # Initialize models
        vae = VAE(
            input_channels=config['input_channels'], 
            latent_dim=config['latent_dim'], 
            hidden_dim=config['hidden_dim']
        ).to(device)
        
        dynamics_model = DynamicsModel(
            latent_dim=config['latent_dim'], 
            action_dim=config['action_dim'], 
            hidden_dim=config['hidden_dim']
        ).to(device)
        
        # Load pre-trained weights
        vae.load_state_dict(torch.load('vae_model_final_4.pth', map_location=device))
        dynamics_model.load_state_dict(torch.load('dynamics_model_final_4.pth', map_location=device))
        
        # Set models to eval mode
        vae.eval()
        dynamics_model.eval()
        
        # Combine models
        combined_model = CombinedVAEDynamics(vae, dynamics_model)
        
    except FileNotFoundError as e:
        print(f"Error loading pre-trained models: {e}")
        print("Please make sure the model files exist and are in the correct location.")
        return
    
    # Initialize CBF model
    print("Initializing CBF model...")
    cbf_model = CBF(
        state_car_dim=2,  # Not used directly, but keeping for compatibility
        state_obstacles_dim=2,  # Not used directly, but keeping for compatibility
        dt=0.1,
        num_hidden_dim=config['cbf_layers'],
        dim_hidden=config['cbf_hidden_dim']
    ).to(device)
    
    # Initialize optimizer for CBF model
    optimizer = optim.Adam(cbf_model.parameters(), lr=config['learning_rate'])
    
    print(config['detach'])
    print(config['use_cql_actions'],)
    # Train the CBF model
    print("Starting CBF training...")
    trained_cbf_model = train_cbf(
        combined_model=combined_model,
        cbf_model=cbf_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=config['epochs'],
        safe_distance=config['safe_distance'],
        eps_safe=config['eps_safe'],
        eps_unsafe=config['eps_unsafe'],
        gradient_loss_weight=config['gradient_loss_weight'],
        checkpoint_interval=config['checkpoint_interval'],
        use_cql_actions=config['use_cql_actions'],
        cql_actions_weight=config['cql_actions_weight'],
        num_action_samples=config['num_action_samples'],
        temp=config['temp'],
        detach=config['detach']
    )
    
    # Finish wandb session
    wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()