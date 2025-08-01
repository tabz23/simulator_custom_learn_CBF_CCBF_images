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
    Time step 67:
    States shape: torch.Size([32, 100, 3, 64, 64])MB deduped)
    Actions shape: torch.Size([32, 100, 2])
    
    Latent dim: 4
    z_prev shape: torch.Size([32, 4])
    prev_action shape: torch.Size([32, 2])
    current_state shape: torch.Size([32, 3, 64, 64])
    current_action shape: torch.Size([32, 2])
    z_current shape: torch.Size([32, 4])
    x_recon_current shape: torch.Size([32, 3, 64, 64])
    z_predicted shape: torch.Size([32, 4])
    model.decoder(z_predicted) shape: torch.Size([32, 3, 64, 64])
    """

    states = batch['states_rgb'].to(device)
    actions = batch['actions'].to(device)

    batch_size, seq_length, channels, height, width = states.shape
    latent_dim = model.encoder.fusion_layer[-1].out_features
    # print(f"States shape: {states.shape}")
    # print(f"Actions shape: {actions.shape}")
    # print(f"Latent dim: {latent_dim}")
    total_loss = torch.zeros(1, device=device)

    # Initial zero latent state and zero action
    z_prev = torch.zeros(batch_size, latent_dim, device=device)
    prev_action = torch.zeros(batch_size, actions.shape[-1], device=device)
    # print(f"z_prev shape: {z_prev.shape}")
    # print(f"prev_action shape: {prev_action.shape}")

    for t in range(1, seq_length):
        # Current states and actions
        current_state = states[:, t]
        current_action = actions[:, t ]###CHANGED THIS
        # print(f"\nTime step {t}:")
        # print(f"current_state shape: {current_state.shape}")
        # print(f"current_action shape: {current_action.shape}")
        # Forward pass
        z_current, x_recon_current, z_predicted = model(
            current_state, z_prev, prev_action
        )
        # print(f"z_current shape: {z_current.shape}")
        # print(f"x_recon_current shape: {x_recon_current.shape}")
        # print(f"z_predicted shape: {z_predicted.shape}")
        
        # Multiple shooting loss components
        prediction_loss = F.mse_loss(z_predicted, z_current)
        # print(f"model.decoder(z_predicted) shape: {model.decoder(z_predicted).shape}")
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

        ###FIX HERE UPDATES ON Z LATENT
        ###FIX HERE UPDATES ON Z LATENT
        ###FIX HERE UPDATES ON Z LATENT
        ###FIX HERE UPDATES ON Z LATENT
        wandb.log({
            'prediction_loss':wstate * prediction_loss.item(),
            'train_recon_loss1': wrec1*recon_loss1.item(),
            'train_recon_loss2': wrec2*recon_loss2.item() # for x axis
        })
        z_prev = z_current.detach()
        prev_action = current_action
        

    # Normalize loss
    total_loss /= (seq_length - 1)
    

    return total_loss

def compute_cbf_loss(cbf_model, dynamics_model, batch, device,
                     safe_distance=4, eps_safe=0.05, eps_unsafe=0.2,
                     safe_loss_weight=1, unsafe_loss_weight=2,
                     action_loss_weight=1, dt=0.1, eps_grad=0.00):
    """Compute the CBF loss."""
    ground_truth_states = batch['ground_truth_states'].to(device)
    actions = batch['actions'].to(device)
    states_rgb = batch['states_rgb'].to(device)
    batch_size, seq_length, channels, height, width = states_rgb.shape

    # print(f"Initial shapes:")
    # print(f"  ground_truth_states: {ground_truth_states.shape}")
    # print(f"  actions: {actions.shape}")
    # print(f"  states_rgb: {states_rgb.shape}")
    # print(f"  batch_size: {batch_size}, seq_length: {seq_length}")

    latent_dim = dynamics_model.encoder.fusion_layer[-1].out_features
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    # Initial zero latent state and zero action
    z_prev = torch.zeros(batch_size, latent_dim, device=device)
    prev_action = torch.zeros(batch_size, actions.shape[-1], device=device)

    # print(f"Initial latent state (z_prev): {z_prev.shape}")
    # print(f"Initial previous action (prev_action): {prev_action.shape}")

    # Initialize accumulators for metrics
    safe_correct = 0
    unsafe_correct = 0
    safe_h_sum = 0
    unsafe_h_sum = 0
    safe_count = 0
    unsafe_count = 0

    for t in range(1, seq_length):
        # print(f"\nTime step {t}:")
        current_state_rgb = states_rgb[:, t]
        current_action = actions[:, t]####CHANGED THIS
        ground_truth_state = ground_truth_states[:, t]

        # print(f"  current_state_rgb: {current_state_rgb.shape}")
        # print(f"  current_action: {current_action.shape}")
        # print(f"  ground_truth_state: {ground_truth_state.shape}")

        # Encode the current state into the latent space
        # with torch.no_grad():##THIS DOESNT ALLOWBACKPROP FROM CBF INTO THE DYNAMICS
        #     z_current, _, _ = dynamics_model(current_state_rgb, z_prev, prev_action)
        # with torch.no_grad():
        # with torch.no_grad():
        # with torch.no_grad():
        # with torch.no_grad():IMP IMPIMP IMPIMP IMP
        # with torch.no_grad():
        # with torch.no_grad():
        z_current, _, _ = dynamics_model(current_state_rgb, z_prev, prev_action)

        # print(f"  z_current (encoded latent state): {z_current.shape}")

        # Define safe and unsafe sets based on ground truth states
        is_safe = torch.norm(ground_truth_state[:, :2] - ground_truth_state[:, 2:], dim=1) > safe_distance
        safe_mask = is_safe.float().to(device).reshape(-1, 1).detach()##jsut incase
        unsafe_mask = (~is_safe).float().to(device).reshape(-1, 1).detach()##just incase

        # print(f"  is_safe: {is_safe}")
        # print(f"  safe_mask: {safe_mask}")
        # print(f"  unsafe_mask: {unsafe_mask}")

        # CBF value for current state
        # z_current = z_current.detach().clone().requires_grad_(True)###THIS WAS BEING DETACHED PREVIOUSLY
        z_current = z_current.clone().requires_grad_(True)
        B = cbf_model(z_current).reshape(-1, 1)

        # print(f"  B (CBF value): {B.shape}")

        # Loss for safe states
        loss_safe_vector = safe_loss_weight * F.relu(-B + eps_safe) * safe_mask
        num_safe_elements = safe_mask.sum()
        loss_safe = loss_safe_vector.sum() / (num_safe_elements + 1e-8)
        total_loss = total_loss + loss_safe

        # print(f"  loss_safe_vector: {loss_safe_vector.shape}")
        # print(f"  num_safe_elements: {num_safe_elements.item()}")
        # print(f"  loss_safe: {loss_safe.item()}")

        # Loss for unsafe states
        loss_unsafe_vector = unsafe_loss_weight * F.relu(B + eps_unsafe) * unsafe_mask
        num_unsafe_elements = unsafe_mask.sum()
        loss_unsafe = loss_unsafe_vector.sum() / (num_unsafe_elements + 1e-8)
        total_loss = total_loss + loss_unsafe

        # print(f"  loss_unsafe_vector: {loss_unsafe_vector.shape}")
        # print(f"  num_unsafe_elements: {num_unsafe_elements.item()}")
        # print(f"  loss_unsafe: {loss_unsafe.item()}")

        # Gradient loss calculation
        B = cbf_model(z_current).reshape(-1, 1)
        grad_b = torch.autograd.grad(B, z_current, grad_outputs=torch.ones_like(B), retain_graph=True)[0]

        # print(f"  grad_b (gradient of B w.r.t. z_current): {grad_b.shape}")

        with torch.no_grad():
            f, g = dynamics_model.dynamics_predictor(z_current)
        x_dot = f + torch.einsum('bsa,ba->bs', g.view(g.shape[0], dynamics_model.dynamics_predictor.state_dim, dynamics_model.dynamics_predictor.num_action), current_action)
        b_dot = torch.einsum('bo,bo->b', grad_b, x_dot).reshape(-1, 1)

        # print(f"  f (dynamics drift): {f.shape}")
        # print(f"  g (dynamics control matrix): {g.shape}")
        # print(f"  x_dot (state derivative): {x_dot.shape}")
        # print(f"  b_dot (derivative of B): {b_dot.shape}")

        gradient = b_dot + 1 * B
        loss_grad_vector = 1 * F.relu(eps_grad - gradient) * safe_mask
        num_grad_elements = safe_mask.sum()
        loss_grad = loss_grad_vector.sum() / (num_grad_elements + 1e-8)
        
        ###REMOVED LOSS_GRAD###REMOVED LOSS_GRAD###REMOVED LOSS_GRAD###REMOVED LOSS_GRAD###REMOVED LOSS_GRAD
        total_loss = total_loss #+ loss_grad #

        # print(f"  gradient: {gradient.shape}")
        # print(f"  loss_grad_vector: {loss_grad_vector.shape}")
        # print(f"  num_grad_elements: {num_grad_elements.item()}")
        # print(f"  loss_grad: {loss_grad.item()}")

        # Compute metrics
        safe_correct += ((B > 0) & is_safe.reshape(-1, 1)).sum().item()
        unsafe_correct += ((B <= 0) & (~is_safe).reshape(-1, 1)).sum().item()
        safe_h_sum += (B * safe_mask).sum().item()
        unsafe_h_sum += (B * unsafe_mask).sum().item()
        safe_count += safe_mask.sum().item()
        unsafe_count += unsafe_mask.sum().item()

        # print(f"  safe_correct: {safe_correct}")
        # print(f"  unsafe_correct: {unsafe_correct}")
        # print(f"  safe_h_sum: {safe_h_sum}")
        # print(f"  unsafe_h_sum: {unsafe_h_sum}")
        # print(f"  safe_count: {safe_count}")
        # print(f"  unsafe_count: {unsafe_count}")

        # Update previous state and action
        z_prev = z_current.detach()
        prev_action = current_action

    # Normalize loss
    total_loss /= (seq_length - 1)

    # print(f"\nFinal total loss: {total_loss.item()}")

    # Calculate metrics
    total_samples = safe_count + unsafe_count
    safe_accuracy = safe_correct / safe_count if safe_count > 0 else 0
    unsafe_accuracy = unsafe_correct / unsafe_count if unsafe_count > 0 else 0
    avg_safe_h = safe_h_sum / safe_count if safe_count > 0 else 0
    avg_unsafe_h = unsafe_h_sum / unsafe_count if unsafe_count > 0 else 0

    # print(f"Final metrics:")
    # print(f"  safe_accuracy: {safe_accuracy}")
    # print(f"  unsafe_accuracy: {unsafe_accuracy}")
    # print(f"  avg_safe_h: {avg_safe_h}")
    # print(f"  avg_unsafe_h: {avg_unsafe_h}")

    return total_loss, {
        'safe_accuracy': safe_accuracy,
        'unsafe_accuracy': unsafe_accuracy,
        'avg_safe_h': avg_safe_h,
        'avg_unsafe_h': avg_unsafe_h
    }
'''
Time step 84:
  current_state_rgb: torch.Size([32, 3, 64, 64])
  current_action: torch.Size([32, 2])
  ground_truth_state: torch.Size([32, 4])
  z_current (encoded latent state): torch.Size([32, 4])
  is_safe: torch.Size([32])
  safe_mask: torch.Size([32, 1])
  unsafe_mask: torch.Size([32, 1])
  B (CBF value): torch.Size([32, 1])
  loss_safe_vector: torch.Size([32, 1])
  num_safe_elements: 27.0
  loss_safe: 0.003053400432690978
  loss_unsafe_vector: torch.Size([32, 1])
  num_unsafe_elements: 5.0
  loss_unsafe: 0.4470077455043793
  grad_b (gradient of B w.r.t. z_current): torch.Size([32, 4])
  f (dynamics drift): torch.Size([32, 4])
  g (dynamics control matrix): torch.Size([32, 8])
  x_dot (state derivative): torch.Size([32, 4])
  b_dot (derivative of B): torch.Size([32, 1])
  gradient: torch.Size([32, 1])
  loss_grad_vector: torch.Size([32, 1])
  num_grad_elements: 27.0
  loss_grad: 0.0
  safe_correct: 2568
  unsafe_correct: 0
  safe_h_sum: 257.6569182872772
  unsafe_h_sum: 11.823146291077137
  safe_count: 2568.0
  unsafe_count: 120.0
  
  
Time step 98:
  current_state_rgb: torch.Size([32, 3, 64, 64])
  current_action: torch.Size([32, 2])
  ground_truth_state: torch.Size([32, 4])
  z_current (encoded latent state): torch.Size([32, 4])
  is_safe: torch.Size([32])
  safe_mask: torch.Size([32, 1])
  unsafe_mask: torch.Size([32, 1])
  B (CBF value): torch.Size([32, 1])
  loss_safe_vector: torch.Size([32, 1])
  num_safe_elements: 31.0
  loss_safe: 0.0032630865462124348
  loss_unsafe_vector: torch.Size([32, 1])
  num_unsafe_elements: 1.0
  loss_unsafe: 0.4445728063583374
  grad_b (gradient of B w.r.t. z_current): torch.Size([32, 4])
  f (dynamics drift): torch.Size([32, 4])
  g (dynamics control matrix): torch.Size([32, 8])
  x_dot (state derivative): torch.Size([32, 4])
  b_dot (derivative of B): torch.Size([32, 1])
  gradient: torch.Size([32, 1])
  loss_grad_vector: torch.Size([32, 1])
  num_grad_elements: 31.0
  loss_grad: 0.0
  safe_correct: 2981
  unsafe_correct: 0
  safe_h_sum: 297.8748092651367
  unsafe_h_sum: 15.256714433431625
  safe_count: 2981.0
  unsafe_count: 155.0
  
   is_safe: tensor([ True,  True,  True,  True,  True,  True,  True,  True, False,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True, False,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True], device='mps:0')
'''
def train_and_validate(dynamics_model, cbf_model, train_loader, val_loader, 
                       optimizer_joint, device, epochs, 
                       safe_distance=4, eps_safe=0.05, eps_unsafe=0.2):
    """
    Train and validate both dynamics and CBF models in parallel
    """
    best_val_loss_dynamics = float('inf')
    best_val_loss_cbf = float('inf')
    global_step = 0

    for epoch in trange(epochs):
        # Training phase
        dynamics_model.train()
        cbf_model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            # Dynamics model training
            optimizer_joint.zero_grad()
            loss_dynamics = compute_multiple_shooting_loss(dynamics_model, batch, device)
            # loss_dynamics.backward()
            # optimizer_dynamics.step()

            # CBF model training
            # optimizer_cbf.zero_grad()
            loss_cbf, cbf_metrics = compute_cbf_loss(cbf_model, dynamics_model, batch, device, 
                                                     safe_distance, eps_safe, eps_unsafe)
            # loss_cbf.backward()
            # optimizer_cbf.step()
            tot_loss=loss_dynamics+loss_cbf
            tot_loss.backward()
            optimizer_joint.step()
            # Logging metrics at every step
            wandb.log({
                'train_loss_dynamics': loss_dynamics.item(),
                'train_loss_cbf': loss_cbf.item(),
                'train_safe_accuracy': cbf_metrics['safe_accuracy'],
                'train_unsafe_accuracy': cbf_metrics['unsafe_accuracy'],
                'train_avg_safe_h': cbf_metrics['avg_safe_h'],
                'train_avg_unsafe_h': cbf_metrics['avg_unsafe_h'],
                'global_step': global_step,
                'epoch': epoch
                ###add here gradient loss
            })

            global_step += 1

        # Validation phase
        dynamics_model.eval()
        cbf_model.eval()
        
        total_val_loss_dynamics = 0.0
        total_val_loss_cbf = 0.0
        val_safe_accuracies = []
        val_unsafe_accuracies = []
        val_safe_h_values = []
        val_unsafe_h_values = []


        #commented vlaidation because of an error for now
        # with torch.no_grad():
        #     for batch_idx, batch in enumerate(val_loader):
        #         # Dynamics validation
        #         val_loss_dynamics = compute_multiple_shooting_loss(dynamics_model, batch, device)
        #         total_val_loss_dynamics += val_loss_dynamics.item()

        #         # CBF validation
        #         val_loss_cbf, cbf_val_metrics = compute_cbf_loss(cbf_model, dynamics_model, batch, device, 
        #                                                          safe_distance, eps_safe, eps_unsafe)
        #         total_val_loss_cbf += val_loss_cbf.item()

        #         # Collect CBF metrics
        #         val_safe_accuracies.append(cbf_val_metrics['safe_accuracy'])
        #         val_unsafe_accuracies.append(cbf_val_metrics['unsafe_accuracy'])
        #         val_safe_h_values.append(cbf_val_metrics['avg_safe_h'])
        #         val_unsafe_h_values.append(cbf_val_metrics['avg_unsafe_h'])

        # # Average validation losses
        # avg_val_loss_dynamics = total_val_loss_dynamics / len(val_loader)
        # avg_val_loss_cbf = total_val_loss_cbf / len(val_loader)

        # # Average CBF metrics
        # avg_val_safe_accuracy = np.mean(val_safe_accuracies)
        # avg_val_unsafe_accuracy = np.mean(val_unsafe_accuracies)
        # avg_val_safe_h = np.mean(val_safe_h_values)
        # avg_val_unsafe_h = np.mean(val_unsafe_h_values)

        # # Logging validation metrics
        # wandb.log({
        #     'val_loss_dynamics': avg_val_loss_dynamics,
        #     'val_loss_cbf': avg_val_loss_cbf,
        #     'val_safe_accuracy': avg_val_safe_accuracy,
        #     'val_unsafe_accuracy': avg_val_unsafe_accuracy,
        #     'val_avg_safe_h': avg_val_safe_h,
        #     'val_avg_unsafe_h': avg_val_unsafe_h,
        #     'epoch': epoch
        # })

        # # Save best models
        # if avg_val_loss_dynamics < best_val_loss_dynamics:
        #     best_val_loss_dynamics = avg_val_loss_dynamics
        #     torch.save(dynamics_model.state_dict(), 'best_dynamics_model.pth')

        # if avg_val_loss_cbf < best_val_loss_cbf:
        #     best_val_loss_cbf = avg_val_loss_cbf
        #     torch.save(cbf_model.state_dict(), 'best_cbf_model.pth')

        # print(f"Epoch {epoch+1}:")
        # print(f"  Dynamic   s Train Loss: {avg_val_loss_dynamics:.4f}, Val Loss: {avg_val_loss_dynamics:.4f}")
        # print(f"  CBF Train Loss: {avg_val_loss_cbf:.4f}")
        # print(f"  CBF Val Safe Accuracy: {avg_val_safe_accuracy:.4f}")
        # print(f"  CBF Val Unsafe Accuracy: {avg_val_unsafe_accuracy:.4f}")

    return dynamics_model, cbf_model

def main():
    seed_everything(2)##it was 1 before
    
    # Initialize wandb
    wandb.init(project="parallel-dynamics-cbf-training", name="combined_training_run")

    # Load dataset
    data = np.load("safe_rl_dataset_images_ALLUNSAFE.npz")

    # Hyperparameters
    input_channels = 3
    latent_dim = 4
    action_dim = 2
    hidden_dim = 64
    batch_size = 32
    learning_rate_dynamics = 1e-4
    learning_rate_cbf = 1e-5
    epochs = 15
    validation_split = 0.2
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # CBF and training parameters
    safe_distance = 4
    eps_safe = 0
    eps_unsafe = 0.2

    # Create dataset
    full_dataset = CombinedDataset(data, sequence_length=10, prediction_horizon=100)

    # Split into train and validation sets
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    dynamics_model = DeterministicDynamicsModel(
        input_channels=input_channels,
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    ).to(device)

    cbf_model = CBF(
        num_hidden_dim=3,  # 3 hidden layers
        dim_hidden=64      # 64 neurons per layer
    ).to(device)

    # Optimizers
    # optimizer_dynamics = optim.Adam(dynamics_model.parameters(), lr=learning_rate_dynamics)
    # optimizer_cbf = optim.Adam(cbf_model.parameters(), lr=learning_rate_cbf)
    joint_optimizer = optim.Adam(
        list(dynamics_model.parameters()) + list(cbf_model.parameters()),
        lr=1e-5
    )

    # Train and validate models
    trained_dynamics_model, trained_cbf_model = train_and_validate(
        dynamics_model, 
        cbf_model, 
        train_loader, 
        val_loader, 
        joint_optimizer,
        device, 
        epochs,
        safe_distance,
        eps_safe,
        eps_unsafe
    )

    # Save final models
    torch.save(trained_dynamics_model.state_dict(), 'final_dynamics_model.pth')
    torch.save(trained_cbf_model.state_dict(), 'final_cbf_model.pth')

    # Finish wandb
    wandb.finish()

if __name__ == "__main__":
    main()
    
    
    
    
    
    """B (CBF value): torch.Size([32, 1])
  loss_safe_vector: torch.Size([32, 1])
  num_safe_elements: 32.0
  loss_safe: 0.0017523202113807201
  loss_unsafe_vector: torch.Size([32, 1])
  num_unsafe_elements: 0.0
  loss_unsafe: 0.0
  grad_b (gradient of B w.r.t. z_current): torch.Size([32, 4])
  f (dynamics drift): torch.Size([32, 4])
  g (dynamics control matrix): torch.Size([32, 8])
  x_dot (state derivative): torch.Size([32, 4])
  b_dot (derivative of B): torch.Size([32, 1])
  gradient: torch.Size([32, 1])
  loss_grad_vector: torch.Size([32, 1])
  num_grad_elements: 32.0
  loss_grad: 0.0
  safe_correct: 576
  unsafe_correct: 0
  safe_h_sum: 56.72461771965027
  unsafe_h_sum: 0.0
  safe_count: 576.0
  unsafe_count: 0.0

Time step 19:
  current_state_rgb: torch.Size([32, 3, 64, 64])
  current_action: torch.Size([32, 2])
  ground_truth_state: torch.Size([32, 4])
  z_current (encoded latent state): torch.Size([32, 4])
  is_safe: tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True], device='mps:0')
  safe_mask: tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], device='mps:0')
  unsafe_mask: tensor([[0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.],
        [0.]], device='mps:0')
  B (CBF value): torch.Size([32, 1])
  loss_safe_vector: torch.Size([32, 1])
  num_safe_elements: 32.0
  loss_safe: 0.0016677735839039087
  loss_unsafe_vector: torch.Size([32, 1])
  num_unsafe_elements: 0.0
  loss_unsafe: 0.0
  grad_b (gradient of B w.r.t. z_current): torch.Size([32, 4])
  f (dynamics drift): torch.Size([32, 4])
  g (dynamics control matrix): torch.Size([32, 8])
  x_dot (state derivative): torch.Size([32, 4])
  b_dot (derivative of B): torch.Size([32, 1])
  gradient: torch.Size([32, 1])
  loss_grad_vector: torch.Size([32, 1])
  num_grad_elements: 32.0
  loss_grad: 0.0
  safe_correct: 608
  unsafe_correct: 0
  safe_h_sum: 59.872472047805786
  unsafe_h_sum: 0.0
  safe_count: 608.0
  unsafe_count: 0.0
    """