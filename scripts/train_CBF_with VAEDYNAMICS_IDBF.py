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
from modules.network import CBF, MLPGaussianActor
from vaedynamics_newstart_good_3_decoderaffine_nosequential_4_deterministic import AffineDynamics, VAE
import random
rng = random.Random()
randomnb=rng.randint(100, 999)
device = torch.device(
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else 
        "cpu"
    )

BC_SAFE_MODEL_PATH = "bc_safe_model_images.pt"
    # Define model parameters
state_dim = 4  # car (2) + obstacle (2)
action_dim = 2
hidden_sizes = (300,300)
action_low = np.array([-3.0, -3.0])  # Based on environment limits
action_high = np.array([3.0, 3.0])

bc_safe_model = MLPGaussianActor(
obs_dim=state_dim,
act_dim=action_dim,
action_low=action_low,
action_high=action_high,
hidden_sizes=hidden_sizes,
activation=nn.ReLU,
device=device
).to(device)

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


# class CombinedDataset(Dataset):
#     def __init__(self, data):
#         """
#         Initialize dataset with sequence creation and prediction horizon
#         Args:
#             data (dict): Dictionary containing actions, states, and done flags
#             sequence_length (int): Total sequence length
#             prediction_horizon (int): Number of steps to predict ahead
#         """
#         self.actions = torch.FloatTensor(data['actions'])
#         self.states_rgb = torch.FloatTensor(data['states_rgb']).permute(0, 3, 1, 2) / 255.0
#         self.dones = torch.FloatTensor(data['dones'])
#         self.ground_truth_states = torch.FloatTensor(data['states'])
#         self.ground_truth_next_states = torch.FloatTensor(data['next_states'])
#         self.valid_indices = torch.where(self.dones == 0)[0]

#     def __len__(self):
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         actual_idx = self.valid_indices[idx].item()
        
#         return {
#             'actions': self.actions[actual_idx],
#             'states_rgb': self.states_rgb[actual_idx],
#             'ground_truth_states': self.ground_truth_states[actual_idx],
#             'ground_truth_next_states': self.ground_truth_next_states[actual_idx]
#         }

def compute_cbf_loss(cbf_model, combined_model, batch, device,
                     safe_distance=4, eps_safe=0.05, eps_unsafe=0.2,
                     safe_loss_weight=1.0, unsafe_loss_weight=2,
                     gradient_loss_weight=1.0, dt=0.1, eps_grad=0.04,
                     bc_safe_model=bc_safe_model,p=0.00000001):
    """Compute the CBF loss with added CQL-inspired loss."""
    ground_truth_states = batch['ground_truth_states'].to(device)
    ground_truth_next_states = batch['ground_truth_next_states'].to(device)
    actions = batch['actions'].to(device)
    states_rgb = batch['states_rgb'].to(device)
    
    batch_size = ground_truth_states.shape[0]

    # Make sure all models are in the right mode
    cbf_model.train()
    combined_model.vae.eval()
    combined_model.dynamics_model.eval()
    
    # Get dynamics model for gradient calculations
    dynamics = combined_model.dynamics_model.dynamics
    bc_safe_model=bc_safe_model
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    # Initialize metrics
    safe_correct = 0
    unsafe_correct = 0
    safe_h_sum = 0
    unsafe_h_sum = 0
    safe_count = batch_size ##all data is safe. this is because we also defne
    unsafe_count = 0
    gradient_loss_sum = 0
    safe_loss_sum = 0
    unsafe_loss_sum = 0


    # Process the batch
    # Encode current state to latent representation (without gradients for VAE)
    with torch.no_grad():
        _, z_current_detached = combined_model.vae(states_rgb)
    
    # Create a copy that requires gradients for CBF calculations
    z_current = z_current_detached.clone().detach().requires_grad_(True)




    ####modified code structure below######
    ##CRUCIAL NOTE I INCREASED THIS TO MORE THAN ALLOWABLE BOUND BECAUSE I WANT TO SIMULATE FARTHER STATE
    unsafe_states=[]
    for i in range(5):
        for j in range(batch_size):
            states_tensor = z_current[j].unsqueeze(0).to(device)
            random_actions_tensor = ( 6* torch.rand(1, 2) - 3 ).to(device)##actions between -3 3

            pi, _, log_probs = bc_safe_model(states_tensor, random_actions_tensor)##i think will be size 1,1 since appending 1 datapt at a time
           
            if torch.exp(log_probs).item()<p:##won't matter much but sample from the policy itself not from uniform
                next_state = combined_model.dynamics_model(states_tensor, random_actions_tensor)
                unsafe_states.append(next_state)
                unsafe_count+=1
                
      ##note: torch.cat Joins tensors along an existing dimension./torch.stack Creates a new dimension to stack tensors
    if (unsafe_states):
        unsafe_states_tensor=torch.cat(unsafe_states, dim=0).detach().requires_grad_(True)
    
    
    # Define safe and unsafe sets based on ground truth states
    # is_safe = torch.norm(ground_truth_states[:, :2] - ground_truth_states[:, 2:], dim=1) > safe_distance
    # is_safe_action_grad = torch.norm(ground_truth_next_states[:, :2] - ground_truth_next_states[:, 2:], dim=1) > safe_distance
    
    # safe_mask = is_safe.float().to(device).reshape(-1, 1).detach()
    # unsafe_mask = (~is_safe).float().to(device).reshape(-1, 1).detach()
    
    # Evaluate CBF on current latent state
    B_safe = cbf_model(z_current).reshape(-1, 1)

    # Calculate safe loss (h(x) > 0 for safe states)
    # Calculate safe loss (h(x) > 0 for safe states)
    loss_safe_vector = safe_loss_weight * F.relu(-B_safe + eps_safe) 
    loss_safe = loss_safe_vector.sum() / safe_count
    safe_loss_sum += loss_safe.item()
    total_loss = total_loss + loss_safe

    if (unsafe_states):
        B_unsafe = cbf_model(unsafe_states_tensor).reshape(-1, 1) ## Added
    
    # Calculate unsafe loss (h(x) < 0 for unsafe states)
        loss_unsafe_vector = unsafe_loss_weight * F.relu(B_unsafe + eps_unsafe) ## Added
        loss_unsafe = loss_unsafe_vector.sum() / unsafe_count ## Added
        unsafe_loss_sum += loss_unsafe.item() ## Added
        total_loss = total_loss + loss_unsafe ## Added

    # Calculate gradient loss (Lie derivative constraint) for safe states only
  
    # Get CBF gradient with respect to latent state
    grad_B = torch.autograd.grad(B_safe, z_current, grad_outputs=torch.ones_like(B_safe), retain_graph=True)[0]
    
    # Use affine dynamics to calculate x_dot
    f, g = dynamics(z_current)
    gu = torch.einsum('bsa,ba->bs', 
                        g.view(g.shape[0], dynamics.state_dim, dynamics.num_action), 
                        actions)
    x_dot = f + gu
    
    # Calculate Lie derivative: L_f h(x) = ∇h(x)ᵀf(x)
    b_dot = torch.sum(grad_B * x_dot, dim=1, keepdim=True)
    
    # CBF condition: ḣ(x) + αh(x) ≥ 0 for safe states
    # Equivalent to: L_f h(x) + αh(x) ≥ 0
    alpha = 1.0  # CBF parameter
    cbf_condition = b_dot + alpha * B_safe

    loss_grad_vector_safe_action = gradient_loss_weight * F.relu(eps_grad - cbf_condition)
    loss_grad_safe = loss_grad_vector_safe_action.sum() / batch_size
    gradient_loss = loss_grad_safe
    gradient_loss_sum += gradient_loss.item()
    total_loss = total_loss + gradient_loss

    # Calculate metrics for accuracy
    safe_correct = torch.sum(B_safe > 0).item() ## Added
    safe_h_sum = B_safe.sum().item() ## Added
    if (unsafe_states):
        unsafe_correct = torch.sum(B_unsafe < 0).item() ## Added
        unsafe_h_sum = B_unsafe.sum().item() ## Added
    
    # Calculate average metrics
    metrics = {
        'safe_accuracy': safe_correct / safe_count,
        # 'unsafe_accuracy': unsafe_correct / unsafe_count,
        'avg_safe_h': safe_h_sum / safe_count,
       # 'avg_unsafe_h': unsafe_h_sum / unsafe_count,
        'safe_loss': safe_loss_sum,
        # 'unsafe_loss': unsafe_loss_sum,
        'gradient_loss': gradient_loss_sum,
        'idbf_enabled': True, ## Added
        'unsafe_generated_batch':unsafe_count
    }
    if (unsafe_states):
        metrics['avg_unsafe_h']=unsafe_h_sum / (unsafe_count +1e-8)
        metrics['unsafe_loss']= unsafe_loss_sum
        metrics['unsafe_accuracy']= unsafe_correct / (unsafe_count+1e-8)
    else:
        metrics['avg_unsafe_h']=0
        metrics['unsafe_loss']= 0
        metrics['unsafe_accuracy']= 0
        
    
    return total_loss, metrics


def train_cbf(combined_model, cbf_model, train_loader, val_loader, optimizer, device, 
              epochs, safe_distance=4, eps_safe=0.05, eps_unsafe=0.2, 
              gradient_loss_weight=1.0,p=None):
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
            'idbf_enabled': True ## Added
            
        }
        
        # Training loop
        train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            optimizer.zero_grad()
            
            loss, metrics = compute_cbf_loss(
                cbf_model, combined_model, batch, device,
                safe_distance=safe_distance, eps_safe=eps_safe, eps_unsafe=eps_unsafe,
                gradient_loss_weight=gradient_loss_weight,
             
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
                'idbf_enabled': metrics['idbf_enabled'], ## Added
                'train_gradient_loss': metrics['gradient_loss'],
                'unsafe samples generated': metrics['unsafe_generated_batch'],
                'global_step': global_step,
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
            

        }
        
        val_batches = 0
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
            # For validation, we need to recreate the computation graph for gradient calculation
            cbf_model.requires_grad_(True)
            
            val_loss, val_metrics = compute_cbf_loss(
                cbf_model, combined_model, batch, device,
                safe_distance=safe_distance, eps_safe=eps_safe, eps_unsafe=eps_unsafe,
                gradient_loss_weight=gradient_loss_weight,p=p
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
            'epoch': epoch
        })
        
        # Check if this is the best model so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(cbf_model.state_dict(), f'best_cbf_VAE_Deterministic_4_cql_{randomnb}.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
  
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}")
        print(f"  Train Safe Accuracy: {epoch_train_metrics['safe_accuracy']:.4f}")
        print(f"  Val Safe Accuracy: {epoch_val_metrics['safe_accuracy']:.4f}")
        print(f"  Train Unsafe Accuracy: {epoch_train_metrics['unsafe_accuracy']:.4f}")
        print(f"  Val Unsafe Accuracy: {epoch_val_metrics['unsafe_accuracy']:.4f}")
        print(f"  val_gradient_loss': {epoch_val_metrics['gradient_loss']:.4f}")
    
    # Save final model
    torch.save(cbf_model.state_dict(), f"best_cbf_VAE_Deterministic_4_cql_final{randomnb}.pth")
    print("Training completed. Final model saved.")
    
    return cbf_model

class CombinedDataset(Dataset):
    def __init__(self, data):
        """
        Initialize dataset with sequence creation and prediction horizon
        Args:
            data (dict): Dictionary containing actions, states, and done flags
            sequence_length (int): Total sequence length
            prediction_horizon (int): Number of steps to predict ahead
        """  
        rgb_states=data["states_rgb"]##################################this is dataset size,width,height,3
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        costs = data["costs"]
        rewards = data["rewards"]
        # episode_starts = data["episode_starts"]
        episode_lengths = data["episode_lengths"]
        

        # Identify safe trajectories (episodes with zero cost throughout)
        safe_episodes = []
        start_idx = 0
        safe_rewards = []
        safe_costs = []
        for length in episode_lengths:
            end_idx = start_idx + length
            if np.all(costs[start_idx:end_idx] == 0):
                safe_episodes.append((start_idx, end_idx))
                safe_rewards.append(np.mean(rewards[start_idx:end_idx]))
                safe_costs.append(np.mean(costs[start_idx:end_idx]))
            start_idx = end_idx
        
        # Extract states, actions, and next_states from safe trajectories
        self.safe_indices = np.concatenate([np.arange(start, end) for start, end in safe_episodes])
        # states = states[safe_indices]
        # actions = actions[safe_indices]
        # next_states = next_states[safe_indices]
        # rgb_states=rgb_states[safe_indices]################################still need to permute them
        
        avg_safe_reward = np.mean(safe_rewards) if safe_rewards else 0
        avg_safe_cost = np.mean(safe_costs) if safe_costs else 0
        
        print(f"Using {len(states)} safe transitions from {len(safe_episodes)} safe trajectories out of {len(episode_lengths)} total trajectories")
        print(f"Average state reward of safe trajectories: {avg_safe_reward:.4f}")
        print(f"Average state cost of safe trajectories: {avg_safe_cost:.4f}")
        
        self.actions = torch.FloatTensor(actions[self.safe_indices])
        self.states_rgb = torch.FloatTensor(rgb_states[self.safe_indices]).permute(0, 3, 1, 2) / 255.0
        # self.dones = torch.FloatTensor(data['dones'])
        self.ground_truth_states = torch.FloatTensor(states[self.safe_indices])
        self.ground_truth_next_states = torch.FloatTensor(next_states[self.safe_indices])
      

    def __len__(self):
        return len(self.safe_indices)
    def __getitem__(self, idx):
        
        return {
            'actions': self.actions[idx],
            'states_rgb': self.states_rgb[idx],
            'ground_truth_states': self.ground_truth_states[idx],
            'ground_truth_next_states': self.ground_truth_next_states[idx]
        }
###
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
    
    parser.add_argument("--p", type=int, default=0.000001, help="")

    args = parser.parse_args()
    return vars(args)  # Convert argparse Namespace to dictionary

def main():
    config = parse_args()
    seed_everything(config['seed'])



    # Initialize wandb
    wandb.init(project="cbf-training-latent-dynamics", name=f"cbf-training-latent-space_deterministic_idbf_{randomnb}")

    wandb.config.update(config)
    

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
        p=config['p']
        

       
    )
    
    # Finish wandb session
    wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()