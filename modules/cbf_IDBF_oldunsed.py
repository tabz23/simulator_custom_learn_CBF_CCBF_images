import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import wandb  # Import wandb
from tqdm import trange

current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from modules.dataset import *
from modules.network import *
from modules.trainer_ccbf import *
from envs.car import *
from matplotlib.collections import LineCollection
import random


DATASET_PATH = "safe_rl_dataset.npz"  # Updated path to store/load dataset
goal_position=np.array([20, 21])
BC_MODEL_PATH = "bc_model.pt"
BC_SAFE_MODEL_PATH = "bc_safe_model.pt"
DATASET_PATH = "safe_rl_dataset.npz"
GOAL_POSITION = np.array([20, 21])
rng = random.Random()  # This uses a new random state
random_value = rng.randint(100, 999)
DEVICE="cpu"

def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def load_dataset():
    """Load the dataset from the specified path."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        return data
    else:
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
seed_everything(42)
#load full dataset
data = load_dataset()
# Define model parameters
state_dim = 4  # car (2) + obstacle (2)
action_dim = 2
hidden_sizes = (64, 64)
action_low = np.array([-3.0, -3.0])
action_high = np.array([3.0, 3.0])
bc_safe_model_exists = os.path.exists(BC_SAFE_MODEL_PATH)

bc_safe_model = MLPGaussianActor(
    obs_dim=state_dim,
    act_dim=action_dim,
    action_low=action_low,
    action_high=action_high,
    hidden_sizes=hidden_sizes,
    activation=nn.ReLU,
    device=DEVICE
).to(DEVICE)

print(f"Loading BC safe model from {BC_SAFE_MODEL_PATH}")
bc_safe_model.load_state_dict(torch.load(BC_SAFE_MODEL_PATH, map_location=DEVICE))
bc_safe_model.to(torch.float32)

"""Create a dataloader from the provided dataset."""
states = data["states"]
actions = data["actions"]
next_states = data["next_states"]
costs = data["costs"]
rewards = data["rewards"]
episode_starts = data["episode_starts"]
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
safe_indices = np.concatenate([np.arange(start, end) for start, end in safe_episodes])
states = states[safe_indices]
actions = actions[safe_indices]
next_states = next_states[safe_indices]
avg_safe_reward = np.mean(safe_rewards) if safe_rewards else 0
avg_safe_cost = np.mean(safe_costs) if safe_costs else 0
print(f"Using {len(states)} safe transitions from {len(safe_episodes)} safe trajectories out of {len(episode_lengths)} total trajectories")
print(f"Average state reward of safe trajectories: {avg_safe_reward:.4f}")
print(f"Average state cost of safe trajectories: {avg_safe_cost:.4f}")

env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=GOAL_POSITION, obstacle_radius=4.0)
dataset = Dataset(state_car_dim=2, state_obstacles_dim=2, control_dim=2, buffer_size=1000000, safe_distance=4.0)

print("Populating dataset object with safe trajectories...")
for i in trange(len(states), desc="Populating dataset"):
    dataset.add_data(states[i], actions[i], next_states[i], idbf_masks=True)
print(f"Dataset populated with {len(states)} transitions")
# wandb.log({"dataset_populated": True, "dataset_size": len(states)})

print("safe buffer has dimensions: ",len(dataset.buffer_safe))
print("unsafe buffer has dimensions :",len(dataset.buffer_unsafe))

wandb.init(
    project="ccbf-car-navigation",
    name=f"{random_value}",
    config={
        "environment": "DubinsCarEnv",
        "goal_position": goal_position.tolist(),
        "max_velocity": 3.0,
        "dt": 0.1,
        "safe_distance": 4.0,
        "seed": 42,
    }
)

use_mps = torch.backends.mps.is_available()
# Initialize CBF
num_hidden_dim=3
dim_hidden=32
use_cql_actions=False
cql_actions_weight=0.01
cql_states_weight=0.1
num_action_samples=5
idbf_data=True
eps_safe=0.2
eps_unsafe=0.2
cbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=num_hidden_dim, dim_hidden=dim_hidden)
cbf = cbf.to("mps" if use_mps else "cpu")
wandb.config.update({
    "num_hidden_dim":num_hidden_dim,
    "dim_hidden":dim_hidden,
    "use_cql_actions": use_cql_actions,

    "safe_distance": 4.0,
    "eps_safe": eps_safe,
    "eps_unsafe": eps_unsafe,
    "safe_loss_weight": 1,
    "unsafe_loss_weight": 1,
    "action_loss_weight": 1,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "cql_actions_weight":cql_actions_weight,
    "cql_states_weight": cql_states_weight,
    "num_action_samples": 10,
    "num_state_samples": 8,
    "state_sample_std": 0.001,
    "step_count": 10000,
    "num_action_samples":5

        
    })

# Setup trainer with new CQL parameters
trainer = Trainer(
    cbf, dataset, 
    safe_distance=4, 
    idbf_data=True,
    eps_safe=eps_safe, 
    eps_unsafe=eps_unsafe,
    safe_loss_weight=1, 
    unsafe_loss_weight=1, 
    action_loss_weight=1,
    dt=0.1, 
    batch_size=128, 
    opt_iter=1, 
    lr=1e-4, 
    device="mps" if use_mps else "cpu",
    # CQL parameters
    use_cql_actions=use_cql_actions, 
    cql_actions_weight=cql_actions_weight,  # Weight for L_CQL_actions loss
    cql_states_weight=cql_states_weight,  # Weight for L_CQL_states loss
    num_action_samples=num_action_samples,   # Number of random actions to sample
    num_state_samples=8,     # Number of nearby states to sample
    state_sample_std=0.1     # Standard deviation for state sampling
)

# Create lists to track metrics (extended to track CQL losses)
losses = []
accuracies = []
safe_h_values = []
unsafe_h_values = []
cql_action_losses = []
cql_state_losses = []
    
# Train CBF with periodic logging and checkpoint saving
print("Training cbf with indbf dataset...")
step_count=20000

# Create tables for detailed training logs in wandb
wandb.define_metric("step")
wandb.define_metric("training/*", step_metric="step")

for i in range(step_count):
    # Log step to wandb
    wandb.log({"step": i})
    
    # Train one step
    acc_np, loss_np, avg_safe_h, avg_unsafe_h = trainer.train_cbf()
    
    # Unpack loss components for better logging
    safe_loss, unsafe_loss, deriv_loss, cql_actions_loss, cql_states_loss = loss_np
    
    # Store metrics for later analysis
    losses.append(loss_np)
    accuracies.append(acc_np)
    safe_h_values.append(avg_safe_h)
    unsafe_h_values.append(avg_unsafe_h)
    cql_action_losses.append(cql_actions_loss)
    cql_state_losses.append(cql_states_loss)
    
    # Log metrics to wandb
    wandb.log({
        "training/acc_safe":acc_np[0],
        "training/acc_unsafe":acc_np[1],
        "training/total_loss": np.sum(loss_np),
        "training/safe_loss": safe_loss,
        "training/unsafe_loss": unsafe_loss,
        "training/deriv_loss": deriv_loss,
        "training/cql_actions_loss": cql_actions_loss,
        "training/cql_states_loss": cql_states_loss,
        "training/accuracy": acc_np,
        "training/safe_h_value": avg_safe_h,
        "training/unsafe_h_value": avg_unsafe_h,
        "training/h_value_gap": avg_safe_h - avg_unsafe_h,
        "training/progress": i / step_count,
        "IDBF":"TRUE"
    })
    
    # Save checkpoint every 1000 steps (or modify frequency as needed)
    if (i+1) % 5000 == 0:
        checkpoint = {
            'step': i,
            'model_state_dict': trainer.cbf.state_dict(),
            'loss': loss_np,
            'acc': acc_np,
            'cql_params': {
                'cql_actions_weight': trainer.cql_actions_weight,
                'cql_states_weight': trainer.cql_states_weight,
                'num_action_samples': trainer.num_action_samples,
                'num_state_samples': trainer.num_state_samples,
                'state_sample_std': trainer.state_sample_std
            }
        }
        checkpoint_filename = f'IDBF_ccbf_checkpoint_{random_value}.pt'
        torch.save(checkpoint, checkpoint_filename)
        print(f"Checkpoint saved to {checkpoint_filename}")

# Save final metrics for analysis
metrics = {
    'losses': np.array(losses),
    'accuracies': np.array(accuracies),
    'safe_h_values': np.array(safe_h_values),
    'unsafe_h_values': np.array(unsafe_h_values),
    'cql_action_losses': np.array(cql_action_losses),
    'cql_state_losses': np.array(cql_state_losses)
}
torch.save(metrics, 'ccbf_training_metrics.pt')

print("CCBF Training complete")

#safe buffer has dimensions:  214587
# unsafe buffer has dimensions : 225817