import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import argparse

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from modules.dataset import *
from modules.network import *
from idbf_implementation import train_bc_model, ImplicitDiverseBarrierFunctionTrainer

DATASET_PATH = "safe_rl_dataset.npz"

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
    """Loads the dataset from file."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        return data
    else:
        print(f"Dataset not found at {DATASET_PATH}")
        return None

class DatasetWrapper:
    """Wrapper for the dataset to provide sampling functionality."""
    def __init__(self, data, device="cpu"):
        self.states = data["states"]
        self.actions = data["actions"]
        self.next_states = data["next_states"]
        self.size = len(self.states)
        self.device = device
        print(f"Dataset loaded with {self.size} samples")
        
    def sample_data(self, batch_size):
        """Sample a batch of data from the dataset."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices], self.actions[indices], self.next_states[indices]

def train_idbf(args):
    """Train the iDBF model."""
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    data = load_dataset()
    if data is None:
        print("No dataset found! Please run generate_dataset() first.")
        return
    
    # Create dataset wrapper
    dataset = DatasetWrapper(data, device=device)
    
    # Create CBF model
    cbf = CBF(
        state_car_dim=2, 
        state_obstacles_dim=2, 
        dt=0.1, 
        num_hidden_dim=args.num_hidden_dim,
        dim_hidden=args.dim_hidden
    ).to(device)
    
    # Define action bounds (for dubins car)
    action_low = np.array([-3.0, -3.0])  # Lower bound of actions
    action_high = np.array([3.0, 3.0])   # Upper bound of actions
    
    # Train BC model first
    print("\n=== Training Behavioral Cloning Model ===")
    bc_model = train_bc_model(
        dataset=dataset,
        obs_dim=4,  # state_car_dim + state_obstacles_dim
        act_dim=2,  # Action dimension for dubins car
        action_low=action_low,
        action_high=action_high,
        device=device,
        num_iterations=args.bc_iterations
    )
    
    # Create iDBF trainer
    trainer = ImplicitDiverseBarrierFunctionTrainer(
        cbf=cbf,
        dataset=dataset,
        bc_model=bc_model,
        safe_distance=args.safe_distance,
        eps_safe=args.eps_safe,
        eps_unsafe=args.eps_unsafe,
        safe_loss_weight=args.safe_loss_weight,
        unsafe_loss_weight=args.unsafe_loss_weight,
        dt=0.1,
        device=device,
        batch_size=args.batch_size,
        opt_iter=args.opt_iter,
        lr=args.lr,
        use_cql_actions=args.use_cql_actions,
        use_cql_states=args.use_cql_states,
        cql_actions_weight=args.cql_actions_weight,
        cql_states_weight=args.cql_states_weight,
        num_candidate_actions=args.num_candidate_actions,
        bc_threshold=args.bc_threshold,
        contrastive_weight=args.contrastive_weight
    )
    
    # Training loop
    print("\n=== Training iDBF Model ===")
    num_epochs = args.num_epochs
    save_interval = args.save_interval
    
    # Initialize lists to store metrics
    epoch_losses = []
    epoch_accs = []
    contrastive_counts = []
    
    for epoch in range(num_epochs):
        # Train for one epoch
        acc, losses, avg_safe_h, avg_unsafe_h, num_contrastive = trainer.train_cbf()
        
        # Store metrics
        epoch_losses.append(losses)
        epoch_accs.append(acc)
        contrastive_counts.append(num_contrastive)
        
        # Print progress
        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Accuracies: Safe h: {acc[0]:.4f}, Unsafe h: {acc[1]:.4f}, Safe deriv: {acc[2]:.4f}")
            print(f"  Losses: Safe h: {losses[0]:.6f}, Unsafe h: {losses[1]:.6f}, Safe deriv: {losses[2]:.6f}")
            print(f"  CQL Losses: Actions: {losses[3]:.6f}, States: {losses[4]:.6f}")
            print(f"  Contrastive Loss: {losses[5]:.6f}, Contrastive Samples: {num_contrastive}")
            print(f"  Average h: Safe: {avg_safe_h:.4f}, Unsafe: {avg_unsafe_h:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint_name = f"idbf_checkpoint_{epoch + 1}_"\
                             f"cql_states{args.use_cql_states}"\
                             f"cql_actions{args.use_cql_actions}_"\
                             f"cql_states_weight{args.cql_states_weight}_"\
                             f"cql_actions_weight{args.cql_actions_weight}_"\
                             f"{int(np.random.rand() * 1000)}.pt"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': cbf.state_dict(),
                'optimizer_state_dict': trainer.cbf_optimizer.state_dict(),
                'safe_accuracy': acc[0],
                'unsafe_accuracy': acc[1],
                'deriv_accuracy': acc[2],
            }, checkpoint_name)
            
            print(f"Checkpoint saved to {checkpoint_name}")
    
    # Plot training metrics
    plt.figure(figsize=(15, 10))
    
    # Plot accuracies
    plt.subplot(2, 2, 1)
    accs = np.array(epoch_accs)
    plt.plot(range(1, num_epochs + 1), accs[:, 0], label='Safe h Accuracy')
    plt.plot(range(1, num_epochs + 1), accs[:, 1], label='Unsafe h Accuracy')
    plt.plot(range(1, num_epochs + 1), accs[:, 2], label='Safe Derivative Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracies')
    plt.grid(True, alpha=0.3)
    
    # Plot losses
    plt.subplot(2, 2, 2)
    losses = np.array(epoch_losses)
    plt.plot(range(1, num_epochs + 1), losses[:, 0], label='Safe h Loss')
    plt.plot(range(1, num_epochs + 1), losses[:, 1], label='Unsafe h Loss')
    plt.plot(range(1, num_epochs + 1), losses[:, 2], label='Safe Derivative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True, alpha=0.3)
    
    # Plot CQL losses
    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs + 1), losses[:, 3], label='CQL Actions Loss')
    plt.plot(range(1, num_epochs + 1), losses[:, 4], label='CQL States Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CQL Losses')
    plt.grid(True, alpha=0.3)
    
    # Plot contrastive metrics
    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs + 1), losses[:, 5], label='Contrastive Loss')
    plt.plot(range(1, num_epochs + 1), np.array(contrastive_counts) / args.opt_iter, label='Contrastive Samples/Iter')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Contrastive Metrics')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"idbf_training_metrics_{int(np.random.rand() * 1000)}.png")
    plt.show()
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an iDBF model")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval for saving checkpoints")
    parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging")
    
    # BC model parameters
    parser.add_argument("--bc_iterations", type=int, default=5000, help="Number of BC training iterations")
    
    # CBF model parameters
    parser.add_argument("--num_hidden_dim", type=int, default=3, help="Number of hidden layers in CBF")
    parser.add_argument("--dim_hidden", type=int, default=32, help="Width of hidden layers in CBF")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--opt_iter", type=int, default=10, help="Number of optimization iterations per epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Loss weights and thresholds
    parser.add_argument("--safe_distance", type=float, default=4.0, help="Safe distance from obstacles")
    parser.add_argument("--eps_safe", type=float, default=0.02, help="Margin for safe states")
    parser.add_argument("--eps_unsafe", type=float, default=0.02, help="Margin for unsafe states")
    parser.add_argument("--safe_loss_weight", type=float, default=1.0, help="Weight for safe loss")
    parser.add_argument("--unsafe_loss_weight", type=float, default=1.5, help="Weight for unsafe loss")
    
    # CQL parameters
    parser.add_argument("--use_cql_actions", action="store_true", help="Use CQL actions loss")
    parser.add_argument("--use_cql_states", action="store_true", help="Use CQL states loss")
    parser.add_argument("--cql_actions_weight", type=float, default=0.01, help="Weight for CQL actions loss")
    parser.add_argument("--cql_states_weight", type=float, default=0.1, help="Weight for CQL states loss")
    
    # iDBF parameters
    parser.add_argument("--num_candidate_