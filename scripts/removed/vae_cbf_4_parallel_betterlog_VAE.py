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

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Import modules
from modules.network import CBF

# Import VAE and dynamics models from the first file
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Create a multi-layer perceptron with given sizes and activations."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

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

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

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
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


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

        self.f = mlp([self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim], activation=nn.ReLU)
        self.g = mlp(
            [self.state_dim] + num_layers * [self.hidden_dim] + [self.state_dim * self.num_action],
            activation=nn.ReLU)

    def forward(self, state):
        return self.f(state), self.g(state)

    def forward_x_dot(self, state, action):
        f, g = self.forward(state)
        gu = torch.einsum('bsa,ba->bs', g.view(g.shape[0], self.state_dim, self.num_action), action)
        x_dot = f + gu
        return x_dot

    def forward_next_state(self, state, action):
        return self.forward_x_dot(state, action) * self.dt + state


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
        _, mu, logvar, z = self.vae(x)
        return z
        
    def forward_dynamics(self, z, action):
        """Predict next state in latent space."""
        return self.dynamics_model(z, action)
        
    def forward(self, current_state_rgb, z_prev, prev_action):
        """Full forward pass: encode current state and get predicted state."""
        z_current = self.encode(current_state_rgb)
        # For dynamics prediction (not used here but keeping for consistency)
        z_predicted = None 
        return z_current, z_predicted, z_prev


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
                'ground_truth_states': sequence_ground_truth_states
            }

            sequences.append(sequence)
            i += 1

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def compute_cbf_loss(cbf_model, combined_model, batch, device,
                     safe_distance=4, eps_safe=0.05, eps_unsafe=0.2,
                     safe_loss_weight=1.0, unsafe_loss_weight=1.05,
                     gradient_loss_weight=1.0, dt=0.1, eps_grad=0.01):
    """Compute the CBF loss."""
    ground_truth_states = batch['ground_truth_states'].to(device)
    actions = batch['actions'].to(device)
    states_rgb = batch['states_rgb'].to(device)
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

    for t in range(seq_length):
        # Process current timestep
        current_state_rgb = states_rgb[:, t]
        current_action = actions[:, t] if t < seq_length - 1 else torch.zeros_like(actions[:, 0])
        ground_truth_state = ground_truth_states[:, t]

        # Encode current state to latent representation (without gradients for VAE)
        with torch.no_grad():
            _, _, _, z_current_detached = combined_model.vae(current_state_rgb)
        
        # Create a copy that requires gradients for CBF calculations
        z_current = z_current_detached.clone().detach().requires_grad_(True)

        # Define safe and unsafe sets based on ground truth states
        # Note: This uses the ground truth states, not latent states
        is_safe = torch.norm(ground_truth_state[:, :2] - ground_truth_state[:, 2:], dim=1) > safe_distance
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
            grad_B = torch.autograd.grad(B.sum(), z_current, create_graph=True)[0]
            
            # Use affine dynamics to calculate x_dot
            f, g = dynamics(z_current)
            gu = torch.einsum('bsa,ba->bs', 
                              g.view(g.shape[0], dynamics.state_dim, dynamics.num_action), 
                              current_action)
            x_dot = f + gu
            
            # Calculate Lie derivative: L_f h(x) = ∇h(x)ᵀf(x)
            b_dot = torch.sum(grad_B * x_dot, dim=1, keepdim=True)
            
            # CBF condition: ḣ(x) + αh(x) ≥ 0 for safe states
            # Equivalent to: L_f h(x) + αh(x) ≥ 0
            alpha = 1.0  # CBF parameter
            cbf_condition = b_dot + alpha * B
            
            # Loss when condition is violated
            loss_grad_vector = gradient_loss_weight * F.relu(eps_grad - cbf_condition) * safe_mask
            num_grad_elements = safe_mask.sum() + 1e-8
            loss_grad = loss_grad_vector.sum() / num_grad_elements
            gradient_loss_sum += loss_grad.item()
            total_loss = total_loss + loss_grad

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
        'gradient_loss': gradient_loss_sum / (seq_length - 1) if seq_length > 1 else 0
    }
    
    return total_loss, metrics


def train_cbf(combined_model, cbf_model, train_loader, val_loader, optimizer, device, 
              epochs, safe_distance=4, eps_safe=0.05, eps_unsafe=0.2, 
              gradient_loss_weight=1.0, checkpoint_interval=5):
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
            'gradient_loss': 0.0
        }
        
        # Training loop
        train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            optimizer.zero_grad()
            
            loss, metrics = compute_cbf_loss(
                cbf_model, combined_model, batch, device,
                safe_distance=safe_distance, eps_safe=eps_safe, eps_unsafe=eps_unsafe,
                gradient_loss_weight=gradient_loss_weight
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
            'gradient_loss': 0.0
        }
        
        val_batches = 0
        # with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
            # For validation, we need to recreate the computation graph for gradient calculation
            cbf_model.requires_grad_(True)
            
            val_loss, val_metrics = compute_cbf_loss(
                cbf_model, combined_model, batch, device,
                safe_distance=safe_distance, eps_safe=eps_safe, eps_unsafe=eps_unsafe,
                gradient_loss_weight=gradient_loss_weight
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
            torch.save(cbf_model.state_dict(), 'best_cbf_VAE_another.pth')
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
    
    # Save final model
    torch.save(cbf_model.state_dict(), 'final_cbf_model.pth')
    print("Training completed. Final model saved.")
    
    return cbf_model


def main():
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Initialize wandb
    wandb.init(project="cbf-training-latent-dynamics", name="cbf-training-latent-space")
    
    # Parameters and configuration
    config = {
        # Dataset parameters
        'dataset_path': "safe_rl_dataset_images_ALLUNSAFE_big_obstacle.npz",
        'sequence_length': 6,
        'prediction_horizon': 6,
        'validation_split': 0.2,
        
        # Model parameters
        'input_channels': 3,
        'latent_dim': 4,
        'action_dim': 2,
        'hidden_dim': 400,
        'cbf_hidden_dim': 64,
        'cbf_layers': 3,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 20,
        
        # CBF parameters
        'safe_distance': 4.0,
        'eps_safe': 0.08,
        'eps_unsafe': 0.1,
        'gradient_loss_weight': 1.0,
        
        # Checkpoint interval
        'checkpoint_interval': 1
    }
    
    # Log configuration to wandb
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
        vae.load_state_dict(torch.load('vae_model_final_2.pth', map_location=device))
        dynamics_model.load_state_dict(torch.load('dynamics_model_final_2.pth', map_location=device))
        
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
        checkpoint_interval=config['checkpoint_interval']
    )
    
    # Finish wandb session
    wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()