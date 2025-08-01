import os
import sys

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
from modules.network import CBF  # Assuming this is still relevant
import random
from tqdm import trange, tqdm
import torchmetrics
from torchvision import models  # Import torchvision models

rng = random.Random()  # This uses a new random state
random_value = rng.randint(100, 999)


def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Create a multi-layer perceptron with given sizes and activations."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class DynamicsDataset(Dataset):
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

            # Skip sequences with done flags in the first prediction horizon
            if torch.any(sequence_dones[:self.prediction_horizon] == 1):
                i += 1
                continue

            sequence = {
                'actions': sequence_actions,
                'states_rgb': sequence_states,
                'dones': sequence_dones
            }

            sequences.append(sequence)
            i += 1

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class RecursiveEncoder(nn.Module):
    """
    Recursive encoder that takes current observation,
    previous latent state, and previous action
    """

    def __init__(self, input_channels, latent_dim, action_dim, hidden_dim=64, use_pretrained_encoder=False):
        super().__init__()
        self.use_pretrained_encoder = use_pretrained_encoder

        if use_pretrained_encoder:
            # Using a pre-trained ResNet model (you can choose others)
            resnet = models.resnet18(pretrained=True)
            self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
            # Modify the first convolutional layer to accept the number of input channels
            self.image_encoder[0] = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Adjust the output dimension to match hidden_dim
            self.image_feature_dim = resnet.fc.in_features  # Get the feature dimension from ResNet
            self.image_to_hidden = nn.Linear(self.image_feature_dim, hidden_dim)  # Linear layer to reduce dimension
        else:
            # Image encoder
            self.image_encoder = nn.Sequential(
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

        # Fusion layer for image, previous state, and action
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

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
        if self.use_pretrained_encoder:
            image_features = self.image_encoder(current_obs)
            image_features = image_features.view(image_features.size(0), -1)  # Flatten
            image_features = self.image_to_hidden(image_features)  # Reduce dimension
        else:
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
            nn.Linear(latent_dim, 128 * 8 * 8),  # Project to the size before conv transpose
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),  # Reshape to a feature map

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # Output: 64x64
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


class DeterministicDynamicsModel(nn.Module):
    """Deterministic Dynamics Model with Recursive Encoder and Decoder"""

    def __init__(self, input_channels, latent_dim, action_dim, hidden_dim=64, use_pretrained_encoder=False):
        super().__init__()

        # Recursive encoder
        self.encoder = RecursiveEncoder(
            input_channels, latent_dim, action_dim, hidden_dim, use_pretrained_encoder=use_pretrained_encoder
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


def compute_multiple_shooting_loss(model, batch, device,
                                    prediction_horizon=5,
                                    wstate=5.0,
                                    wrec1=5.0,
                                    wrec2=5.0):
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

    ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
    for t in range(1, seq_length):
        # Current states and actions
        current_state = states[:, t]
        current_action = actions[:, t]

        # Forward pass
        z_current, x_recon_current, z_predicted = model(
            current_state, z_prev, prev_action
        )

        # Multiple shooting loss components
        # Prediction error for future states

        prediction_loss = F.mse_loss(z_predicted, z_current)
        # ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
        # Reconstruction losses
        # recon_loss1 = F.mse_loss(
        #     model.decoder(z_predicted),  # Predicted state reconstruction
        #     current_state
        # )
        recon_loss1 = 1 - ssim(model.decoder(z_predicted), current_state)

        recon_loss2 = 1 - ssim(x_recon_current, current_state)
        # recon_loss2 = F.mse_loss(
        #     x_recon_current,  # Direct observation reconstruction
        #     current_state
        # )

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


def train(model, train_loader, val_loader, optimizer, device, epochs=100):
    """Training function with validation and step-wise logging"""
    model.to(device)
    best_val_loss = float('inf')

    wandb.init(project="deterministic-dynamics-model", name="training_run")

    for epoch in trange(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):  # Keep track of batch index
            optimizer.zero_grad()

            loss = compute_multiple_shooting_loss(model, batch, device)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Log metrics every step


        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                val_loss = compute_multiple_shooting_loss(model, batch, device)
                total_val_loss += val_loss.item()

                # Log validation metrics every step
                wandb.log({
                    'val_loss_step': val_loss.item(),
                    'epoch': epoch + (batch_idx / len(val_loader)),  # for x axis
                })

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Logging epoch average


        # Model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'CCBF_images_model_final_fixedaction.pth')
            print("saved model")

        print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

    wandb.finish()
    return model


def main():
    # Load dataset
    data = np.load("safe_rl_dataset_images_ALLUNSAFE_big_obstacle.npz")
    seed_everything(1)
    # Hyperparameters
    input_channels = 3
    latent_dim = 8
    action_dim = 2
    hidden_dim = 400
    batch_size = 32
    learning_rate = 1e-4
    epochs = 100
    validation_split = 0.2
    use_pretrained_encoder = True # set to False to avoid errors if not using
    # Device configuration
    device = "mps"

    # Create dataset
    full_dataset = DynamicsDataset(data, sequence_length=6, prediction_horizon=6)

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
        hidden_dim=hidden_dim,
        use_pretrained_encoder=use_pretrained_encoder
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model = train(model, train_loader, val_loader, optimizer, device, epochs)

    # Save final model
    torch.save(trained_model.state_dict(), 'final_dynamics_model.pth')


if __name__ == "__main__":
    main()
