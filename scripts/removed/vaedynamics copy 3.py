import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RecursiveEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, action_dim):
        super(RecursiveEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 + latent_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)
    
    def forward(self, I_k, x_k_minus_1, u_k_minus_1):
        I_k = torch.relu(self.conv1(I_k))
        I_k = I_k.view(I_k.size(0), -1)  # Flatten
        x_u_concat = torch.cat([I_k, x_k_minus_1, u_k_minus_1], dim=1)
        x_k = torch.relu(self.fc1(x_u_concat))
        x_k = self.fc2(x_k)
        return x_k

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x_k):
        x_k = torch.relu(self.fc1(x_k))
        I_hat = self.fc2(x_k)
        return I_hat

class DynamicsModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)
    
    def forward(self, x_k_minus_1, u_k_minus_1):
        x_u_concat = torch.cat([x_k_minus_1, u_k_minus_1], dim=1)
        x_k_pred = torch.relu(self.fc1(x_u_concat))
        x_k_pred = self.fc2(x_k_pred)
        return x_k_pred

def compute_loss(x_k, x_k_pred, I_k, I_hat, I_pred, wstate=1.0, wrec1=1.0, wrec2=1.0):
    """
    Compute the dynamic loss.
    Args:
        x_k: Ground truth latent state at time step k.
        x_k_pred: Predicted latent state from the dynamics model.
        I_k: Ground truth observation at time step k.
        I_hat: Reconstructed observation from the current latent state.
        I_pred: Reconstructed observation from the predicted latent state.
        wstate: Weight for the latent state prediction error.
        wrec1: Weight for the first observation reconstruction error.
        wrec2: Weight for the second observation reconstruction error.
    Returns:
        Total loss value.
    """
    # Latent state prediction error
    state_loss = torch.mean((x_k_pred - x_k) ** 2)
    
    # Observation reconstruction error
    rec_loss1 = torch.mean((I_pred - I_k) ** 2)
    rec_loss2 = torch.mean((I_hat - I_k) ** 2)

    # Compute the total loss
    total_loss = wstate * state_loss + wrec1 * rec_loss1 + wrec2 * rec_loss2
    return total_loss

def main():
    # Model Initialization
    input_dim = 3  # RGB image
    latent_dim = 128
    action_dim = 4  # Action size (for example, control inputs)

    encoder = RecursiveEncoder(input_dim=input_dim, latent_dim=latent_dim, action_dim=action_dim)
    decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)  # Reconstruct RGB images
    dynamics_model = DynamicsModel(latent_dim=latent_dim, action_dim=action_dim)

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics_model.parameters()), lr=1e-4)

    # Example data
    batch_size = 32
    height, width = 64, 64  # Example image size
    I_k_batch = torch.rand(batch_size, input_dim, height, width)  # Current measurements (images)
    x_k_minus_1_batch = torch.rand(batch_size, latent_dim)  # Previous latent state
    u_k_minus_1_batch = torch.rand(batch_size, action_dim)  # Previous action

    # Training loop
    num_epochs = 100
    num_batches = 10  # Example number of batches per epoch

    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            I_k = I_k_batch[batch_idx]  # Current image
            x_k_minus_1 = x_k_minus_1_batch[batch_idx]  # Previous latent state
            u_k_minus_1 = u_k_minus_1_batch[batch_idx]  # Previous action

            # Forward pass through the recursive encoder
            x_k = encoder(I_k, x_k_minus_1, u_k_minus_1)
            
            # Forward pass through the decoder to get the reconstruction of the current latent state
            I_hat = decoder(x_k)
            
            # Use the dynamics model to predict the next latent state
            x_k_pred = dynamics_model(x_k_minus_1, u_k_minus_1)
            
            # Forward pass through the decoder to get the reconstructed observation from the predicted latent state
            I_pred = decoder(x_k_pred)

            # Compute the loss
            loss = compute_loss(x_k, x_k_pred, I_k, I_hat, I_pred, wstate=1.0, wrec1=1.0, wrec2=1.0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
    main()
