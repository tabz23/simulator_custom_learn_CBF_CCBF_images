import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from vaedynamics_newstart_good_3_decoderaffine import DeterministicDynamicsModel, DynamicsDataset
from modules.network import CBF

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
latent_dim = 4
action_dim = 2
cbf_hidden_dim = 128
batch_size = 64
lr_cbf = 1e-4
safe_distance = 4.0
eps_safe = 0.02
eps_unsafe = 0.02
safe_loss_weight = 1.0
unsafe_loss_weight = 1.5
cql_actions_weight = 0.1
num_action_samples = 10
temp = 0.7

class TransitionDataset(Dataset):
    def __init__(self, latent_states, actions, ground_truth_states, dones):
        self.latent_states = latent_states
        self.actions = actions
        self.ground_truth_states = ground_truth_states
        self.dones = dones

    def __len__(self):
        return len(self.latent_states) - 1  # Since next state is required

    def __getitem__(self, idx):
        if self.dones[idx]:
            # If current is terminal, next is invalid; skip by returning next index
            return self[(idx + 1) % len(self)]
        return {
            'current_latent': self.latent_states[idx],
            'action': self.actions[idx],
            'next_latent': self.latent_states[idx + 1],
            'ground_truth': self.ground_truth_states[idx],
            'done': self.dones[idx]
        }

class CBFTrainer:
    def __init__(self, cbf, dynamics_model, dataset, device):
        self.cbf = cbf
        self.dynamics_model = dynamics_model
        self.dataset = dataset
        self.device = device
        self.optimizer = optim.Adam(self.cbf.parameters(), lr=lr_cbf, weight_decay=5e-5)
        self.loss_fn = nn.MSELoss()

    def get_mask(self, ground_truth):
        diff = ground_truth[:, :2] - ground_truth[:, 2:4]
        dist = torch.norm(diff, dim=1)
        safe_mask = dist > safe_distance
        unsafe_mask = ~safe_mask
        return safe_mask, unsafe_mask

    def compute_h_dot(self, latent, action):
        latent.requires_grad_(True)
        B = self.cbf(latent)
        grad_B = torch.autograd.grad(B.sum(), latent, create_graph=True)[0]
        with torch.no_grad():
            f, g = self.dynamics_model.dynamics_predictor(latent)
            g = g.view(-1, latent_dim, action_dim)
            gu = torch.einsum('bij,bi->bj', g, action)
            x_dot = f + gu
        h_dot = torch.einsum('bi,bi->b', grad_B, x_dot).unsqueeze(1)
        return B, h_dot, grad_B

    def train_step(self, batch):
        current_latent = batch['current_latent'].to(self.device)
        action = batch['action'].to(self.device)
        ground_truth = batch['ground_truth'].to(self.device)
        done = batch['done'].to(self.device)

        # Filter out transitions after done flags
        valid_mask = ~done
        current_latent = current_latent[valid_mask]
        action = action[valid_mask]
        ground_truth = ground_truth[valid_mask]

        if current_latent.size(0) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        B, h_dot, grad_B = self.compute_h_dot(current_latent, action)
        safe_mask, unsafe_mask = self.get_mask(ground_truth)

        # Safety losses
        loss_h_safe = torch.relu(eps_safe - B[safe_mask]).mean() * safe_loss_weight
        loss_h_unsafe = torch.relu(B[unsafe_mask] + eps_unsafe).mean() * unsafe_loss_weight

        # Derivative loss
        deriv_cond = h_dot + B
        loss_deriv = torch.relu(-deriv_cond[safe_mask]).mean()

        # CQL loss for actions
        if safe_mask.any():
            safe_latent = current_latent[safe_mask]
            random_actions = torch.rand(safe_latent.size(0), num_action_samples, action_dim, device=device) * 6 - 3
            f, g = self.dynamics_model.dynamics_predictor(safe_latent.unsqueeze(1).repeat(1, num_action_samples, 1).view(-1, latent_dim))
            g = g.view(-1, latent_dim, action_dim)
            gu = torch.einsum('bij,bik->bj', g, random_actions.view(-1, action_dim))
            x_dot_cql = f + gu
            next_latent_cql = safe_latent.unsqueeze(1) + x_dot_cql.view(safe_latent.size(0), num_action_samples, latent_dim) * 0.1
            B_cql = self.cbf(next_latent_cql.view(-1, latent_dim)).view(safe_latent.size(0), num_action_samples)
            log_sum_exp = torch.logsumexp(B_cql / temp, dim=1)
            cql_loss = (log_sum_exp - B[safe_mask].squeeze()).mean() * cql_actions_weight
        else:
            cql_loss = torch.tensor(0.0, device=device)

        total_loss = loss_h_safe + loss_h_unsafe + loss_deriv + cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), loss_h_safe.item(), loss_h_unsafe.item(), loss_deriv.item(), cql_loss.item()

def precompute_latent_states(dynamics_model, dataset):
    dynamics_model.eval()
    latent_states = []
    ground_truths = []
    actions = []
    dones = []

    with torch.no_grad():
        for seq in dataset:
            seq_latent = []
            z_prev = torch.zeros(1, latent_dim).to(device)
            for t in range(len(seq['states_rgb'])):
                img = seq['states_rgb'][t].unsqueeze(0).to(device)
                action = seq['actions'][t-1].unsqueeze(0).to(device) if t > 0 else torch.zeros(1, action_dim).to(device)
                z_current = dynamics_model.encoder(img, z_prev, action)
                seq_latent.append(z_current.cpu())
                z_prev = z_current
            latent_states.extend(seq_latent)
            ground_truths.extend(seq['ground_truth_states'].numpy())
            actions.extend(seq['actions'].numpy())
            dones.extend(seq['dones'].numpy())
    
    return (
        torch.cat(latent_states),
        torch.FloatTensor(np.array(actions)),
        torch.FloatTensor(np.array(ground_truths)),
        torch.FloatTensor(np.array(dones))
    )

def main():
    # Load dataset
    data = np.load("safe_rl_dataset_images.npz")
    full_dataset = DynamicsDataset(data)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Train dynamics model
    dynamics_model = DeterministicDynamicsModel(input_channels=3, latent_dim=latent_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-4)
    
    # Train dynamics model (simplified, adjust epochs as needed)
    for epoch in range(10):  # Example epoch count
        for batch in DataLoader(train_dataset, batch_size=32, shuffle=True):
            loss = compute_multiple_shooting_loss(dynamics_model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Precompute latent states
    latent_states, actions, ground_truths, dones = precompute_latent_states(dynamics_model, full_dataset)
    transition_dataset = TransitionDataset(latent_states, actions, ground_truths, dones)
    train_loader = DataLoader(transition_dataset, batch_size=batch_size, shuffle=False)

    # Initialize CBF and trainer
    cbf = CBF(state_dim=latent_dim, hidden_dim=cbf_hidden_dim).to(device)
    trainer = CBFTrainer(cbf, dynamics_model, train_loader, device)
    
    # Train CBF
    wandb.init(project="latent_cbf_training")
    best_loss = float('inf')
    for epoch in range(100):
        total_loss = 0.0
        for batch in train_loader:
            loss, loss_safe, loss_unsafe, loss_deriv, loss_cql = trainer.train_step(batch)
            total_loss += loss
            wandb.log({
                "total_loss": loss,
                "loss_safe": loss_safe,
                "loss_unsafe": loss_unsafe,
                "loss_deriv": loss_deriv,
                "loss_cql": loss_cql
            })
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(cbf.state_dict(), "best_cbf_model.pth")
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()