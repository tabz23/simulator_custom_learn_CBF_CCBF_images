import torch
from torch import nn
import numpy as np

class Trainer():
    def __init__(self,
                 cbf,
                 dataset,
                 safe_distance=4,
                 eps_safe=0.02,
                 eps_unsafe=0.02,
                 safe_loss_weight=1,
                 unsafe_loss_weight=1.5,
                 action_loss_weight=1,
                 dt=0.1,
                 device="cpu",
                 batch_size=64,
                 opt_iter=10,
                 lr=1e-4,
                 use_cql_actions=True,
                 use_cql_states=True,
                 cql_actions_weight=0.1,
                 cql_states_weight=0.1,
                 num_action_samples=10,
                 num_state_samples=10,
                 state_sample_std=0.1
                 ):
        self.cbf = cbf
        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.opt_iter = opt_iter
        self.eps_safe = eps_safe
        self.eps_unsafe = eps_unsafe
        self.safe_distance = safe_distance
        self.safe_loss_weight = safe_loss_weight
        self.unsafe_loss_weight = unsafe_loss_weight
        self.action_loss_weight = action_loss_weight
        self.dt = dt
        self.device = device
        
        # CQL-inspired loss parameters
        self.use_cql_actions = use_cql_actions
        self.use_cql_states = use_cql_states
        self.cql_actions_weight = cql_actions_weight
        self.cql_states_weight = cql_states_weight
        self.num_action_samples = num_action_samples
        self.num_state_samples = num_state_samples
        self.state_sample_std = state_sample_std
        
        self.cbf_optimizer = torch.optim.Adam(self.cbf.parameters(), lr=self.lr, weight_decay=5e-5)
    
    def sample_random_actions(self, batch_size):
        first_dim = 3 * torch.rand(batch_size, 1, device=self.device)  # Uniform in [0,2]
        second_dim = 6 * torch.rand(batch_size, 1, device=self.device) - 3  # Uniform in [-3,3]
        return torch.cat([first_dim, second_dim], dim=1)
    
    def sample_nearby_states(self, state, std=1):
        # Sample states in the vicinity of the current state
        noise = torch.randn_like(state) * std
        return state + noise
    
    def compute_next_states(self, state, action):
        next_state = state.clone()
        next_state[:, :2] = next_state[:, :2] + action * self.dt
        return next_state
    
    def get_mask(self, state):
        dist = torch.norm(state[:,:2] - state[:,2:4], dim=1)
        safe_mask = dist > self.safe_distance
        unsafe_mask = dist <= self.safe_distance
        return safe_mask, unsafe_mask
    
    def compute_safe_cbf_loss(self, safe_states, safe_next_states):
        """Calculate losses for safe states"""
        relu = nn.ReLU()
        
        # Compute CBF values
        h_safe = self.cbf(safe_states)
        # print(h_safe)
        h_dot_safe = self.cbf.compute_h_dot(safe_states, safe_next_states)
        deriv_cond_safe = h_dot_safe + h_safe
        
        # Calculate losses
        loss_h_safe = self.safe_loss_weight * torch.mean(relu(self.eps_safe - h_safe))
        loss_deriv_safe = torch.mean(relu(-deriv_cond_safe))
        
        # Calculate accuracies
        acc_h_safe = torch.mean((h_safe >= 0).float())
        acc_deriv_safe = torch.mean((deriv_cond_safe > 0).float())
        
        return h_safe, loss_h_safe, loss_deriv_safe, acc_h_safe, acc_deriv_safe
    
    def compute_unsafe_cbf_loss(self, unsafe_states):
        """Calculate losses for unsafe states"""
        relu = nn.ReLU()
        
        # Compute CBF values
        h_unsafe = self.cbf(unsafe_states)
        print(h_unsafe)
        
        # Calculate loss
        loss_h_unsafe = self.unsafe_loss_weight * torch.mean(relu(self.eps_unsafe + h_unsafe))
        
        # Calculate accuracy
        acc_h_unsafe = torch.mean((h_unsafe < 0).float())
        
        return h_unsafe, loss_h_unsafe, acc_h_unsafe
    
    def compute_cql_actions_loss(self, safe_states, safe_next_states):
        """Calculate CQL-inspired losses for random actions"""
        if safe_states.shape[0] == 0:  # No safe states in batch
            return torch.tensor(0.0, device=self.device)
        
        # Compute h value for the actual next state
        actual_next_h = self.cbf(safe_next_states)
        
        # Sample random actions and compute resulting next states
        all_random_next_h = []
        for _ in range(self.num_action_samples):
            random_actions = self.sample_random_actions(safe_states.shape[0])
            random_next_states = self.compute_next_states(safe_states, random_actions)
            random_next_h = self.cbf(random_next_states)
            all_random_next_h.append(random_next_h.squeeze())
        
        # Stack all h values
        if all_random_next_h:
            stacked_h_values = torch.stack(all_random_next_h, dim=1)
            combined_h_values = torch.cat([stacked_h_values, actual_next_h.squeeze().unsqueeze(1)], dim=1)
            
            # Compute logsumexp over the action dimension
            temp = 0.5
            logsumexp_h = temp * torch.logsumexp(combined_h_values/temp, dim=1)
            
            # CQL loss: logsumexp(h(s'_a)) - h(s'_i)
            cql_actions_term = logsumexp_h - actual_next_h.squeeze()
            loss_cql_actions = self.cql_actions_weight * torch.mean(cql_actions_term)
            return loss_cql_actions
        
        return torch.tensor(0.0, device=self.device)
    
    def compute_cql_states_loss(self, safe_states):
        """Calculate CQL-inspired losses for nearby states"""
        if safe_states.shape[0] == 0:  # No safe states in batch
            return torch.tensor(0.0, device=self.device)
            
        in_dist_h = self.cbf(safe_states)
        
        all_nearby_h = []
        for _ in range(self.num_state_samples):
            nearby_states = self.sample_nearby_states(safe_states, std=self.state_sample_std)
            nearby_h = self.cbf(nearby_states)
            all_nearby_h.append(nearby_h)
        
        # Stack all nearby h values
        all_nearby_h_values = torch.stack(all_nearby_h, dim=1)
        
        # Compute logsumexp over the sampled states
        logsumexp_nearby_h = torch.logsumexp(all_nearby_h_values, dim=1, keepdim=True)
        
        # CQL states loss: logsumexp(h(s_nearby)) - h(s_i)
        loss_cql_states = torch.mean(logsumexp_nearby_h - in_dist_h)
        loss_cql_states = self.cql_states_weight * loss_cql_states
        return loss_cql_states
    
    def train_cbf(self):
        loss_np = np.zeros(5, dtype=np.float32)  # Track 5 loss components
        acc_np = np.zeros(3, dtype=np.float32)   # Track 3 accuracy metrics
        total_safe_h = 0
        total_unsafe_h = 0
        num_safe_samples = 0
        num_unsafe_samples = 0
        
        for i in range(self.opt_iter):
            # Sample data
            state, control, next_state = self.dataset.sample_data(batch_size=self.batch_size)
            state = torch.from_numpy(state).to(self.device)
            control = torch.from_numpy(control).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device)
            
            # Get safe/unsafe masks
            safe_mask, unsafe_mask = self.get_mask(state)
            
            # Split data into safe and unsafe batches
            safe_states = state[safe_mask]
            unsafe_states = state[unsafe_mask]
            safe_next_states = next_state[safe_mask]
            
            # Count samples
            num_safe = safe_mask.sum().float()
            num_unsafe = unsafe_mask.sum().float()
            
            # Compute losses for safe states
            h_safe, loss_h_safe, loss_deriv_safe, acc_h_safe, acc_deriv_safe = self.compute_safe_cbf_loss(
                safe_states, safe_next_states)
            
            # Compute losses for unsafe states
            h_unsafe, loss_h_unsafe, acc_h_unsafe = self.compute_unsafe_cbf_loss(unsafe_states)
            
            # Compute CQL-inspired losses
            loss_cql_actions = torch.tensor(0.0, device=self.device)
            loss_cql_states = torch.tensor(0.0, device=self.device)
            
            if self.use_cql_actions and num_safe > 0:
                loss_cql_actions = self.compute_cql_actions_loss(safe_states, safe_next_states)
            
            if self.use_cql_states and num_safe > 0:
                loss_cql_states = self.compute_cql_states_loss(safe_states)
            
            # Total loss
            loss = loss_h_safe + loss_h_unsafe + loss_deriv_safe + loss_cql_actions + loss_cql_states
            
            # Backpropagation
            self.cbf_optimizer.zero_grad()
            loss.backward()
            self.cbf_optimizer.step()
            
            # Update metrics
            acc_np[0] += acc_h_safe.detach().cpu().numpy()
            acc_np[1] += acc_h_unsafe.detach().cpu().numpy()
            acc_np[2] += acc_deriv_safe.detach().cpu().numpy()
            
            loss_np[0] += loss_h_safe.detach().cpu().numpy()
            loss_np[1] += loss_h_unsafe.detach().cpu().numpy()
            loss_np[2] += loss_deriv_safe.detach().cpu().numpy()
            loss_np[3] += loss_cql_actions.detach().cpu().numpy()
            loss_np[4] += loss_cql_states.detach().cpu().numpy()
            
            # Accumulate h values
            if num_safe > 0:
                total_safe_h += torch.sum(h_safe).detach().cpu().numpy()
            if num_unsafe > 0:
                total_unsafe_h += torch.sum(h_unsafe).detach().cpu().numpy()
            
            num_safe_samples += num_safe.detach().cpu().numpy()
            num_unsafe_samples += num_unsafe.detach().cpu().numpy()
        
        # Calculate averages
        avg_safe_h = total_safe_h / (num_safe_samples + 1e-5)
        avg_unsafe_h = total_unsafe_h / (num_unsafe_samples + 1e-5)
        
        return acc_np / self.opt_iter, loss_np / self.opt_iter, avg_safe_h, avg_unsafe_h