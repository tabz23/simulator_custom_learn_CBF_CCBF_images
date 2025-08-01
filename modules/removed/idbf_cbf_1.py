import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

class BCTrainer:
    """
    Trainer for Behavioral Cloning model that outputs a multi-modal Gaussian distribution
    over actions conditioned on states.
    """
    def __init__(self, 
                 actor_model,
                 dataset,
                 lr=3e-4,
                 batch_size=256,
                 device="cpu"):
        self.actor = actor_model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
    def train_step(self):
        # Sample data from dataset
        states, actions, _ = self.dataset.sample_data(batch_size=self.batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        
        # Get action distributions from actor
        _, _, log_probs = self.actor(states, actions)
        
        # Maximize log probability of actions
        loss = -log_probs.mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_iterations=10000):
        """Train the behavioral cloning model"""
        print("Training Behavioral Cloning model...")
        losses = []
        
        for i in range(num_iterations):
            loss = self.train_step()
            losses.append(loss)
            
            if (i+1) % 1000 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss:.4f}")
                
        return losses
    
    def evaluate_density(self, states, actions):
        """
        Evaluate the log probability density of actions given states
        
        Args:
            states: [B, state_dim] tensor of states
            actions: [B, action_dim] tensor of actions
            
        Returns:
            log_probs: [B] tensor of log probabilities
        """
        with torch.no_grad():
            _, _, log_probs = self.actor(states, actions)
        return log_probs


class ImplicitDiverseBarrierFunctionTrainer(Trainer):
    """
    Extended Trainer class that implements iDBF (Implicit Diverse Barrier Function) approach
    by using a trained BC model to generate contrastive examples.
    """
    def __init__(self,
                 cbf,
                 dataset,
                 bc_model,
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
                 use_cql_actions=False,
                 use_cql_states=False,
                 cql_actions_weight=0.1,
                 cql_states_weight=0.1,
                 num_action_samples=10,
                 num_state_samples=10,
                 state_sample_std=0.1,
                 # iDBF specific parameters
                 num_candidate_actions=20,
                 bc_threshold=-5.0,  # Log probability threshold for unlikely actions
                 contrastive_weight=1.0
                 ):
        super().__init__(
            cbf=cbf,
            dataset=dataset,
            safe_distance=safe_distance,
            eps_safe=eps_safe,
            eps_unsafe=eps_unsafe,
            safe_loss_weight=safe_loss_weight,
            unsafe_loss_weight=unsafe_loss_weight,
            action_loss_weight=action_loss_weight,
            dt=dt,
            device=device,
            batch_size=batch_size,
            opt_iter=opt_iter,
            lr=lr,
            use_cql_actions=use_cql_actions,
            use_cql_states=use_cql_states,
            cql_actions_weight=cql_actions_weight,
            cql_states_weight=cql_states_weight,
            num_action_samples=num_action_samples,
            num_state_samples=num_state_samples,
            state_sample_std=state_sample_std
        )
        
        # iDBF specific attributes
        self.bc_model = bc_model
        self.num_candidate_actions = num_candidate_actions
        self.bc_threshold = bc_threshold
        self.contrastive_weight = contrastive_weight
    
    def generate_contrastive_samples(self, safe_states):
        """
        Generate contrastive samples by:
        1. Sampling random candidate actions for each safe state
        2. Evaluating their density using the BC model
        3. Selecting actions with density below threshold
        4. Forward-propagating these actions to generate unsafe states
        
        Args:
            safe_states: [B, state_dim] tensor of safe states
            
        Returns:
            contrastive_states: [B', state_dim] tensor of contrastive states
            is_unsafe: [B'] boolean tensor indicating which contrastive states are actually unsafe
        """
        batch_size = safe_states.shape[0]
        all_contrastive_states = []
        is_unsafe = []
        
        # For each safe state
        for i in range(batch_size):
            state = safe_states[i:i+1].repeat(self.num_candidate_actions, 1)  # [num_candidate_actions, state_dim]
            
            # Sample random actions
            candidate_actions = self.sample_random_actions(self.num_candidate_actions)  # [num_candidate_actions, action_dim]
            
            # Evaluate log probability under the BC model
            log_probs = self.bc_model.evaluate_density(state, candidate_actions)  # [num_candidate_actions]
            
            # Find actions with density below threshold
            low_density_mask = log_probs < self.bc_threshold
            if torch.sum(low_density_mask) == 0:
                # If no actions are below threshold, take the lowest density ones
                _, indices = torch.topk(log_probs, max(1, self.num_candidate_actions // 5), largest=False)
                low_density_mask = torch.zeros_like(log_probs, dtype=torch.bool)
                low_density_mask[indices] = True
            
            # Get the low-density actions
            low_density_actions = candidate_actions[low_density_mask]  # [num_low_density, action_dim]
            
            if len(low_density_actions) > 0:
                # Forward-propagate to get potential unsafe states
                low_density_states = state[low_density_mask]  # [num_low_density, state_dim]
                contrastive_states = self.compute_next_states(low_density_states, low_density_actions)
                
                # Check which ones are actually unsafe
                _, unsafe_mask = self.get_mask(contrastive_states)  # [num_low_density]
                
                all_contrastive_states.append(contrastive_states)
                is_unsafe.append(unsafe_mask)
        
        # Combine results from all states
        if all_contrastive_states:
            all_contrastive_states = torch.cat(all_contrastive_states, dim=0)
            is_unsafe = torch.cat(is_unsafe, dim=0)
            return all_contrastive_states, is_unsafe
        else:
            # Return empty tensors if no contrastive samples were found
            return torch.zeros((0, safe_states.shape[1]), device=self.device), torch.zeros(0, dtype=torch.bool, device=self.device)
    
    def train_cbf(self):
        """
        Train the CBF with contrastive samples from the iDBF approach
        """
        loss_np = np.zeros(6, dtype=np.float32)  # Now tracking 6 loss components (added contrastive loss)
        acc_np = np.zeros(3, dtype=np.float32)   # Track 3 accuracy metrics
        total_safe_h = 0
        total_unsafe_h = 0
        num_safe_samples = 0
        num_unsafe_samples = 0
        total_contrastive_samples = 0
        
        for i in range(self.opt_iter):
            relu = nn.ReLU()
            
            # Sample data
            state, control, next_state = self.dataset.sample_data(batch_size=self.batch_size)
            state = torch.from_numpy(state).to(self.device)
            control = torch.from_numpy(control).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device)
            
            # Get safe/unsafe masks
            safe_mask, unsafe_mask = self.get_mask(state)
            
            # Compute CBF values for all states at once
            h = self.cbf(state)
            h_dot = self.cbf.compute_h_dot(state, next_state)
            
            # Count samples
            num_safe = torch.sum(safe_mask).float()
            num_unsafe = torch.sum(unsafe_mask).float()
            
            # Compute losses directly with masking
            # Safe states losses
            loss_h_safe = self.safe_loss_weight * torch.sum(relu(self.eps_safe - h).reshape(-1, 1) * safe_mask.reshape(-1, 1)) / (num_safe + 1e-5)
            
            deriv_cond = h_dot + h
            loss_deriv_safe = torch.sum(relu(-deriv_cond).reshape(-1, 1) * safe_mask.reshape(-1, 1)) / (num_safe + 1e-5)
            
            # Unsafe states losses
            loss_h_unsafe = self.unsafe_loss_weight * torch.sum(relu(self.eps_unsafe + h).reshape(-1, 1) * unsafe_mask.reshape(-1, 1)) / (num_unsafe + 1e-5)
            
            # Calculate accuracies
            acc_h_safe = torch.sum((h >= 0).reshape(-1, 1) * safe_mask.reshape(-1, 1)) / (num_safe + 1e-5)
            acc_h_unsafe = torch.sum((h < 0).reshape(-1, 1) * unsafe_mask.reshape(-1, 1)) / (num_unsafe + 1e-5)
            acc_deriv_safe = torch.sum((deriv_cond > 0).reshape(-1, 1) * safe_mask.reshape(-1, 1)) / (num_safe + 1e-5)
            
            # Compute CQL-inspired losses
            loss_cql_actions = torch.tensor(0.0, device=self.device)
            loss_cql_states = torch.tensor(0.0, device=self.device)
            
            if self.use_cql_actions and num_safe > 0:
                safe_states = state[safe_mask]
                actual_next_states = next_state[safe_mask]
                actual_next_h = self.cbf(actual_next_states)
                
                all_random_next_h = []
                for _ in range(self.num_action_samples):
                    random_actions = self.sample_random_actions(safe_states.shape[0])
                    random_next_states = self.compute_next_states(safe_states, random_actions)
                    random_next_h = self.cbf(random_next_states)
                    all_random_next_h.append(random_next_h.squeeze())
                
                if all_random_next_h:
                    stacked_h_values = torch.stack(all_random_next_h, dim=1)
                    combined_h_values = torch.cat([stacked_h_values, actual_next_h.squeeze().unsqueeze(1)], dim=1)
                    
                    temp = 0.7
                    
                    logsumexp_h = temp * torch.logsumexp(combined_h_values/temp, dim=1)
                    
                    cql_actions_term = logsumexp_h - actual_next_h.squeeze()
                    loss_cql_actions = self.cql_actions_weight * torch.mean(cql_actions_term)
            
            if self.use_cql_states and num_safe > 0:
                safe_states = state[safe_mask]
                in_dist_h = self.cbf(safe_states)
                
                all_nearby_h = []
                for _ in range(self.num_state_samples):
                    nearby_states = self.sample_nearby_states(safe_states, std=self.state_sample_std)
                    nearby_h = self.cbf(nearby_states)
                    all_nearby_h.append(nearby_h)
                
                all_nearby_h_values = torch.stack(all_nearby_h, dim=1)
                logsumexp_nearby_h = torch.logsumexp(all_nearby_h_values, dim=1, keepdim=True)
                
                loss_cql_states = torch.mean(logsumexp_nearby_h - in_dist_h)
                loss_cql_states = self.cql_states_weight * loss_cql_states
            
            # Generate contrastive samples using the BC model
            loss_contrastive = torch.tensor(0.0, device=self.device)
            if num_safe > 0:
                safe_states = state[safe_mask]
                contrastive_states, contrastive_unsafe_mask = self.generate_contrastive_samples(safe_states)
                
                if contrastive_states.shape[0] > 0:
                    # Compute CBF values for contrastive samples
                    contrastive_h = self.cbf(contrastive_states)
                    
                    # Contrastive loss - we want h to be negative for actually unsafe states
                    contrastive_loss = torch.sum(
                        relu(self.eps_unsafe + contrastive_h).reshape(-1, 1) * 
                        contrastive_unsafe_mask.reshape(-1, 1)
                    ) / (torch.sum(contrastive_unsafe_mask) + 1e-5)
                    
                    loss_contrastive = self.contrastive_weight * contrastive_loss
                    total_contrastive_samples += torch.sum(contrastive_unsafe_mask).item()
            
            # Total loss
            loss = loss_h_safe + loss_h_unsafe + loss_deriv_safe + loss_cql_actions + loss_cql_states + loss_contrastive
            
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
            loss_np[5] += loss_contrastive.detach().cpu().numpy()
            
            # Accumulate h values
            total_safe_h += torch.sum(h.reshape(-1, 1) * safe_mask.reshape(-1, 1)).detach().cpu().numpy()
            total_unsafe_h += torch.sum(h.reshape(-1, 1) * unsafe_mask.reshape(-1, 1)).detach().cpu().numpy()
            
            num_safe_samples += num_safe.detach().cpu().numpy()
            num_unsafe_samples += num_unsafe.detach().cpu().numpy()
        
        # Calculate averages
        avg_safe_h = total_safe_h / (num_safe_samples + 1e-5)
        avg_unsafe_h = total_unsafe_h / (num_unsafe_samples + 1e-5)
        
        return acc_np / self.opt_iter, loss_np / self.opt_iter, avg_safe_h, avg_unsafe_h, total_contrastive_samples


# Function to initialize and train the BC model
def train_bc_model(dataset, obs_dim, act_dim, action_low, action_high, device="cpu", num_iterations=5000):
    """
    Initialize and train a BC model on the given dataset
    
    Args:
        dataset: Dataset object containing state-action pairs
        obs_dim: Dimension of the observation space
        act_dim: Dimension of the action space
        action_low: Lower bound of actions
        action_high: Upper bound of actions
        device: Device to use for training
        num_iterations: Number of training iterations
        
    Returns:
        Trained BC model
    """
    # Initialize BC model
    bc_model = MLPGaussianActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=[256, 256],
        activation=nn.ReLU,
        device=device
    ).to(device)
    
    # Initialize trainer
    bc_trainer = BCTrainer(
        actor_model=bc_model,
        dataset=dataset,
        lr=3e-4,
        batch_size=256,
        device=device
    )
    
    # Train model
    bc_trainer.train(num_iterations=num_iterations)
    
    return bc_model