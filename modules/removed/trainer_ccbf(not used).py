import torch
from torch import nn
import numpy as np

class Trainer():
    def __init__(self,
                 cbf,
                 dataset,
                 idbf_data=False,
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
                 state_sample_std=0.1,
                 temp=0.7,
                 detach=False
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
        self.temp=temp
        self.detach=detach
        
        self.idbf_data=idbf_data
        
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
        first_dim = 6 * torch.rand(batch_size, 1, device=self.device) - 3  # Uniform in [-3,3]
        second_dim = 6 * torch.rand(batch_size, 1, device=self.device) - 3  # Uniform in [-3,3]
        return torch.cat([first_dim, second_dim], dim=1)
    
    def compute_next_states(self, state, action):
        next_state = state.clone()
        next_state[:, :2] = next_state[:, :2] + action* self.dt
        return next_state
    
    def get_mask(self, state):
        dist = torch.norm(state[:,:2] - state[:,2:4], dim=1)
        safe_mask = dist > self.safe_distance
        unsafe_mask = dist <= self.safe_distance
        return safe_mask, unsafe_mask
    
    def train_cbf(self):
        loss_np = np.zeros(5, dtype=np.float32)  # Track 5 loss components
        acc_np = np.zeros(3, dtype=np.float32)   # Track 3 accuracy metrics
        total_safe_h = 0
        total_unsafe_h = 0
        num_safe_samples = 0
        num_unsafe_samples = 0
        
        for i in range(self.opt_iter):
            relu = nn.ReLU()
            
            # Sample data
            state, control, next_state = self.dataset.sample_data(batch_size=self.batch_size)
            state = torch.from_numpy(state).to(self.device)
            control = torch.from_numpy(control).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device)
            
            # Get safe/unsafe masks
            if not self.idbf_data:
                safe_mask, unsafe_mask = self.get_mask(state)##ACTUALLY THIS IS USELESS REMOVE IT LATER
                
            else:##if idbf training then just use the fact that dataset already creates first 32 samples as safe and second 32 as unsafe assuming there are more than batchsize/2 unsafe samples in dataset buffer
                safe_mask = np.concatenate([np.full(self.batch_size// 2, True, dtype=bool), 
                                        np.full(self.batch_size // 2, False, dtype=bool)])
                unsafe_mask = np.concatenate([np.full(self.batch_size// 2, False, dtype=bool), 
                        np.full(self.batch_size // 2, True, dtype=bool)])
                safe_mask = torch.tensor(safe_mask, dtype=torch.bool).to(self.device)
                unsafe_mask = torch.tensor(unsafe_mask, dtype=torch.bool).to(self.device)

            # Compute CBF values for all states at once
            h = self.cbf(state)
            h_dot = self.cbf.compute_h_dot(state, next_state)
            # print("curr h top: ", h)
            # print("h safe",self.cbf(state[safe_mask]))
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
            # print("current h",h)
            # print("next h",self.cbf(next_state))
            # Calculate accuracies
            acc_h_safe = torch.sum((h >= 0).reshape(-1, 1) * safe_mask.reshape(-1, 1)) / (num_safe + 1e-5)
            acc_h_unsafe = torch.sum((h < 0).reshape(-1, 1) * unsafe_mask.reshape(-1, 1)) / (num_unsafe + 1e-5)
            acc_deriv_safe = torch.sum((deriv_cond > 0).reshape(-1, 1) * safe_mask.reshape(-1, 1)) / (num_safe + 1e-5)
            
            # Compute CQL-inspired losses
            loss_cql_actions = torch.tensor(0.0, device=self.device)
            loss_cql_states = torch.tensor(0.0, device=self.device)
            
            if not self.idbf_data:
                if self.use_cql_actions and num_safe > 0:
                    safe_states = state[safe_mask]
                    # print("state",state)
                    # print("state[safe_mask]",state[safe_mask])
                    actual_next_states = next_state[safe_mask]
                    actual_next_h = self.cbf(actual_next_states)
                    # print("curr h below :", self.cbf(safe_states))
                    # print("next state",actual_next_h)
                    
                    all_random_next_h = []
                    for _ in range(self.num_action_samples):
                        random_actions = self.sample_random_actions(safe_states.shape[0])
                        random_next_states = self.compute_next_states(safe_states, random_actions)
                        random_next_h = self.cbf(random_next_states)
                        all_random_next_h.append(random_next_h.squeeze())
                    
                    if all_random_next_h:
                        stacked_h_values = torch.stack(all_random_next_h, dim=1)
                        combined_h_values = torch.cat([stacked_h_values, actual_next_h.squeeze().unsqueeze(1)], dim=1)

                        logsumexp_h = self.temp * torch.logsumexp(combined_h_values/self.temp, dim=1)
                        # print(logsumexp_h[0:10])
                        # print(actual_next_h[0:10])
                        
                        if self.detach:
                            cql_actions_term = logsumexp_h - actual_next_h.squeeze().detach()
                        else:
                            cql_actions_term = logsumexp_h - actual_next_h.squeeze()#.detach()###I DETACHED THE SECOND TERM. HOPEFULY PROMOTES LESS POSITIVE AVERAGE CBF VALUES SINCE SECOND TERM IF IT INCREAASES, LOSS DECREASES. BUT FIRST TERM (AVG CBF) IF DECREASES, LOSS DECREASES. GOAL IS TO SURELY REDUCE AVG SAFE CBF VALUE AND BE MORE CONSERVATIVE
                      
                        loss_cql_actions = self.cql_actions_weight * torch.mean(cql_actions_term)

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
            total_safe_h += torch.sum(h.reshape(-1, 1) * safe_mask.reshape(-1, 1)).detach().cpu().numpy()
            total_unsafe_h += torch.sum(h.reshape(-1, 1) * unsafe_mask.reshape(-1, 1)).detach().cpu().numpy()
            
            num_safe_samples += num_safe.detach().cpu().numpy()
            num_unsafe_samples += num_unsafe.detach().cpu().numpy()
        
        # Calculate averages
        avg_safe_h = total_safe_h / (num_safe_samples + 1e-5)
        avg_unsafe_h = total_unsafe_h / (num_unsafe_samples + 1e-5)
        
        return acc_np / self.opt_iter, loss_np / self.opt_iter, avg_safe_h, avg_unsafe_h