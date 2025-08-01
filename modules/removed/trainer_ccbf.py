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
    
    # def sample_random_actions(self, batch_size):
    #     # torch.rand(batch_size, 2, device=self.device) generates values in [0,1)
    #     # functions return [0,4]-[2,2]=[-2,2]
    #     return 4 * torch.rand(batch_size, 2, device=self.device) - 2
    
    def sample_random_actions(self, batch_size):
        first_dim = 3 * torch.rand(batch_size, 1, device=self.device)  # Uniform in [0,2]
        second_dim = 6 * torch.rand(batch_size, 1, device=self.device) - 3  # Uniform in [-3,3]
        return torch.cat([first_dim, second_dim], dim=1)



    # def sample_nearby_states(self, state, std=1):
    #     # Sample states in the vicinity of the current state
    #     #torch.randn_like(state) draws from uniform distribution N(0,1)
    #     noise = torch.randn_like(state) * std
    #     return state + noise
    
    def compute_next_states(self, state, action):
        next_state = state.clone()
        next_state[:, :2] = next_state[:, :2] + action * self.dt
        return next_state
    
    def train_cbf(self):
        loss_np = np.zeros(5, dtype=np.float32)  # Updated to include CQL losses
        acc_np = np.zeros(3, dtype=np.float32)
        total_safe_h = 0
        total_unsafe_h = 0
        num_safe_samples = 0
        num_unsafe_samples = 0

        
        for i in range(self.opt_iter):
            relu = nn.ReLU()
            state, control, next_state = self.dataset.sample_data(batch_size=self.batch_size)
            state = torch.from_numpy(state).to(self.device)
            control = torch.from_numpy(control).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device)
            
            safe_mask, unsafe_mask = self.get_mask(state)
            # print("cbf curr state",self.cbf(state[safe_mask]))
            h = self.cbf(state)
            # print(h)
            # print("p")
            # print(h>0)
            # print("p")
            # print(safe_mask)
            
            # print(state[0:5])
            # print(state[safe_mask][0:5])
            # print(self.cbf(state[0:5]))
            # print(self.cbf(state[safe_mask][0:5])) 
            # print((-self.cbf(state[0:5])).reshape(-1,1))
            # print(safe_mask.reshape(-1,1)[0:5])
            # print((-self.cbf(state[0:5])).reshape(-1,1) * safe_mask.reshape(-1,1)[0:5])
            # print(safe_mask)
            # print("1",torch.sum(relu(-self.cbf(state[0:64])).reshape(-1,1) * safe_mask.reshape(-1,1)[0:64]))
            # print(state.shape)
            # print(state[safe_mask].shape)
            
            # print(self.cbf(state))
            # print(self.cbf(state[safe_mask]))
            
            h_dot = self.cbf.compute_h_dot(state, next_state)
            
            num_safe = torch.sum(safe_mask).float()
            num_unsafe = torch.sum(unsafe_mask).float()
            
            
            # Apply safe_loss_weight to safe loss
            # print("2",torch.sum(relu(-h).reshape(-1,1) * safe_mask.reshape(-1,1)) )
            loss_h_safe = self.safe_loss_weight * torch.sum(relu(self.eps_safe-h).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe+1e-5)
            loss_h_unsafe = self.unsafe_loss_weight * torch.sum(relu(self.eps_unsafe+h).reshape(-1,1) * unsafe_mask.reshape(-1,1)) / (num_unsafe+1e-5)

            # Normalize accuracies to be between 0 and 1
            acc_h_safe = torch.sum((h >= 0).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe + 1e-5)
            # print( torch.sum((h >= 0).reshape(-1,1) * safe_mask.reshape(-1,1)) )
            # print(h)
            # print((h >= 0).reshape(-1,1))
            # print(safe_mask.reshape(-1,1))
            # print((h < 0).reshape(-1,1) * unsafe_mask.reshape(-1,1))
            acc_h_unsafe = torch.sum((h < 0).reshape(-1,1) * unsafe_mask.reshape(-1,1)) / (num_unsafe + 1e-5)
            
            deriv_cond = h_dot + h
            loss_deriv_safe = torch.sum(relu(-deriv_cond).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe + 1e-5)
            acc_deriv_safe = torch.sum((deriv_cond > 0).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe + 1e-5)
            
            # Calculate CQL-inspired losses
            loss_cql_actions = torch.tensor(0.0, device=self.device)
            loss_cql_states = torch.tensor(0.0, device=self.device)
            
            
             #all_log_sum_exp.append(actual_next_h)##add the actual cbf value of the next state for computing the logsumexp
            # if self.use_cql_actions and num_safe > 0:
            #     # L_CQL_actions: penalize OOD next states from random actions
            #     safe_states = state[safe_mask]
            #     print(f"safe_states shape: {safe_states.shape}")  # Print shape of safe states
                
            #     # Compute h value for the actual next state
            #     actual_next_states = next_state[safe_mask]
            #     print(f"actual_next_states shape: {actual_next_states.shape}")  # Print shape of actual next states
                
            #     actual_next_h = self.cbf(actual_next_states)
            #     print(f"actual_next_h shape: {actual_next_h.shape}")  # Print shape of CBF values for actual next states
                
            #     # Sample random actions and compute resulting next states
            #     all_log_sum_exp = []
            #     for _ in range(self.num_action_samples):
            #         random_actions = self.sample_random_actions(safe_states.shape[0])  # batch_size, 2
            #         print(f"random_actions shape: {random_actions.shape}")  # Print shape of sampled random actions
                    
            #         random_next_states = self.compute_next_states(safe_states, random_actions)  # batch_size, 4
            #         print(f"random_next_states shape: {random_next_states.shape}")  # Print shape of computed next states
                    
            #         random_next_h = self.cbf(random_next_states)  # batch_size, 1
            #         print(f"random_next_h shape: {random_next_h.shape}")  # Print shape of CBF values for random next states
                    
            #         all_log_sum_exp.append(random_next_h)  # Add batch_size -> all_log_sum_exp will be a list of batch_size, 1
                    
            #     # After we exit, all_log_sum_exp will have num_action_samples elements in the list, each of size batch_size, 1
            #     print(f"all_log_sum_exp length: {len(all_log_sum_exp)}")  # Print the length (number of action samples)
                
            #     # Stack all h values [batch_size, num_samples, 1]
            #     all_h_values = torch.stack(all_log_sum_exp, dim=1)
            #     print(f"all_h_values shape: {all_h_values.shape}")  # Print the shape of the stacked tensor
                
            #     # Compute logsumexp over the action dimension
            #     logsumexp_h = torch.logsumexp(all_h_values, dim=1, keepdim=True)
            #     print(f"logsumexp_h shape: {logsumexp_h.shape}")  # Print shape after logsumexp
                
            #     # CQL loss: logsumexp(h(s'_a)) - h(s'_i)
            #     loss_cql_actions = torch.mean(logsumexp_h - actual_next_h)
            #     print(f"loss_cql_actions shape: {loss_cql_actions.shape}")  # Print shape of the computed CQL loss
                
            #     loss_cql_actions = self.cql_actions_weight * loss_cql_actions
            #     print(f"loss_cql_actions after scaling shape: {loss_cql_actions.shape}")  # Print shape after applying the weight
            #     '''
            #     safe_states shape: torch.Size([64, 4])
            #     actual_next_states shape: torch.Size([64, 4])
            #     `actual_next_h` shape: torch.Size([64, 1])
            #     random_actions shape: torch.Size([64, 2])
            #     random_next_states shape: torch.Size([64, 4])
            #     random_next_h shape: torch.Size([64, 1])
            #     random_actions shape: torch.Size([64, 2])
            #     random_next_states shape: torch.Size([64, 4])
            #     random_next_h shape: torch.Size([64, 1])
            #     ...
            #     all_log_sum_exp length: 10
            #     all_h_values shape: torch.Size([64, 10, 1])
            #     logsumexp_h shape: torch.Size([64, 1, 1])
            #     loss_cql_actions shape: torch.Size([])
            #     loss_cql_actions after scaling shape: torch.Size([])
            #     '''
            
            
            
            if self.use_cql_actions and num_safe > 0:
                # L_CQL_actions: penalize OOD next states from random actions
                # dist = torch.norm(state[:,:2] - state[:,2:4], dim=1)
                # print(dist)
                # safe_mask_current = dist > self.safe_distance
                # dist = torch.norm(next_state[:,:2] - next_state[:,2:4], dim=1)
                # # print(dist)
                # safe_mask_next = dist > self.safe_distance
                # safe_mask=safe_mask_current & safe_mask_next ##elementwise logical and
                # print("state.shape",state.shape)
                safe_states = state[safe_mask]
                # print("safe_mask.shape",safe_mask.shape)
                # print("safe_states.shape",safe_states.shape)
                # print("state[safe_mask].shape",state[safe_mask].shape)
                # print(safe_mask)
                # print(safe_states)#correct
                # print(state)#correct
                
                # print(safe_mask_next)
                # print(safe_mask_current)
                
                # print(f"safe_states shape: {safe_states.shape}")  # Print shape of safe states
                
                # Compute h value for the actual next state
                actual_next_states = next_state[safe_mask]
                # print(torch.norm(actual_next_states[:,:2] - actual_next_states[:,2:4], dim=1)[0:15])
                # print(f"actual_next_states shape: {actual_next_states.shape}")  # Print shape of actual next states
                actual_next_h = self.cbf(actual_next_states)
             #   print("cbf next state", actual_next_h[0:15])
                
            #    print(torch.norm(safe_states[:,:2] - safe_states[:,2:4], dim=1)[0:15])
         ####       print("cbf curr state",self.cbf(safe_states))
                # print(f"actual_next_h shape: {actual_next_h.shape}")  # Print shape of CBF values for actual next states
                
                # Sample random actions and compute resulting next states
                all_random_next_h = []
                for _ in range(self.num_action_samples):
                    random_actions = self.sample_random_actions(safe_states.shape[0])  # [num_safe, 2]
                    # print(f"random_actions shape: {random_actions.shape}")  # Print shape of sampled random actions
                    
                    random_next_states = self.compute_next_states(safe_states, random_actions)  # [num_safe, 4]
                    # print(f"random_next_states shape: {random_next_states.shape}")  # Print shape of computed next states
                    
                    random_next_h = self.cbf(random_next_states)  # [num_safe, 1]
                    # print(f"random_next_h shape: {random_next_h.shape}")  # Print shape of CBF values for random next states
                    
                    # Ensure proper shape for aggregation
                    all_random_next_h.append(random_next_h.squeeze())  # Remove the last dimension to get [num_safe]
                    # print(f"random_next_h squeezed shape: {random_next_h.squeeze().shape}")  # Print shape after squeezing
                
                # Stack all h values to get [num_safe, num_samples]
                if all_random_next_h:  # Check if the list is not empty
                    stacked_h_values = torch.stack(all_random_next_h, dim=1)  # [num_safe, num_samples]
                    # print(f"stacked_h_values shape: {stacked_h_values.shape}")  # Print shape of stacked tensor
                    
                    # Include the actual next state h value for a proper CQL calculation
                    combined_h_values = torch.cat([stacked_h_values, actual_next_h.squeeze().unsqueeze(1)], dim=1)  # [num_safe, num_samples+1]
                    # print(f"combined_h_values shape: {combined_h_values.shape}")  # Print shape of combined tensor
                    
                    # Compute logsumexp over the action dimension
                    temp=0.5
                    logsumexp_h = temp*torch.logsumexp(combined_h_values/temp, dim=1)  # [num_safe]
                    # print(f"logsumexp_h shape: {logsumexp_h.shape}")  # Print shape of logsumexp
                    
                    # CQL loss: logsumexp(h(s'_a)) - h(s'_i)
                    cql_actions_term = logsumexp_h - actual_next_h.squeeze()  # [num_safe]
                    # print("logsumexp_h",logsumexp_h[:10])
                    # print("actual_next_h", actual_next_h[0:15])
                    
                    
                    
                    # print(f"cql_actions_term shape: {cql_actions_term.shape}")  # Print shape of the CQL actions term
                    ##loss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actionsloss_cql_actions
                    loss_cql_actions = self.cql_actions_weight * torch.mean(cql_actions_term)
                    ##

                    # print(f"loss_cql_actions shape: {loss_cql_actions.shape}")  # Print shape of the CQL loss
                    '''                   
                    safe_states shape: torch.Size([64, 4])
                    actual_next_states shape: torch.Size([64, 4])
                    actual_next_h shape: torch.Size([64, 1])
                    random_actions shape: torch.Size([64, 2])
                    random_next_states shape: torch.Size([64, 4])
                    ...
                    random_next_h shape: torch.Size([64, 1])
                    random_next_h squeezed shape: torch.Size([64])
                    stacked_h_values shape: torch.Size([64, 10])
                    combined_h_values shape: torch.Size([64, 11])
                    logsumexp_h shape: torch.Size([64])
                    cql_actions_term shape: torch.Size([64])
                    loss_cql_actions shape: torch.Size([])
                    '''
                
            if self.use_cql_states and num_safe > 0:
                # L_CQL_states: penalize high CBF values in the vicinity of in-distribution states
                safe_states = state[safe_mask]
                in_dist_h = self.cbf(safe_states)
                
                all_nearby_h = []
                for _ in range(self.num_state_samples):
                    nearby_states = self.sample_nearby_states(safe_states, std=self.state_sample_std)
                    nearby_h = self.cbf(nearby_states)
                    all_nearby_h.append(nearby_h)
                
                # Stack all nearby h values [batch_size, num_samples]
                all_nearby_h_values = torch.stack(all_nearby_h, dim=1)
                
                # Compute logsumexp over the sampled states
                logsumexp_nearby_h = torch.logsumexp(all_nearby_h_values, dim=1, keepdim=True)
                
                # CQL states loss: logsumexp(h(s_nearby)) - h(s_i)
                loss_cql_states = torch.mean(logsumexp_nearby_h - in_dist_h)
                loss_cql_states = self.cql_states_weight * loss_cql_states
            
            # Total loss
            loss = loss_h_safe + loss_h_unsafe + loss_deriv_safe + loss_cql_actions + loss_cql_states
            
            self.cbf_optimizer.zero_grad()
            loss.backward()
            self.cbf_optimizer.step()
            
            # Detach and move to CPU before converting to numpy
            acc_np[0] += acc_h_safe.detach().cpu().numpy()
            acc_np[1] += acc_h_unsafe.detach().cpu().numpy()
            acc_np[2] += acc_deriv_safe.detach().cpu().numpy()
            
            loss_np[0] += loss_h_safe.detach().cpu().numpy()
            loss_np[1] += loss_h_unsafe.detach().cpu().numpy()
            loss_np[2] += loss_deriv_safe.detach().cpu().numpy()
            loss_np[3] += loss_cql_actions.detach().cpu().numpy()
            loss_np[4] += loss_cql_states.detach().cpu().numpy()
            
            total_safe_h += torch.sum(h.reshape(-1,1) * safe_mask.reshape(-1,1)).detach().cpu().numpy()
            total_unsafe_h += torch.sum(h.reshape(-1,1) * unsafe_mask.reshape(-1,1)).detach().cpu().numpy()
            
            num_safe_samples += num_safe.detach().cpu().numpy()
            num_unsafe_samples += num_unsafe.detach().cpu().numpy()
        
        avg_safe_h = total_safe_h / (num_safe_samples + 1e-5)
        avg_unsafe_h = total_unsafe_h / (num_unsafe_samples + 1e-5)
        
        return acc_np / self.opt_iter, loss_np / self.opt_iter, avg_safe_h, avg_unsafe_h

    def get_mask(self, state):
        dist = torch.norm(state[:,:2] - state[:,2:4], dim=1)
        safe_mask = dist > self.safe_distance
        unsafe_mask = dist <= self.safe_distance
        return safe_mask, unsafe_mask
    
    
