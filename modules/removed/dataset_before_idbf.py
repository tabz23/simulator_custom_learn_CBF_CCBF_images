import numpy as np
import torch 
import torch as nn
from network import *
BC_SAFE_MODEL_PATH = "bc_safe_model.pt"
DEVICE="cpu"
class Dataset(object):
    
    def __init__(self,state_car_dim, state_obstacles_dim, control_dim, buffer_size=10000, safe_distance=4.0):
        self.total_state_dim= state_car_dim+state_obstacles_dim
        self.control_dim=control_dim
        self.safe_distance=safe_distance
        
        self.buffer_size = buffer_size
        self.buffer_safe = []
        self.buffer_unsafe = []

        
    def add_data(self, total_state, control, total_state_next, idbf_masks=False):
        
        if not idbf_masks:

            data = [np.copy(total_state).astype(np.float32), 
                    np.copy(control).astype(np.float32), 
                    np.copy(total_state_next).astype(np.float32)]
        
            dist=np.linalg.norm(total_state[:2] - total_state[2:4])
        
            # print(dist)
            if dist<self.safe_distance:
                self.buffer_unsafe.append(data)
                self.buffer_unsafe=self.buffer_unsafe[-self.buffer_size:]
                
            ###IMPORTANT ###IMPORTANT ###IMPORTANT ###IMPORTANT ###IMPORTANT ###IMPORTANT
            ###DONT ADD ALL THE SAFE DATA add buffer region so tha unsafe set a bit bigger
            elif dist>=self.safe_distance+3:
                self.buffer_safe.append(data)
                self.buffer_safe=self.buffer_safe[-self.buffer_size:]
        else:
            
            data = [np.copy(total_state).astype(np.float32), 
            np.copy(control).astype(np.float32), 
            np.copy(total_state_next).astype(np.float32)]
            
            ##add all input data to the safe buffer since we are passing safe_trajectories to add_data
            self.buffer_safe.append(data)
            self.buffer_safe=self.buffer_safe[-self.buffer_size:]
            state_dim = 4  # car (2) + obstacle (2)
            action_dim = 2
            hidden_sizes = (64, 64)
            action_low = np.array([-3.0, -3.0])
            action_high = np.array([3.0, 3.0])
            bc_safe_model = MLPGaussianActor(
                obs_dim=state_dim,
                act_dim=action_dim,
                action_low=action_low,
                action_high=action_high,
                hidden_sizes=hidden_sizes,
                activation=nn.ReLU,
                device=DEVICE
            ).to(DEVICE)
            bc_safe_model.load_state_dict(torch.load(BC_SAFE_MODEL_PATH, map_location=DEVICE))
            states_tensor = torch.tensor(total_state, dtype=torch.float32).to(self.device)
            actions_tensor = torch.tensor(control, dtype=torch.float32).to(self.device)
            _, _, log_probs = self.actor(states_tensor, actions_tensor)##i think will be size 1,1 since appending 1 datapt at a time
            ##if proba less than 35 % then unsafe
            
            if torch.exp(log_probs)<0.35:
                self.buffer_unsafe.append(data)
                self.buffer_unsafe=self.buffer_unsafe[-self.buffer_size:]
            elif dist>=self.safe_distance+3:
                self.buffer_safe.append(data)
                self.buffer_safe=self.buffer_safe[-self.buffer_size:]
            

    def sample_data(self,batch_size):
        num_safe=batch_size // 2 ##we want to sample this many
        num_unsafe=batch_size-num_safe  ##we want to sample this many
        
        if (len(self.buffer_safe)!=0):##we actually have safe samples
            s_safe, u_safe, s_next_safe= self.sample_data_from_buffer(num_safe, self.buffer_safe)
        if (len(self.buffer_unsafe)!=0):##we actually have unsafe samples
            s_unsafe, u_unsafe, s_next_unsafe  = self.sample_data_from_buffer(num_unsafe, self.buffer_unsafe)
       

        s = np.concatenate([s_safe, s_unsafe ], axis=0)
        u = np.concatenate([u_safe, u_unsafe], axis=0)
        s_next = np.concatenate([s_next_safe, s_next_unsafe], axis=0)
        
        return s,  u, s_next

    def sample_data_from_buffer(self, batch_size, buffer):

        indices = np.random.randint(len(buffer), size=(batch_size))
        s = np.zeros((batch_size, self.total_state_dim), dtype=np.float32)
        u = np.zeros((batch_size, self.control_dim), dtype=np.float32)
        s_next = np.zeros((batch_size, self.total_state_dim), dtype=np.float32)

        for i, ind in enumerate(indices):
            state, control, state_next = buffer[ind]
            s[i] = state
            u[i] = control
            s_next[i] = state_next
        return s,  u,  s_next
        
        
        