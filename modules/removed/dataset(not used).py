import numpy as np
import torch 
import torch as nn
from modules.network import *

BC_SAFE_MODEL_PATH = "bc_safe_model.pt"
DEVICE="cpu"
class Dataset(object):
    
    def __init__(self,state_car_dim, state_obstacles_dim, control_dim, buffer_size=10000, safe_distance=4.0):
        self.total_state_dim= state_car_dim+state_obstacles_dim
        self.control_dim=control_dim
        self.safe_distance=safe_distance
        self.device=DEVICE
              
        self.buffer_size = buffer_size
        self.buffer_safe = []
        self.buffer_unsafe = []
               
        self.bc_safe_model = MLPGaussianActor(
            obs_dim=state_car_dim + state_obstacles_dim,##takes state dimension 4
            act_dim=control_dim,##action dimension 2
            action_low=np.array([-3.0, -3.0]),
            action_high=np.array([3.0, 3.0]),
            hidden_sizes=(64, 64),
            activation=nn.ReLU,
            device=DEVICE
        ).to(DEVICE)
        self.bc_safe_model.load_state_dict(torch.load(BC_SAFE_MODEL_PATH, map_location=DEVICE))
        self.bc_safe_model.eval()  # Set to evaluation mode

    def add_data(self, total_state, control, total_state_next, state_image, state_image_next, idbf_masks=False):     
        if not idbf_masks:
            
            data = [
                np.copy(total_state).astype(np.float32), 
                np.copy(control).astype(np.float32), 
                np.copy(total_state_next).astype(np.float32),
                np.copy(state_image),  # Include the RGB image
                np.copy(state_image_next) # Include the next RGB image
            ]
            dist=np.linalg.norm(total_state[:2] - total_state[2:4])#based on groud truth state label images
        
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
            
            ##IMPORTANT
            ##add all input data to the safe buffer since we are passing safe_trajectories to add_data
            self.buffer_safe.append(data)
            self.buffer_safe=self.buffer_safe[-self.buffer_size:]

            states_tensor = torch.tensor(total_state, dtype=torch.float32).to(self.device)
            for i in range(5):
                random_actions_tensor = torch.rand(1,2) * 4 - 2 ##shape is 1,2. action between -2,2
                pi, _, log_probs = self.bc_safe_model(states_tensor, random_actions_tensor)##i think will be size 1,1 since appending 1 datapt at a time
                ##if proba less than 35 % then unsafe
                if torch.exp(log_probs).item()<0.000001:##won't matter much but sample from the policy itself not from uniform
                    # print(torch.exp(log_probs).item())
                    next_state=np.copy(total_state)##first 2 dimensions are position car. update position of car but keep last 2 dimensions(position obstacle) same
                    next_state[0:2]=next_state[:2] + random_actions_tensor.cpu().numpy()[0] * 0.1##dt is 0.1
                    unsafe_data = [
                        next_state.astype(np.float32),
                        random_actions_tensor.cpu().numpy()[0].astype(np.float32),
                        np.copy(total_state_next).astype(np.float32)
                    ]
                    self.buffer_unsafe.append(unsafe_data)  
                    self.buffer_unsafe=self.buffer_unsafe[-self.buffer_size:]
                    
    def sample_data(self,batch_size):
        num_safe=batch_size // 2 ##we want to sample this many
        num_unsafe=batch_size-num_safe  ##we want to sample this many
        
        s_safe, u_safe, s_next_safe, img_safe, img_next_safe = [], [], [], [], []
        s_unsafe, u_unsafe, s_next_unsafe, img_unsafe, img_next_unsafe = [], [], [], [], []
        
        if (len(self.buffer_safe)!=0):##we actually have safe samples
            s_safe, u_safe, s_next_safe, img_safe, img_next_safe = self.sample_data_from_buffer(num_safe, self.buffer_safe)
        if (len(self.buffer_unsafe)!=0):##we actually have unsafe samples
            s_unsafe, u_unsafe, s_next_unsafe, img_unsafe, img_next_unsafe = self.sample_data_from_buffer(num_unsafe, self.buffer_unsafe)
        s = np.concatenate([s_safe, s_unsafe ], axis=0)
        u = np.concatenate([u_safe, u_unsafe], axis=0)
        s_next = np.concatenate([s_next_safe, s_next_unsafe], axis=0)
        img = np.concatenate([img_safe, img_unsafe], axis=0)
        img_next = np.concatenate([img_next_safe, img_next_unsafe], axis=0)
        
        return s, u, s_next, img, img_next

    def sample_data_from_buffer(self, batch_size, buffer):
        ##buffer[0][3].shape is (64, 64, 3) -> *buffer[0][3].shape unpacks this into 64, 64, 3
        if len(buffer) == 0:
            return [], [], [], [], []
        indices = np.random.randint(len(buffer), size=(batch_size))
        s = np.zeros((batch_size, self.total_state_dim), dtype=np.float32)
        u = np.zeros((batch_size, self.control_dim), dtype=np.float32)
        s_next = np.zeros((batch_size, self.total_state_dim), dtype=np.float32)
        img=np.zeros((batch_size, *buffer[0][3].shape),dtype=np.float32)
        img_next=np.zeros((batch_size, *buffer[0][3].shape), dtype=np.float32)
        

        for i, ind in enumerate(indices):
            state, control, state_next, image, image_next = buffer[ind]
            s[i] = state
            u[i] = control
            s_next[i] = state_next
            img[i]=image
            img_next[i]=image_next
        return s,  u,  s_next, img, img_next
        
        
        