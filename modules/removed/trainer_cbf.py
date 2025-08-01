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
                lr=1e-4
                ##ADD HERE BOOL FOR TRAIN USING INDBF
                 ):
        self.cbf=cbf
        self.dataset=dataset
        self.lr=lr
        self.batch_size=batch_size
        self.opt_iter=opt_iter
        self.eps_safe=eps_safe
        self.eps_unsafe=eps_unsafe
        self.safe_distance=safe_distance
        self.safe_loss_weight=safe_loss_weight
        self.unsafe_loss_weight=unsafe_loss_weight
        self.action_loss_weight=action_loss_weight
        self.dt=dt
        self.device = device
        self.cbf_optimizer=torch.optim.Adam(self.cbf.parameters(), lr=self.lr, weight_decay=5e-5)
        
    def train_cbf(self):
        loss_np=np.zeros(3,dtype=np.float32)
        acc_np=np.zeros(3,dtype=np.float32)
        total_safe_h=0
        total_unsafe_h=0
        num_safe_samples=0
        num_unsafe_samples=0
##ADD NEW WAY TO TRAIN WHERE WE USE DIFFERENT DATASET WITH UNSAFE BEING COMING FROM INDBF
        
        for i in range(self.opt_iter):
            state, control, next_state=self.dataset.sample_data(batch_size=self.batch_size)
            state=torch.from_numpy(state).to(self.device)
            control=torch.from_numpy(control).to(self.device)
            next_state=torch.from_numpy(next_state).to(self.device)
            
            safe_mask,unsafe_mask=self.get_mask(state)
            
            h=self.cbf(state)
            h_dot=self.cbf.compute_h_dot(state,next_state)
            
            num_safe=torch.sum(safe_mask).float()
            num_unsafe=torch.sum(unsafe_mask).float()
            
            relu=nn.ReLU()
            # Apply safe_loss_weight to safe loss
            loss_h_safe=self.safe_loss_weight * torch.sum(relu(self.eps_safe-h).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe+1e-5)
            # print(loss_h_safe)
            loss_h_unsafe=self.unsafe_loss_weight * torch.sum(relu(self.eps_unsafe+h).reshape(-1,1) * unsafe_mask.reshape(-1,1)) / (num_unsafe+1e-5)

            # Normalize accuracies to be between 0 and 1
            acc_h_safe = torch.sum((h >= 0).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe + 1e-5)
            acc_h_unsafe = torch.sum((h < 0).reshape(-1,1) * unsafe_mask.reshape(-1,1)) / (num_unsafe + 1e-5)
            
            deriv_cond = h_dot + h
            loss_deriv_safe = torch.sum(relu(-deriv_cond).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe + 1e-5)
            acc_deriv_safe = torch.sum((deriv_cond > 0).reshape(-1,1) * safe_mask.reshape(-1,1)) / (num_safe + 1e-5)
            
            loss = loss_h_safe + loss_h_unsafe + loss_deriv_safe
            
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
            
            total_safe_h += torch.sum(h.reshape(-1,1) * safe_mask.reshape(-1,1)).detach().cpu().numpy()
            total_unsafe_h += torch.sum(h.reshape(-1,1) * unsafe_mask.reshape(-1,1)).detach().cpu().numpy()
            
            num_safe_samples += num_safe.detach().cpu().numpy()
            num_unsafe_samples += num_unsafe.detach().cpu().numpy()
        
        avg_safe_h = total_safe_h / (num_safe_samples + 1e-5)
        avg_unsafe_h = total_unsafe_h / (num_unsafe_samples + 1e-5)
        
        return acc_np / self.opt_iter, loss_np / self.opt_iter, avg_safe_h, avg_unsafe_h

    def get_mask(self,state):
        dist=torch.norm(state[:,:2] - state[:,2:4], dim=1)
        safe_mask=dist>self.safe_distance
        unsafe_mask=dist<=self.safe_distance
        return safe_mask, unsafe_mask
    ##ADD MASK METHOD FOR IF USING IDBF