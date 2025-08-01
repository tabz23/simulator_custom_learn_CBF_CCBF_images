import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.normal import Normal

def build_MLP(hidden_dims, dropout=0, activation=torch.nn.ReLU,with_bn=False,no_act_last_layer=True):
    modules=[]
    for i in range(len(hidden_dims)-1):##adds all hidden layers including output layer
        modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        if not (no_act_last_layer and i==len(hidden_dims)-2):# ensures that activation, batch normalization, and dropout are skipped only for the second-to-last layer when no_act_last_layer=True
            if with_bn:
                modules.append(torch.nn.BatchNorm1d(hidden_dims[i+1]))
            modules.append(activation())
            if dropout>0:
                modules.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*modules)
        
             
class CBF(torch.nn.Module):
    def __init__(self, state_car_dim=2, state_obstacles_dim=2,dt=0.1,num_hidden_dim=1,dim_hidden=4):
        super().__init__()
        self.full_state_dim=4
        # self.full_state_dim=state_car_dim+state_obstacles_dim
        self.dt=dt
        self.activation=torch.nn.Tanh()
        self.dim_hidden=dim_hidden
        
        layers=[self.full_state_dim]
        for i in range(num_hidden_dim):
            layers.append(dim_hidden)
        layers.append(1)
        self.cbf=build_MLP(layers)
    
    def forward(self, full_state):##takes in car and obstacle coordinates
        return self.activation(self.cbf(full_state))
    
    def compute_h_dot(self, state, next_state):
        return (self.forward(next_state) - self.forward(state) ) / self.dt
    
def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multi-layer perceptron with the specified sizes and activations.

    Args:
        sizes (list): A list of integers specifying the size of each layer in the MLP.
        activation (nn.Module): The activation function to use for all layers except the output layer.
        output_activation (nn.Module): The activation function to use for the output layer. Defaults to nn.Identity.

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = torch.nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return torch.nn.Sequential(*layers)

class MLPGaussianActor(nn.Module):
    """
    A MLP Gaussian actor
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        action_low (np.ndarray): A 1D numpy array of lower bounds for each action dimension.
        action_high (np.ndarray): A 1D numpy array of upper bounds for each action dimension.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        device (str): The device to use for computation (cpu or cuda).
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 action_low,
                 action_high,
                 hidden_sizes,
                 activation,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.action_low = torch.nn.Parameter(torch.tensor(action_low,dtype=torch.float32,
                                                          device=device)[None, ...],
                                             requires_grad=False)  # (1, act_dim)
        self.action_high = torch.nn.Parameter(torch.tensor(action_high,dtype=torch.float32,##float32 to train on mac mps
                                                           device=device)[None, ...],
                                              requires_grad=False)  # (1, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = torch.sigmoid(self.mu_net(obs))
        mu = self.action_low + (self.action_high - self.action_low) * mu
        std = torch.exp(self.log_std)
        return mu, Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None, deterministic=False):
        '''
        Produce action distributions for given observations, and
        optionally compute the log likelihood of given actions under
        those distributions.
        If act is None, sample an action
        '''
        mu, pi = self._distribution(obs)
        if act is None:
            act = pi.sample()
        if deterministic:
            act = mu
        logp_a = self._log_prob_from_distribution(pi, act)
        return pi, act, logp_a


LOG_STD_MAX = 2
LOG_STD_MIN = -20

