import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from scripts.vaedynamics_newstart_good_3_decoderaffine_nosequential_4_deterministic import DynamicsModel, VAE
# Silence cvxopt output
solvers.options['show_progress'] = False

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Import required modules
from modules.network import MLPGaussianActor, CBF
# from modules.dataset import Dataset
from envs.car import DubinsCarEnv

# Define constants
DATASET_PATH = "/Users/i.k.tabbara/Documents/python directory/safe_rl_dataset_images_ALLUNSAFE_big_obstacle_3000traj.npz"
GOAL_POSITION = np.array([20, 21])

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsFalse_cql_states_weight0.1_cql_actions_weight0.01_225.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999cql_statesFalsecql_actionsTrue_cql_states_weight0.1_cql_actions_weight0.01_122.pt" 
# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsFalse_cql_states_weight0.1_cql_actions_weight0.01_225.pt"



# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql.pth"
# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4.pth"

# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_479.pth"
# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_336.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_final755.pth"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_final295.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_436.pth"#this is next
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_657.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_436.pth"#this is next
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_657.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_692.pth"#this is next
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_432.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_final746.pth"#this is next
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_final472.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_402.pth"#this is next
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_489.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_985.pth"#this is next
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_461.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_637.pth"#this is next
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_379.pth"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_478.pth"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_933.pth"#this is next

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_478.pth" #CBF#THIS IS WHAT I USED FOR REPORT
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_933.pth" #CCBF#THIS IS WHAT I USED FOR REPORT

CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_478.pth"#cbf in paper
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_915.pth"#idbf in paper


CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_478.pth"
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_272.pth"##temp used for newly trained clone of 933. performs as good for bc and bc safe but worse for pd controller

CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_589.pth"
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_272.pth"##temp used for newly trained clone of 933. performs as good for bc and bc safe but worse for pd controller

CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_589.pth"
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_607.pth"##check without detach used this for paper appendix
with_constraint=False
"""==================================================for 25 runs each and for final755 (not sure if chose nominal when could find solution)
Evaluating BC model (all data)..
Evaluation over 25 episodes:
Average Reward: 32.46
Average Cost: 1.92
Success Rate: 100.0%
Collision Rate: 12.0%
.
Evaluating BC-Safe model (safe data only)...
Evaluation over 25 episodes:
Average Reward: 33.21
Average Cost: 1.48
Success Rate: 96.0%
Collision Rate: 8.0%

Evaluating nominal CBF-QP controller (goal-reaching)...alpha=0.5
CBF-QP Controller Evaluation over 25 episodes:
Average Reward: 26.77
Average Cost: 7.68
Success Rate: 48.0%
Collision Rate: 40.0%

Evaluating CBF-QP controller (BC + CBF)...alpha=0.1
CBF-QP Controller Evaluation over 25 episodes:
Average Reward: 33.01
Average Cost: 2.04
Success Rate: 96.0%
Collision Rate: 8.0%
"""


BC_MODEL_PATH = "bc_model_images.pt"
BC_SAFE_MODEL_PATH = "bc_safe_model_images.pt"
DEVICE = "cpu"


##i think above ccbf checkpt was good

vae_checkpoint_file = "/Users/i.k.tabbara/Documents/python directory/vae_model_final_4.pth"
dynamics_checkpoint_file = "/Users/i.k.tabbara/Documents/python directory/dynamics_model_final_4.pth"

input_channels = 3
latent_dim = 4
action_dim = 2
hidden_dim = 400
batch_size = 32  # Set to 1 for easy visualization

vae = VAE(input_channels=input_channels, latent_dim=latent_dim, hidden_dim=hidden_dim)
dynamics_model = DynamicsModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)

vae.load_state_dict(torch.load(vae_checkpoint_file, map_location=DEVICE))
dynamics_model.load_state_dict(torch.load(dynamics_checkpoint_file, map_location=DEVICE))
vae.to(DEVICE)
dynamics_model.to(DEVICE)
vae.eval()
dynamics_model.eval()


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class BCTrainer:
    """Behavior Cloning Trainer for the DubinsCar environment."""
    
    def __init__(self, actor, optimizer, device="cpu"):
        """Initialize the BC trainer with an actor network and optimizer."""
        self.actor = actor
        self.optimizer = optimizer
        self.device = device
        self.loss_history = []
        
    def train_step(self, states, actions):
        """Perform a single training step using behavior cloning."""
        self.optimizer.zero_grad()
        
        # Move data to device
        states_tensor = states.clone().detach().to(torch.float32).to(DEVICE)#torch.tensor(states, dtype=torch.float32).to(self.device)

        actions_tensor = actions.clone().detach().to(torch.float32).to(DEVICE)#torch.tensor(actions, dtype=torch.float32).to(self.device)
        
        # Forward pass
        _, _, log_probs = self.actor(states_tensor, actions_tensor)
        
        weight_hyperparam=5
        # BC loss is negative log likelihood of expert actions
        loss = -log_probs.mean() * weight_hyperparam
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, epochs=10, log_interval=10):
        """Train the actor using behavior cloning for multiple epochs."""
        print(f"Training behavior cloning model for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for rgb_image, states, actions, _ in dataloader:
                # print(rgb_image.shape)
                rgb_image = rgb_image.to(torch.float32).to(DEVICE)## by default it was float64, change to float32
                _, z = vae(rgb_image)
    
                loss = self.train_step(z, actions)
                epoch_loss += loss
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            self.loss_history.append(avg_epoch_loss)
            
            if (epoch ) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return self.loss_history

###UNCOMMENT TWO LINES BELOW TO PRODUCE IMAGES WHEN CALLING BELOW FUNCTION
plt.ion()
fig_img_state, ax_img_state = plt.subplots(figsize=(4, 4), facecolor='white')##this is top
# goal_position = np.array([20, 21])
def get_image_from_state(state):

    # fig, ax = plt.subplots(figsize=(4,4), facecolor='white')##commented from here and added it on top instead
    # fig, ax = plt.subplots(figsize=(4,4), facecolor='white')
    ax_img_state.clear()
    ax_img_state.set_xlim(-30, 30)
    ax_img_state.set_ylim(-30, 30)
    ax_img_state.set_facecolor('white')  # White background
    obstacle_circle = plt.Circle(
        (state[2],state[3]), 
        4, ##hardcode circle radius as 4 because in the env.onstacle_radius will not be 4 as we are vaying it for generating trajectories using cbf
        color='brown', 
        alpha=1
    
    )
    
    
        # Add a small rectangle at goal_position FOR SAKE OF PUTTING IN PAPER ONLY
    # goal_rect = plt.Rectangle(
    #     (goal_position[0] - 1, goal_position[1] - 1),  # Bottom-left corner
    #     width=2, height=2,  # Rectangle size
    #     color='blue', alpha=0.7
    # )
        # Create legend
    # import matplotlib.patches as patches
    # legend_elements = [
    #     patches.Patch(color='blue', alpha=0.7, label='Goal'),
    #     patches.Circle((0, 0), radius=4, color='brown', alpha=1, label='Obstacle'),
    #     plt.Line2D([0], [0], marker='o', color='black', markersize=10, label='Agent')
    # ]
    # ax_img_state.legend(handles=legend_elements, loc='upper left', fontsize='large')
    # ax_img_state.add_patch(goal_rect)
    
    
    ax_img_state.add_artist(obstacle_circle)
    ax_img_state.scatter(state[0], state[1], color='black', s=300)
    # Remove axis and white space
    ax_img_state.set_axis_off()
    plt.tight_layout(pad=0)
    
    
    # plt.draw()  ##THESE CAN BE USED INSTEAD OF ENV.RENDER
    # plt.pause(0.0000001) 

    # Save figure to memory buffer
    buf = BytesIO()
    fig_img_state.savefig(buf, format='png', dpi=16, bbox_inches='tight', pad_inches=0, transparent=False)
    buf.seek(0)

    # Convert to RGB array
    img = Image.open(buf)
    # plt.close(fig)#####
    # img.show()
    # img.save("image.png")  # Save the image to a file
    img_array = np.array(img.convert('RGB'))
    
    ###UNCOMMENT THE TWO BELOW TO SHOW THE PRODUCED IMAGES
    # plt.pause(0.00001)
    # plt.draw()
    # print(img_array.shape)
    buf.close()  # Close the buffer to free up memory
    return img_array/255.0

def create_dataloader(data, batch_size=64, only_safe=False):
    """Create a dataloader from the provided dataset."""
    rgb_states=data["states_rgb"]##################################this is dataset size,width,height,3
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    costs = data["costs"]
    rewards = data["rewards"]
    episode_starts = data["episode_starts"]
    episode_lengths = data["episode_lengths"]
    
    if only_safe:
        # Identify safe trajectories (episodes with zero cost throughout)
        safe_episodes = []
        start_idx = 0
        safe_rewards = []
        safe_costs = []
        for length in episode_lengths:
            end_idx = start_idx + length
            if np.all(costs[start_idx:end_idx] == 0):
                safe_episodes.append((start_idx, end_idx))
                safe_rewards.append(np.mean(rewards[start_idx:end_idx]))
                safe_costs.append(np.mean(costs[start_idx:end_idx]))
            start_idx = end_idx
        
        # Extract states, actions, and next_states from safe trajectories
        safe_indices = np.concatenate([np.arange(start, end) for start, end in safe_episodes])
        states = states[safe_indices]
        actions = actions[safe_indices]
        next_states = next_states[safe_indices]
        rgb_states=rgb_states[safe_indices]################################still need to permute them
        
        avg_safe_reward = np.mean(safe_rewards) if safe_rewards else 0
        avg_safe_cost = np.mean(safe_costs) if safe_costs else 0
        
        print(f"Using {len(states)} safe transitions from {len(safe_episodes)} safe trajectories out of {len(episode_lengths)} total trajectories")
        print(f"Average state reward of safe trajectories: {avg_safe_reward:.4f}")
        print(f"Average state cost of safe trajectories: {avg_safe_cost:.4f}")
    else:
        # Calculate average reward and cost for all trajectories
        all_rewards = []
        all_costs = []
        start_idx = 0
        for length in episode_lengths:
            end_idx = start_idx + length
            all_rewards.append(np.mean(rewards[start_idx:end_idx]))
            all_costs.append(np.mean(costs[start_idx:end_idx]))
            start_idx = end_idx
        
        avg_all_reward = np.mean(all_rewards)
        avg_all_cost = np.mean(all_costs)
        
        print(f"Using all {len(states)} transitions from {len(episode_lengths)} trajectories")
        print(f"Average reward of all trajectories: {avg_all_reward:.4f}")
        print(f"Average cost of all trajectories: {avg_all_cost:.4f}")
    
    # Create dataset class for batching
    class SimpleDataset:
        def __init__(self, rgb_states, states, actions, next_states):
            self.states = states
            self.rgb_states=rgb_states
            
            self.actions = actions
            self.next_states = next_states
            self.length = len(states)
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            rgb_image = self.rgb_states[idx].transpose(2, 0, 1) / 255.0 ##PERMUTE THE IMAGE
            return rgb_image, self.states[idx], self.actions[idx], self.next_states[idx]################################s
    
    dataset = SimpleDataset(rgb_states, states, actions, next_states)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader


def load_dataset():
    """Load the dataset from the specified path."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        return data
    else:
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

def evaluate_model(actor, env, num_episodes=5, render=False, render_delay=0.0):
    """Evaluate a trained model in the environment."""
    total_rewards = []
    total_costs = []

    episode_successes = []  # Track success for each episode (0 or 1)
    episode_collisions = []  # Track collision for each episode (0 or 1)
    for episode in range(num_episodes):
        start_time = time.time()  # Start time of the episode
        
        
    
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        step = 0
        episode_collision = False
        
        while not done and step < 200:  # Cap at 200 steps to prevent infinite episodes
            # Prepare state input (append obstacle position)
            full_state = np.concatenate([state, env.obstacle_position])
            
            # Get action from policy
            with torch.no_grad():
                # state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(DEVICE)
                image_numpy=get_image_from_state(full_state)
                image_tensor=torch.FloatTensor(image_numpy).unsqueeze(0).permute(0,3,1,2).to(DEVICE)#.unsqueeze(0)-> batch size=1 ( operation in PyTorch adds a new dimension at 0 )
                _, z = vae(image_tensor)
                

                _, action, _ = actor(z, deterministic=True)
                action = action.cpu().numpy().flatten()##since it returns 1, action the neural net where 1 correspond to batch size we flatten it
            
            # Take step in environment
            next_state, reward, cost, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_cost += cost
            step += 1
            
            # Check for collision
            if cost > 0:
                episode_collision = True
            
            # Render if requested
            if render:
                env.render()
                time.sleep(render_delay)
            
            # Update state
            state = next_state
        
        # Close rendering window
        # if render:
        #     plt.close()
        
        # Update metrics
        end_time = time.time()  # End time of the episode
        time_taken = end_time - start_time  # Calculate the time taken
        print(f"Episode {episode} done. Time taken: {time_taken:.2f} seconds")
        
        # Print metrics after every episode
        # print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, " +
        #       f"Steps = {step}, Goal reached = {info['goal_reached']}, Collision = {episode_collision}")
    
    # Calculate averages
        # Update metrics
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        episode_successes.append(int(info["goal_reached"]))
        episode_collisions.append(int(episode_collision))
    
    # Calculate overall averages
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)

    success_rate = np.mean(episode_successes) * 100
    collision_rate = np.mean(episode_collisions) * 100

    num_batches = 5
    batch_size = len(total_rewards) // num_batches  # Ensure equal-sized batches

    # Compute batch means for rewards and costs
    reward_batch_means = [np.mean(total_rewards[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]
    cost_batch_means = [np.mean(total_costs[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]

    # Compute batch means for success and collision rates
    success_batch_means = [np.mean(episode_successes[i * batch_size: (i + 1) * batch_size]) * 100 for i in range(num_batches)]
    collision_batch_means = [np.mean(episode_collisions[i * batch_size: (i + 1) * batch_size]) * 100 for i in range(num_batches)]

    # Compute standard deviation of batch means
    std_reward = np.std(reward_batch_means, ddof=1)
    std_cost = np.std(cost_batch_means, ddof=1)
    success_std = np.std(success_batch_means, ddof=1)
    collision_std = np.std(collision_batch_means, ddof=1)
        
        # Print metrics after every episode
        # print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, " +
        #       f"Steps = {step}, Goal reached = {info['goal_reached']}, Collision = {episode_collision}")
    
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Cost: {avg_cost:.2f} ± {std_cost:.2f}")
    # print(f"Steps: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Success Rate: {success_rate:.1f}% ± {success_std:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}% ± {collision_std:.1f}%")
    
    # Return both means and standard deviations
    return {
        'reward': (avg_reward, std_reward),
        'cost': (avg_cost, std_cost),
        # 'steps': (avg_steps, std_steps),
        'success_rate': (success_rate, success_std),
        'collision_rate': (collision_rate, collision_std)
    }

class CBFQPController_BC:
    """Controller that combines BC policy with CBF-QP safety filter."""
    
    def __init__(self, actor, cbf, device="cpu",alpha=0.1):
        """Initialize the controller with a policy network and CBF."""
        self.actor = actor
        self.cbf = cbf
        self.device = device
        self.dt = 0.1  # Environment timestep
        self.alpha=alpha
        
    def get_safe_action(self, state, car_dynamics_model):
        """
        Get a safe action by solving a QP that minimizes deviation from the BC policy 
        while satisfying CBF constraints using CVXPY.
        """
        # Get nominal action from behavior cloning policy
        with torch.no_grad():##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix##fix
            image_numpy=get_image_from_state(state)##get the image
            image_tensor=torch.FloatTensor(image_numpy).unsqueeze(0).permute(0,3,1,2).to(DEVICE)#.unsqueeze(0)-> batch size=1 ( operation in PyTorch adds a new dimension at 0 )
            _, z = vae(image_tensor)
                
            
            # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, action, _ = self.actor(z, deterministic=True)
            nominal_action = action.cpu().numpy().flatten()
        
         # Current control barrier function value

        # image_numpy=get_image_from_state(state)
        # image_tensor=torch.FloatTensor(image_numpy).unsqueeze(0).permute(0,3,1,2).to(DEVICE)#.unsqueeze(0)-> batch size=1 ( operation in PyTorch adds a new dimension at 0 )
        # _, z = vae(image_tensor)
        # plt.ion()
        # plt.imshow(image_numpy)
        # plt.axis('off')  # Remove the axis
        # plt.show(image_numpy)
        
        h_x = self.cbf(z).item()##h_x is scalar
        # Get gradient of CBF with respect to state
        z.requires_grad_(True)
        h_x_tensor = self.cbf(z)
        gradient_B=torch.autograd.grad(h_x_tensor,z)[0]#torch.Size([1, 4])
        gradient_numpy=gradient_B.cpu().numpy()#shape  is 1,4 
        f, g = dynamics_model.dynamics.forward(z)
        cbf_lie_derivative = torch.einsum("bs,bs->b", f, gradient_B)
        right_side = cbf_lie_derivative + self.alpha * h_x_tensor.squeeze(-1)###CHANGE ALPHA HERE CHANGE ALPHA HERE.  LOW ALPHA->SAFER, WORSE REWARD
        right_numpy = right_side.detach().cpu().numpy()
        grad_b_g = torch.einsum(
            'bs,bsa->ba', 
            gradient_B, 
            g.view(g.shape[0], dynamics_model.dynamics.state_dim, dynamics_model.dynamics.num_action)
        )
        grad_b_g_numpy = grad_b_g.detach().cpu().numpy()
        # Solve QP to find safe action
        u = cp.Variable(dynamics_model.dynamics.num_action)
        objective = cp.Minimize(cp.sum_squares(u - nominal_action))

        if (with_constraint):
            constraints = [grad_b_g_numpy @ u >= -right_numpy,
                    u<=3, u>=-3
                        ]
        else:
            constraints = [grad_b_g_numpy @ u >= -right_numpy,
                  # u<=3, u>=-3
                    ]

        prob = cp.Problem(objective, constraints)
        try:
            # Solve using OSQP solver
            prob.solve(solver=cp.OSQP)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                safe_action = u.value
            else:
                print(f"Warning: QP could not find optimal solution (status: {prob.status}), using nominal action")
                safe_action = nominal_action
        except Exception as e:
            print(f"Error in QP solver: {e}")
            safe_action = nominal_action
        
        return safe_action, nominal_action

def evaluate_cbf_qp_controller(controller, env, num_episodes=1, render=True, render_delay=0.000):
    """Evaluate the CBF-QP controller in the environment with side-by-side visualization."""
    total_rewards = []
    total_costs = []
    episode_successes = []  # Track success for each episode (0 or 1)
    episode_collisions = []  # Track collision for each episode (0 or 1)


    for episode in range(num_episodes):
        start_time = time.time()  # Start time of the episode
        
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        step = 0
        episode_collision = False
        
        # For tracking actions and CBF values
        nominal_actions = []
        safe_actions = []
        cbf_values = []
        car_positions = []  # Track car trajectory
        
        # Set up live plotting
        # if not render:
        #     # Turn on interactive mode
        #     plt.ion()

        #     # Create a separate figure without affecting global figures
        #     figx, axes = plt.subplots(3, 1, figsize=(10, 8), num="CBF Analysis")  
        #     figx.canvas.manager.set_window_title('CBF Analysis')

        #     # Unpacking subplots
        #     ax2, ax3, ax4 = axes

        #     # Titles and labels
        #     ax2.set_title('CBF Values Over Time')
        #     ax2.set_xlabel('Time Step')
        #     ax2.set_ylabel('CBF Value')
        #     ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line for CBF

        #     ax3.set_title('X-Velocity Actions')
        #     ax3.set_xlabel('Time Step')
        #     ax3.set_ylabel('Action Value')

        #     ax4.set_title('Y-Velocity Actions')
        #     ax4.set_xlabel('Time Step')
        #     ax4.set_ylabel('Action Value')

        #     plt.tight_layout()

        #     # Define plot lines
        #     cbf_plot, = ax2.plot([], [], 'g-', label='CBF Value')

        #     nominal_x_plot, = ax3.plot([], [], 'b-', label='Nominal')
        #     safe_x_plot, = ax3.plot([], [], 'r-', label='Safe')

        #     nominal_y_plot, = ax4.plot([], [], 'b-', label='Nominal')
        #     safe_y_plot, = ax4.plot([], [], 'r-', label='Safe')

        #     # Add legends
        #     ax2.legend()
        #     ax3.legend()
        #     ax4.legend()

        
        while not done :  # Cap at 200 steps to prevent infinite episodes
            # Prepare state input (append obstacle position)
            full_state = np.concatenate([state, env.obstacle_position])
            
            # Get CBF value for current state
            with torch.no_grad():
                # state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(controller.device)
                
                image_numpy=get_image_from_state(full_state)
                image_tensor=torch.FloatTensor(image_numpy).unsqueeze(0).permute(0,3,1,2).to(DEVICE)#.unsqueeze(0)-> batch size=1 ( operation in PyTorch adds a new dimension at 0 )
                # print(f"Shape of image_numpy: {image_numpy.shape}") 
                _, z = vae(image_tensor)
                cbf_value = controller.cbf(z).item()
                cbf_values.append(cbf_value)
            
            # Get safe action from CBF-QP controller
            safe_action, nominal_action = controller.get_safe_action(full_state, env)
            
            # Store actions and positions for visualization
            nominal_actions.append(nominal_action)
            safe_actions.append(safe_action)
            car_positions.append(state[:2])  # Store car position [x, y]
            
            # Take step in environment
            next_state, reward, cost, done, info = env.step(safe_action)
            
            # Update metrics
            episode_reward += reward
            episode_cost += cost
            if cost > 0:
                episode_collision = True
            step += 1
            
            # First render the environment (this will use the environment's built-in rendering)
            if render:
                env.render()
            
            # Update our custom plots
            # if not render and step % 1 == 0:  # Update every step
            #     # Update CBF plot
            #     steps_array = np.arange(len(cbf_values))
            #     cbf_plot.set_data(steps_array, cbf_values)
            #     ax2.set_xlim(0, max(20, step))
            #     ax2.set_ylim(min(min(cbf_values), -0.1) - 0.1, max(max(cbf_values), 0.1) + 0.1)
            #     # Update action plots if we have actions
            #     if len(nominal_actions) > 0:
            #         nominal_actions_array = np.array(nominal_actions)
            #         safe_actions_array = np.array(safe_actions)
            #         steps_array = np.arange(len(nominal_actions))
            #         # X-velocity actions
            #         nominal_x_plot.set_data(steps_array, nominal_actions_array[:, 0])
            #         safe_x_plot.set_data(steps_array, safe_actions_array[:, 0])
            #         ax3.set_xlim(0, max(20, step))
            #         ax3.set_ylim(
            #             min(nominal_actions_array[:, 0].min(), safe_actions_array[:, 0].min()) - 0.5,
            #             max(nominal_actions_array[:, 0].max(), safe_actions_array[:, 0].max()) + 0.5
            #         )
            #         # Y-velocity actions
            #         nominal_y_plot.set_data(steps_array, nominal_actions_array[:, 1])
            #         safe_y_plot.set_data(steps_array, safe_actions_array[:, 1])
            #         ax4.set_xlim(0, max(20, step))
            #         ax4.set_ylim(
            #             min(nominal_actions_array[:, 1].min(), safe_actions_array[:, 1].min()) - 0.5,
            #             max(nominal_actions_array[:, 1].max(), safe_actions_array[:, 1].max()) + 0.5
            #         )
            #     figx.canvas.draw_idle()
            #     figx.canvas.flush_events()
                
                time.sleep(render_delay)
            
            # Update state
            state = next_state
        
        # End of episode, turn off interactive mode
        # if render:
        #     plt.ioff()

            # plt.close(traj_fig)
            # plt.close(figx)
        
        
        
        end_time = time.time()  # End time of the episode
        time_taken = end_time - start_time  # Calculate the time taken
        print(f"Episode {episode} done. Time taken: {time_taken:.2f} seconds")
        # Update metrics
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        episode_successes.append(int(info["goal_reached"]))
        episode_collisions.append(int(episode_collision))
    
    # Calculate overall averages
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)

    success_rate = np.mean(episode_successes) * 100
    collision_rate = np.mean(episode_collisions) * 100

    num_batches = 5
    batch_size = len(total_rewards) // num_batches  # Ensure equal-sized batches

    # Compute batch means for rewards and costs
    reward_batch_means = [np.mean(total_rewards[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]
    cost_batch_means = [np.mean(total_costs[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]

    # Compute batch means for success and collision rates
    success_batch_means = [np.mean(episode_successes[i * batch_size: (i + 1) * batch_size]) * 100 for i in range(num_batches)]
    collision_batch_means = [np.mean(episode_collisions[i * batch_size: (i + 1) * batch_size]) * 100 for i in range(num_batches)]

    # Compute standard deviation of batch means
    std_reward = np.std(reward_batch_means, ddof=1)
    std_cost = np.std(cost_batch_means, ddof=1)
    success_std = np.std(success_batch_means, ddof=1)
    collision_std = np.std(collision_batch_means, ddof=1)
        
        # Print metrics after every episode
        # print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, " +
        #       f"Steps = {step}, Goal reached = {info['goal_reached']}, Collision = {episode_collision}")
    
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Cost: {avg_cost:.2f} ± {std_cost:.2f}")
    # print(f"Steps: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Success Rate: {success_rate:.1f}% ± {success_std:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}% ± {collision_std:.1f}%")
    
    # Return both means and standard deviations
    return {
        'reward': (avg_reward, std_reward),
        'cost': (avg_cost, std_cost),
        # 'steps': (avg_steps, std_steps),
        'success_rate': (success_rate, success_std),
        'collision_rate': (collision_rate, collision_std)
    }

def train_all_models(data):
    """Train BC models on all data and only safe data."""
    # Define model parameters
    state_dim = 4  # car (2) + obstacle (2)
    action_dim = 2
    hidden_sizes = (300,300)
    action_low = np.array([-3.0, -3.0])  # Based on environment limits
    action_high = np.array([3.0, 3.0])
    
    # Create dataloaders
    all_data_loader = create_dataloader(data, batch_size=64, only_safe=False)
    safe_data_loader = create_dataloader(data, batch_size=64, only_safe=True)
    
    # Create models
    print("\n" + "="*50)
    print("Training BC model on all data...")
    print("="*50)
    bc_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    bc_optimizer = optim.Adam(bc_model.parameters(), lr=1e-4)
    bc_trainer = BCTrainer(bc_model, bc_optimizer, device=DEVICE)
    bc_loss_history = bc_trainer.train(all_data_loader, epochs=10, log_interval=1)
    
    # Save BC model
    torch.save(bc_model.state_dict(), BC_MODEL_PATH)
    print(f"BC model saved to {BC_MODEL_PATH}")
    
    print("\n" + "="*50)
    print("Training BC-Safe model on safe data only...")
    print("="*50)
    bc_safe_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    bc_safe_optimizer = optim.Adam(bc_safe_model.parameters(), lr=1e-4)
    bc_safe_trainer = BCTrainer(bc_safe_model, bc_safe_optimizer, device=DEVICE)
    bc_safe_loss_history = bc_safe_trainer.train(safe_data_loader, epochs=10, log_interval=1)
    
    # Save BC-Safe model
    torch.save(bc_safe_model.state_dict(), BC_SAFE_MODEL_PATH)
    print(f"BC-Safe model saved to {BC_SAFE_MODEL_PATH}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(bc_loss_history, label="BC (All Data)")
    plt.plot(bc_safe_loss_history, label="BC-Safe (Safe Data Only)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("bc_training_loss.png")
    plt.close()
    
    return bc_model, bc_safe_model

def load_or_train_models(data):
    """Load models if they exist, otherwise train them."""
    # Define model parameters
    state_dim = 4  
    action_dim = 2
    hidden_sizes = (300,300)
    action_low = np.array([-3.0, -3.0])
    action_high = np.array([3.0, 3.0])
    
    # Check if models exist
    bc_model_exists = os.path.exists(BC_MODEL_PATH)
    bc_safe_model_exists = os.path.exists(BC_SAFE_MODEL_PATH)
    
    # Create models
    bc_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    bc_safe_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    # Load models if they exist, otherwise train them
    if bc_model_exists and bc_safe_model_exists  :
        # safe_data_loader = create_dataloader(data, batch_size=64, only_safe=True)
    
        print(f"Loading BC models from {BC_MODEL_PATH} and {BC_SAFE_MODEL_PATH}")
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=DEVICE))
        bc_safe_model.load_state_dict(torch.load(BC_SAFE_MODEL_PATH, map_location=DEVICE))
    else:
        print("Training new BC models...")
        bc_model, bc_safe_model = train_all_models(data)
        
    
    return bc_model, bc_safe_model

def load_cbf_model(path):
    """Load the CBF model from checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CBF checkpoint not found at {path}")
    
    print(f"Loading CBF model from {path}")
    
    # Initialize CBF model
    cbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=3, dim_hidden=128).to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    cbf.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    cbf.eval()
    
    return cbf

class CBFQPController_goal_reaching:
    """Controller that combines BC policy with CBF-QP safety filter."""
    
    def __init__(self, actor, cbf, device="cpu",env=None,alpha=0.1):
        """Initialize the controller with a policy network and CBF."""
        self.actor = actor
        self.cbf = cbf
        self.device = device
        self.dt = 0.1  # Environment timestep
        self.env=env
        self.alpha=alpha
    def get_safe_action(self, state, car_dynamics_model):###RECEIVES FULL STATE AS STATE INPUT (XCAR,YCAR,XOBST,YOBST)
        """
        Get a safe action by solving a QP that minimizes deviation from the BC policy 
        while satisfying CBF constraints using CVXPY.
        """
        # Get nominal action from behavior cloning policy
        
        ############################################################################################################################################
        nominal_action=self.env.goal_reaching_controller()##shape 2,
        # Current control barrier function value
        with torch.no_grad():
            image_numpy=get_image_from_state(state)
            image_tensor=torch.FloatTensor(image_numpy).unsqueeze(0).permute(0,3,1,2).to(DEVICE)#.unsqueeze(0)-> batch size=1 ( operation in PyTorch adds a new dimension at 0 )
            _, z = vae(image_tensor)
        # plt.ion()
        # plt.imshow(image_numpy)
        # plt.axis('off')  # Remove the axis
        # plt.show(image_numpy)
        
        h_x = self.cbf(z).item()##h_x is scalar
        # Get gradient of CBF with respect to state
        z.requires_grad_(True)
        h_x_tensor = self.cbf(z)
        gradient_B=torch.autograd.grad(h_x_tensor,z)[0]#torch.Size([1, 4])
        gradient_numpy=gradient_B.cpu().numpy()#shape  is 1,4 
        f, g = dynamics_model.dynamics.forward(z)
        cbf_lie_derivative = torch.einsum("bs,bs->b", f, gradient_B)
        right_side = cbf_lie_derivative + self.alpha * h_x_tensor.squeeze(-1)#removes the last dimension of h_x_tensor if it has a size of 1 in that dimension.###CHANGE ALPHA HERE CHANGE ALPHA HERE.  LOW ALPHA->SAFER, WORSE REWARD
        right_numpy = right_side.detach().cpu().numpy()
        grad_b_g = torch.einsum(
            'bs,bsa->ba', 
            gradient_B, 
            g.view(g.shape[0], dynamics_model.dynamics.state_dim, dynamics_model.dynamics.num_action)
        )
        grad_b_g_numpy = grad_b_g.detach().cpu().numpy()
        # Solve QP to find safe action
        u = cp.Variable(dynamics_model.dynamics.num_action)
        objective = cp.Minimize(cp.sum_squares(u - nominal_action))
        
        if (with_constraint):
            constraints = [grad_b_g_numpy @ u >= -right_numpy,
                    u<=3, u>=-3
                        ]
        else:
            constraints = [grad_b_g_numpy @ u >= -right_numpy,
                  # u<=3, u>=-3
                    ]


        prob = cp.Problem(objective, constraints)
        try:
            # Solve using OSQP solver
            prob.solve(solver=cp.OSQP)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                safe_action = u.value
            else:
                print(f"Warning: QP could not find optimal solution (status: {prob.status}), using nominal action")
                safe_action = nominal_action
        except Exception as e:
            print(f"Error in QP solver: {e}")
            safe_action = nominal_action
        ############################################################################################################################################
        ############################################################################################################################################
        # nominal_action=self.env.goal_reaching_controller()##shape 2,
        # # Current control barrier function value
        # current_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # h_x = self.cbf(current_state_tensor).item()##h_x is scalar
        # # Get gradient of CBF with respect to state
        # current_state_tensor.requires_grad_(True)
        # h_x_tensor = self.cbf(current_state_tensor)
        # gradient=torch.autograd.grad(h_x_tensor,current_state_tensor)[0]#torch.Size([1, 4])
        # gradient_numpy=gradient.cpu().numpy()#shape  is 1,4 
        # # xdot = f(x) + g(x)u where:
        # # f(x) = [0, 0, 0, 0]
        # #g = np.array([[1, 0, ], [0, 1, ], [0, 0], [0, 0]])
        # g = np.array([[1, 0, ], [0, 1, ], [0, 0], [0, 0]])   ##shape is 4,2

        # # Decision variables (control inputs)
        # u = cp.Variable(2) #shape 2,
        
        # # Objective: minimize ||u - u_nominal||^2
        # objective = cp.Minimize(cp.sum_squares(u - nominal_action))

        # # CBF constraint: grad(h)^T g(x)u >= -alpha * h(x)
        #   # CBF parameter ##if bigger more conservative, reacting strongly to small violations and maintaining a greater safety margin/can make the control action aggressive, potentially affecting performance or feasibility
        # B_x = -self.alpha * h_x  # Right hand side of the constraint
        
        
        # # Control limits
        # control_limit = 3.0
        # # Constraints
        # constraints = [
        #     gradient_numpy @ g @ u >= B_x,  # 1,4 @  4,2 @ 2, = 1,
        #     #u <= control_limit,               # Upper control limits
        #     #u >= -control_limit               # Lower control limits
        # ]
        
        # # Create and solve the problem
        # prob = cp.Problem(objective, constraints)
        
        # try:
        #     # Solve using OSQP solver
        #     prob.solve(solver=cp.OSQP)
            
        #     if prob.status == "optimal" or prob.status == "optimal_inaccurate":
        #         safe_action = u.value
        #     else:
        #         print(f"Warning: QP could not find optimal solution (status: {prob.status}), using nominal action")
        #         safe_action = nominal_action
        # except Exception as e:
        #     print(f"Error in QP solver: {e}")
        #     safe_action = nominal_action
        
        return safe_action, nominal_action

def evaluate_cbf_qp_controller_goal_reaching(controller, env, num_episodes=1, render=True, render_delay=0.000):
    """Evaluate the CBF-QP controller in the environment with side-by-side visualization."""
    total_rewards = []
    total_costs = []
    episode_successes = []  # Track success for each episode (0 or 1)
    episode_collisions = []  # Track collision for each episode (0 or 1)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        step = 0
        episode_collision = False
        
        # For tracking actions and CBF values
        nominal_actions = []
        safe_actions = []
        cbf_values = []
        car_positions = []  # Track car trajectory
        
        # Set up live plotting
        # if render:
            # Create a new figure for our custom plots
            # This is separate from the environment's rendering
            # plt.ion()  # Turn on interactive mode
            # fig = plt.figure(figsize=(10, 8))
            # fig.canvas.manager.set_window_title('CBF Analysis')

            # # Create subplots for CBF and actions
            # ax2 = fig.add_subplot(3, 1, 1)
            # ax2.set_title('CBF Values Over Time')
            # ax2.set_xlabel('Time Step')
            # ax2.set_ylabel('CBF Value')
            # ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line for CBF
            
            # # Subplot for x-velocity actions
            # ax3 = fig.add_subplot(3, 1, 2)
            # ax3.set_title('X-Velocity Actions')
            # ax3.set_xlabel('Time Step')
            # ax3.set_ylabel('Action Value')
            
            # # Subplot for y-velocity actions
            # ax4 = fig.add_subplot(3, 1, 3)
            # ax4.set_title('Y-Velocity Actions')
            # ax4.set_xlabel('Time Step')
            # ax4.set_ylabel('Action Value')
            
            # plt.tight_layout()
            
            # # Position figures side by side on screen
            # # Get screen dimensions and position windows accordingly
            # manager = plt.get_current_fig_manager()
            # if hasattr(manager, 'window'):
            #     try:
            #         # Try to position the windows - this will only work on some systems
            #         # with certain backends like TkAgg or Qt
            #         traj_fig.canvas.manager.window.wm_geometry("+0+0")
            #         fig.canvas.manager.window.wm_geometry("+600+0")
            #     except:
            #         pass  # If positioning fails, just continue

            # cbf_plot, = ax2.plot([], [], 'g-', label='CBF Value')
            
            # nominal_x_plot, = ax3.plot([], [], 'b-', label='Nominal')
            # safe_x_plot, = ax3.plot([], [], 'r-', label='Safe')
            
            # nominal_y_plot, = ax4.plot([], [], 'b-', label='Nominal')
            # safe_y_plot, = ax4.plot([], [], 'r-', label='Safe')
            
            # # Add legends
    
            # ax2.legend()
            # ax3.legend()
            # ax4.legend()

        
        while not done and step < 200:  # Cap at 200 steps to prevent infinite episodes
            # Prepare state input (append obstacle position)
            full_state = np.concatenate([state, env.obstacle_position])
            
            # Get CBF value for current state
            with torch.no_grad():
                state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(controller.device)
                image_numpy=get_image_from_state(full_state)
                image_tensor=torch.FloatTensor(image_numpy).unsqueeze(0).permute(0,3,1,2).to(DEVICE)#.unsqueeze(0)-> batch size=1 ( operation in PyTorch adds a new dimension at 0 )
                _, z = vae(image_tensor)
                cbf_value = controller.cbf(z).item()
                cbf_values.append(cbf_value)
            
            # Get safe action from CBF-QP controller
            safe_action, nominal_action = controller.get_safe_action(full_state, env)
            
            # Store actions and positions for visualization
            nominal_actions.append(nominal_action)
            safe_actions.append(safe_action)
            car_positions.append(state[:2])  # Store car position [x, y]
            
            # Take step in environment
            next_state, reward, cost, done, info = env.step(safe_action)
            
            # Update metrics
            episode_reward += reward
            episode_cost += cost
            if cost > 0:
                episode_collision = True
            step += 1
            
            # First render the environment (this will use the environment's built-in rendering)
            if render:
                env.render()
            
            # # Update our custom plots
            # if render and step % 1 == 0:  # Update every step
            #     car_positions_array = np.array(car_positions)

            #     # Update CBF plot
            #     steps_array = np.arange(len(cbf_values))
            #     cbf_plot.set_data(steps_array, cbf_values)
            #     ax2.set_xlim(0, max(20, step))
            #     ax2.set_ylim(min(min(cbf_values), -0.1) - 0.1, max(max(cbf_values), 0.1) + 0.1)
                
            #     # Update action plots if we have actions
            #     if len(nominal_actions) > 0:
            #         nominal_actions_array = np.array(nominal_actions)
            #         safe_actions_array = np.array(safe_actions)
            #         steps_array = np.arange(len(nominal_actions))
                    
            #         # X-velocity actions
            #         nominal_x_plot.set_data(steps_array, nominal_actions_array[:, 0])
            #         safe_x_plot.set_data(steps_array, safe_actions_array[:, 0])
            #         ax3.set_xlim(0, max(20, step))
            #         ax3.set_ylim(
            #             min(nominal_actions_array[:, 0].min(), safe_actions_array[:, 0].min()) - 0.5,
            #             max(nominal_actions_array[:, 0].max(), safe_actions_array[:, 0].max()) + 0.5
            #         )
                    
            #         # Y-velocity actions
            #         nominal_y_plot.set_data(steps_array, nominal_actions_array[:, 1])
            #         safe_y_plot.set_data(steps_array, safe_actions_array[:, 1])
            #         ax4.set_xlim(0, max(20, step))
            #         ax4.set_ylim(
            #             min(nominal_actions_array[:, 1].min(), safe_actions_array[:, 1].min()) - 0.5,
            #             max(nominal_actions_array[:, 1].max(), safe_actions_array[:, 1].max()) + 0.5
            #         )

            #     fig.canvas.draw_idle()
            #     fig.canvas.flush_events()
                
            #     time.sleep(render_delay)
            
            # Update state
            state = next_state
        
        # End of episode, turn off interactive mode
        if render:
            plt.ioff()
            
            # Save the final figures
            # traj_fig.savefig(f"trajectory_episode_{episode+1}.png")
            # fig.savefig(f"cbf_analysis_episode_{episode+1}.png")
            
            # plt.close(traj_fig)
            # plt.close(fig)
        
        # Update metrics
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        episode_successes.append(int(info["goal_reached"]))
        episode_collisions.append(int(episode_collision))
    
    # Calculate overall averages
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)

    success_rate = np.mean(episode_successes) * 100
    collision_rate = np.mean(episode_collisions) * 100

    num_batches = 5
    batch_size = len(total_rewards) // num_batches  # Ensure equal-sized batches

    # Compute batch means for rewards and costs
    reward_batch_means = [np.mean(total_rewards[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]
    cost_batch_means = [np.mean(total_costs[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]

    # Compute batch means for success and collision rates
    success_batch_means = [np.mean(episode_successes[i * batch_size: (i + 1) * batch_size]) * 100 for i in range(num_batches)]
    collision_batch_means = [np.mean(episode_collisions[i * batch_size: (i + 1) * batch_size]) * 100 for i in range(num_batches)]

    # Compute standard deviation of batch means
    std_reward = np.std(reward_batch_means, ddof=1)
    std_cost = np.std(cost_batch_means, ddof=1)
    success_std = np.std(success_batch_means, ddof=1)
    collision_std = np.std(collision_batch_means, ddof=1)
        
        # Print metrics after every episode
        # print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, " +
        #       f"Steps = {step}, Goal reached = {info['goal_reached']}, Collision = {episode_collision}")
    
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Cost: {avg_cost:.2f} ± {std_cost:.2f}")
    # print(f"Steps: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Success Rate: {success_rate:.1f}% ± {success_std:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}% ± {collision_std:.1f}%")
    
    # Return both means and standard deviations
    return {
        'reward': (avg_reward, std_reward),
        'cost': (avg_cost, std_cost),
        # 'steps': (avg_steps, std_steps),
        'success_rate': (success_rate, success_std),
        'collision_rate': (collision_rate, collision_std)
    }
def main():
    """Main function to train, evaluate, and visualize all models."""
    # Set random seed for reproducibility
    seed=3
    seed_everything(seed)

    print(f"Using device: {DEVICE}")
    
    # Load dataset
    data = load_dataset()

    # Load or train Behavior Cloning (BC) models
    bc_model, bc_safe_model = load_or_train_models(data)
    
    # Create environment for testing
    ###MAKE IT EASIER TO DETERMINE IF WE REACHED OBJECTIVE. REACHED EPSILON IS 1 INSTEAD OF 0.5
    env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=GOAL_POSITION, obstacle_radius=4.0,reached_epsilon=2)
    
    # # Load the CBF model
    cbf_model = load_cbf_model(CBF_CHECKPOINT)
    ccbf_model = load_cbf_model(CCBF_CHECKPOINT)
    print("loaded cbf and ccbf\n")

    # Initialize metrics for each model
    
    bc_metrics = None
    bc_safe_metrics = None
    cbf_qp_metrics = None
    
    
    
    
    alpha=1
    num_episodes=50
    
    
    # # Evaluate BC model (on all data)
    print("\n" + "="*50)
    print("Evaluating BC model (all data)...")
    print("="*50)
    bc_metrics = evaluate_model(bc_model, env, num_episodes=num_episodes, render=False, render_delay=0.0000000)
    
    # Evaluate BC-Safe model (on safe data only)
    print("\n" + "="*50)
    print("Evaluating BC-Safe model (safe data only)...")
    print("="*50)
    bc_safe_metrics = evaluate_model(bc_safe_model, env, num_episodes=num_episodes, render=False, render_delay=0.0000000)



    # # # # CBF-QP controller for goal-reaching with behavior cloning model
    cbf_qp_controller_goal_reaching = CBFQPController_goal_reaching(bc_model, cbf_model, device=DEVICE, env=env,alpha=alpha)
    print("\n" + "="*50)
    print("Evaluating nominal CBF-QP controller (goal-reaching)...")
    print("="*50)
    cbf_qp_metrics_goal_reaching = evaluate_cbf_qp_controller(cbf_qp_controller_goal_reaching, env, num_episodes=num_episodes, render=False, render_delay=0.0000000)


    # CBF-QP controller using BC model and CBF
    cbf_qp_controller_bc = CBFQPController_BC(bc_model, cbf_model, device=DEVICE,alpha=alpha)
    print("\n" + "="*50)
    print("Evaluating CBF-QP controller (BC + CBF)...")
    print("="*50)
    cbf_qp_metrics_bc = evaluate_cbf_qp_controller(cbf_qp_controller_bc, env, num_episodes=num_episodes, render=False, render_delay=0.0000)

    # NEW: CBF-QP controller using BC-Safe model and CBF
    cbf_qp_controller_bc_safe = CBFQPController_BC(bc_safe_model, cbf_model, device=DEVICE,alpha=alpha)
    print("\n" + "="*50)
    print("Evaluating CBF-QP controller (BC-Safe + CBF)...")
    print("="*50)
    cbf_qp_metrics_bc_safe = evaluate_cbf_qp_controller(cbf_qp_controller_bc_safe, env, num_episodes=num_episodes, render=False, render_delay=0.0000)






    
    # CBF-QP controller for goal-reaching with behavior cloning model
    ccbf_qp_controller_goal_reaching = CBFQPController_goal_reaching(bc_model, ccbf_model, device=DEVICE, env=env,alpha=alpha)
    print("\n" + "="*50)
    print("Evaluating nominal CCBF-QP controller (goal-reaching)...")
    print("="*50)
    ccbf_qp_metrics_goal_reaching = evaluate_cbf_qp_controller(ccbf_qp_controller_goal_reaching, env, num_episodes=num_episodes, render=False, render_delay=0.0000)

    # CBF-QP controller using BC model and CBF
    ccbf_qp_controller_bc = CBFQPController_BC(bc_model, ccbf_model, device=DEVICE,alpha=alpha)
    print("\n" + "="*50)
    print("Evaluating CCBF-QP controller (BC + CCBF)...")
    print("="*50)
    ccbf_qp_metrics_bc = evaluate_cbf_qp_controller(ccbf_qp_controller_bc, env, num_episodes=num_episodes, render=False, render_delay=0.0000)

    # NEW: CBF-QP controller using BC-Safe model and CBF
    ccbf_qp_controller_bc_safe = CBFQPController_BC(bc_safe_model, ccbf_model, device=DEVICE,alpha=alpha)
    print("\n" + "="*50)
    print("Evaluating CCBF-QP controller (BC-Safe + CCBF)...")
    print("="*50)
    ccbf_qp_metrics_bc_safe = evaluate_cbf_qp_controller(ccbf_qp_controller_bc_safe, env, num_episodes=num_episodes, render=False, render_delay=0.0000)




    # Comparison of all methods
    print("\n" + "="*50)
    print("COMPARISON OF ALL METHODS")
    print("="*50)
    
    # Prepare the methods and metrics for comparison
# Prepare the methods and metrics for comparison
    methods = [
        "BC (All Data)", 
        "BC-Safe (Safe Data Only)", 
        
        "CBF-QP (Goal Reaching)",
        "CBF-QP (BC + CBF)", 
        "CBF-QP (BC-Safe + CBF)", 

        "CCBF-QP (Goal Reaching)", 
        "CCBF-QP (BC + CBF)", 
        "CCBF-QP (BC-Safe + CBF)"
    ]

    all_metrics = [
        bc_metrics, 
        bc_safe_metrics, 
        
        cbf_qp_metrics_goal_reaching, 
        cbf_qp_metrics_bc, 
        cbf_qp_metrics_bc_safe, 

        ccbf_qp_metrics_goal_reaching, 
        ccbf_qp_metrics_bc, 
        ccbf_qp_metrics_bc_safe
    ]
    # Print the headers for the comparison table
    print(f"{'Method':<35} {'Reward (avg±std)':<20} {'Cost (avg±std)':<20} {'Success % (avg±std)':<20} {'Collision % (avg±std)':<20}")
    print("-" * 115)

    # Print the metrics for each method
    print("CBF_CHECKPOINT: ", CBF_CHECKPOINT)
    print("CCBF_CHECKPOINT: ", CCBF_CHECKPOINT)
    for method, metrics in zip(methods, all_metrics):
        if metrics is not None:
            reward_tuple = metrics['reward']
            cost_tuple = metrics['cost']
            success_tuple = metrics['success_rate']
            collision_tuple = metrics['collision_rate']
            
            print(f"{method:<35} {reward_tuple[0]:>6.2f}±{reward_tuple[1]:<6.2f} {cost_tuple[0]:>6.2f}±{cost_tuple[1]:<6.2f} {success_tuple[0]:>6.1f}±{success_tuple[1]:<6.1f} {collision_tuple[0]:>6.1f}±{collision_tuple[1]:<6.1f}")
        else:
            print(f"{method:<35} {'N/A':<20} {'N/A':<20} {'N/A':<20} {'N/A':<20}")

    # Open the file in append mode
    with open("render.txt", "a") as f:
        # Write checkpoint information
        f.write(f"num episodes: {num_episodes}\n")
        f.write(f"seed {seed}\n")
        f.write(f"with constraint {with_constraint}\n")
        f.write(f"CBF_CHECKPOINT alpha_{alpha}: {CBF_CHECKPOINT}\n")
        f.write(f"CCBF_CHECKPOINT alpha_{alpha}: {CCBF_CHECKPOINT}\n")
        
        # Write header
        f.write(f"{'Method':<35} {'Reward (avg±std)':<20} {'Cost (avg±std)':<20} {'Success % (avg±std)':<20} {'Collision % (avg±std)':<20}\n")
        
        # Write the metrics for each method
        for method, metrics in zip(methods, all_metrics):
            if metrics is not None:
                reward_tuple = metrics['reward']
                cost_tuple = metrics['cost']
                success_tuple = metrics['success_rate']
                collision_tuple = metrics['collision_rate']
                
                result_line = f"{method:<35} {reward_tuple[0]:>6.2f}±{reward_tuple[1]:<6.2f} {cost_tuple[0]:>6.2f}±{cost_tuple[1]:<6.2f} {success_tuple[0]:>6.1f}±{success_tuple[1]:<6.1f} {collision_tuple[0]:>6.1f}±{collision_tuple[1]:<6.1f}\n"
            else:
                result_line = f"{method:<35} {'N/A':<20} {'N/A':<20} {'N/A':<20} {'N/A':<20}\n"
            
            f.write(result_line)  # Append to file

if __name__ == "__main__":
    main()

