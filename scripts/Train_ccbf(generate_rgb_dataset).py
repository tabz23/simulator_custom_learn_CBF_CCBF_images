import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import wandb  # Import wandb
from PIL import Image
from io import BytesIO
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# from modules.dataset import *
from modules.network import *
# from modules.trainer_ccbf import *
from envs.car import *
from matplotlib.collections import LineCollection
import random
DATASET_PATH = "safe_rl_dataset_images_ALLUNSAFE_big_obstacle_3000traj.npz"  # Updated path to store/load dataset
goal_position=np.array([20, 21])


rng = random.Random()  # This uses a new random state
random_value = rng.randint(100, 999)
def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
def visualize_trajectories(data, num_trajectories=10):
    """Visualize selected trajectories from the dataset with improved clarity."""
    # Determine episode boundaries
    if "episode_starts" in data and "episode_lengths" in data:
        # Use pre-computed episode boundaries if available
        episode_starts = data["episode_starts"]
        episode_lengths = data["episode_lengths"]
    else:
        # Otherwise reconstruct from done flags
        dones = data["dones"]
        episode_starts = [0]
        episode_lengths = []
        
        current_length = 0
        for i in range(len(dones)):
            current_length += 1
            if dones[i] == 1.0:
                episode_lengths.append(current_length)
                if i < len(dones) - 1:  # Don't add start if last episode
                    episode_starts.append(i + 1)
                current_length = 0
    
    # Select random trajectories to visualize
    import random
    num_episodes = len(episode_starts)
    print(f"Dataset contains {num_episodes} trajectories")
    
    selected_indices = random.sample(range(num_episodes), min(num_trajectories, num_episodes))
    
    # Create figure for visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Add a colormap for trajectory visualization
    cmap = plt.cm.viridis
    
    for plot_idx, traj_idx in enumerate(selected_indices):
        # Get start and length of this trajectory
        start_idx = episode_starts[traj_idx]
        length = episode_lengths[traj_idx] if traj_idx < len(episode_lengths) else len(data["states"]) - start_idx
        
        # Extract trajectory data
        states = data["states"][start_idx:start_idx + length]
        
        # Get car positions and obstacle positions
        car_positions = states[:, :2]  # First 2 elements are car position
        obstacle_position = states[0, 2:4]  # Elements 2-3 are obstacle position (constant within trajectory)
        obstacle_radius = 4.0  # This appears to be constant in the dataset
        
        # Create subplot
        ax = fig.add_subplot(3, 4, plot_idx + 1)
        
        # Plot trajectory with color gradient to show direction
        points = np.array([car_positions[:, 0], car_positions[:, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, length)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.8)

        lc.set_array(np.arange(length))
        line = ax.add_collection(lc)
        
        # Mark start and end points
        ax.scatter(car_positions[0, 0], car_positions[0, 1], color='green', s=50, zorder=5, label="Start")
        ax.scatter(car_positions[-1, 0], car_positions[-1, 1], color='red', s=50, zorder=5, label="End")
        
        # Plot obstacle
        obstacle = plt.Circle((obstacle_position[0], obstacle_position[1]), obstacle_radius, 
                             color='gray', alpha=0.5, label="Obstacle")
        ax.add_patch(obstacle)
        
        # Plot goal
        goal_size = 1.5  # Size of goal region
        goal_rect = plt.Rectangle((goal_position[0] - goal_size, goal_position[1] - goal_size), 
                                 width=2*goal_size, height=2*goal_size, 
                                 color='blue', alpha=0.3, label="Goal")
        ax.add_patch(goal_rect)
        
        # Set plot properties
        ax.set_xlim(-20, 25)
        ax.set_ylim(-20, 25)
        ax.set_aspect('equal')
        ax.set_title(f"Trajectory {traj_idx} (Length: {length})")
        
        # Add legend to first plot only
        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.suptitle("Sample Trajectories from Dataset", fontsize=16, y=1.02)
    
    # Save figure and log to wandb
    plt.savefig("trajectories_visualization.png", bbox_inches='tight')
    wandb.log({"trajectories_visualization": wandb.Image("trajectories_visualization.png")})
    plt.show()

def visualize_dataset_statistics(data):
    """Visualize comprehensive dataset statistics."""
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    costs = data["costs"]
    dones = data["dones"]
    
    # Calculate statistics
    total_transitions = len(states)
    safe_transitions = np.sum(costs == 0)
    unsafe_transitions = np.sum(costs > 0)
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # 1. Plot reward distribution
    plt.subplot(2, 3, 1)
    plt.hist(rewards, bins=50, color='blue', alpha=0.7)
    plt.title("Reward Distribution")
    plt.xlabel("Reward Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    # 2. Plot cost distribution
    plt.subplot(2, 3, 2)
    labels = ['Safe (cost=0)', 'Unsafe (cost>0)']
    counts = [safe_transitions, unsafe_transitions]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    plt.axis('equal')
    plt.title("Safety Distribution")
    
    # 3. Plot car positions
    plt.subplot(2, 3, 3)
    car_positions = states[:, :2]
    plt.hexbin(
    car_positions[:, 0], car_positions[:, 1], 
    gridsize=50, cmap='CMRmap_r', 
    extent=[np.percentile(car_positions[:, 0], 2), np.percentile(car_positions[:, 0], 99), 
            np.percentile(car_positions[:, 1], 2), np.percentile(car_positions[:, 1], 99)]##CHANGED HERE THE PERCENTILE FOR CLEANER PLOT
)

    plt.colorbar(label="Density")
    plt.title("Car Position Distribution")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    # 4. Plot action distribution
    plt.subplot(2, 3, 4)
    plt.hist2d(actions[:, 0], actions[:, 1], bins=30, cmap='Greens')
    plt.colorbar(label="Frequency")
    plt.title("Action Distribution")
    plt.xlabel("Action Dimension 1")
    plt.ylabel("Action Dimension 2")
    
    # 5. Plot obstacle positions
    plt.subplot(2, 3, 5)
    obstacle_positions = states[:, 2:4]
    unique_obstacles = np.unique(obstacle_positions, axis=0)
    plt.scatter(unique_obstacles[:, 0], unique_obstacles[:, 1], 
               c=range(len(unique_obstacles)), cmap='rainbow', s=100, alpha=0.7)
    plt.title(f"Unique Obstacle Positions\n({len(unique_obstacles)} positions)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    # 6. Plot episodes information
    plt.subplot(2, 3, 6)
    episode_ends = np.where(dones == 1)[0] + 1
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts
    
    plt.hist(episode_lengths, bins=20, color='purple', alpha=0.7)
    plt.title("Episode Length Distribution")
    plt.xlabel("Episode Length (steps)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    # Add overall title and summary
    plt.suptitle("Dataset Statistics Overview", fontsize=16, y=0.98)
    plt.figtext(0.5, 0.01, f"Total Transitions: {total_transitions} | Episodes: {len(episode_lengths)} | Avg Episode Length: {np.mean(episode_lengths):.1f}", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure and log to wandb
    plt.savefig("dataset_statistics.png", bbox_inches='tight')
    wandb.log({"dataset_statistics": wandb.Image("dataset_statistics.png")})
    
    # Log all key statistics to wandb as separate metrics
    wandb.log({
        "total_transitions": total_transitions,
        "safe_transitions": safe_transitions,
        "unsafe_transitions": unsafe_transitions,
        "safe_percentage": (safe_transitions / total_transitions) * 100,
        "total_episodes": len(episode_lengths),
        "average_episode_length": float(np.mean(episode_lengths)),
        "median_episode_length": float(np.median(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "average_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "unique_obstacle_positions": len(unique_obstacles)
    })
    
    plt.show()
# def generate_dataset_images(num_trajectories=1):
#     """Generates and saves a dataset of trajectories with rewards, costs, and clean RGB images."""
#     print(f"Generating dataset with {num_trajectories} trajectories...")
#     wandb.log({"dataset_generation_started": True, "num_trajectories": num_trajectories})

#     state_list, action_list, next_state_list = [], [], []
#     reward_list, cost_list, done_list = [], [], []
#     states_rgb_list, states_rgb_next_list = [], []  # New list to store RGB images
    
#     # Track episode boundaries for visualization
#     episode_starts = [0]
#     episode_lengths = []

#     # Initialize with obstacle radius 0
#     r = 0.00001###
    
#     env = DubinsCarEnv(max_velocity=3.0, dt=0.1, obstacle_radius=r, goal_position=goal_position)
#     state = env.reset()
    
#     # Setup figure for rendering images without displaying them
#     fig, ax = plt.subplots(figsize=(4,4), facecolor='white')
#     plt.ion()  # Turn on interactive mode
    
#     trajectory_count = 0
#     step_count = 0
#     current_episode_length = 0
    
#     while trajectory_count < num_trajectories:
#         # Increase obstacle radius every while, this takes effect in the environment generating trajectories
#         if (step_count % 40000) == 0 and step_count > 0:
#             r = min(r + 0.05, 2)  # Cap radius at 5 to prevent impossible scenarios. changed this a bit from code that has no images
#             '''
#             db: Run summary:
#             with if (step_count % 40000)  r = min(r + 0.05, 2) 
# wandb:                 average_cost 0.0528
# wandb:       average_episode_length 138.689
# wandb:               average_reward 0.60875
# wandb:       completed_trajectories 2000
# wandb:                dataset_found False
# wandb: dataset_generation_completed True
# wandb:  dataset_generation_progress 1.0
# wandb:   dataset_generation_started True
# wandb:               dataset_loaded True
# wandb:                 dataset_path safe_rl_dataset_imag...
# wandb:                 dataset_size 277378
# wandb:             num_trajectories 2000
# wandb:              obstacle_radius 0.30001
# wandb:     safe_transitions_percent 94.7202
# wandb:           total_trajectories 2000
# wandb:             trajectory_count 173
# '''
#             env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=goal_position, obstacle_radius=r)
#             state = env.reset()
#             print(f"traj {trajectory_count}: Increased obstacle radius to {r}")
#             wandb.log({"obstacle_radius": r, "trajectory_count": trajectory_count})
#         # print(r)
#         # env.render() ##NOTE THAT RENDER DEPENDS ON ENV.OBSTACLE_RADIUS. AND NOTE THAT IF R=0 PARAM GIVEN TO ENV THEN IT SETS R TO 4
#         # Calculate naive controller action
#         velocity_command = env.goal_reaching_controller()
        
#         # Apply CBF to make the action safe
#         velocity_command_safe = env.goal_reaching__safe_controller(velocity_command)
        
#         # Render the current state to an image without displaying
#         ax.clear()
#         ax.set_xlim(-30, 30)
#         ax.set_ylim(-30, 30)
#         ax.set_facecolor('white')  # White background
        
#         # Draw car as a small blue dot
#         ax.scatter(state[0], state[1], color='blue', s=5)
        
#         # Draw obstacle as a green circle
#         obstacle_circle = plt.Circle(
#             (env.obstacle_position[0], env.obstacle_position[1]), 
#             4, ##hardcode circle radius as 4 because in the env.onstacle_radius will not be 4 as we are vaying it for generating trajectories using cbf
#             color='green', 
#             alpha=0.5
#         )
#         ax.add_artist(obstacle_circle)
        
#         # Remove axis and white space
#         ax.set_axis_off()
#         plt.tight_layout(pad=0)
        
#         # Save figure to memory buffer
#         buf = BytesIO()
#         fig.savefig(buf, format='png', dpi=16, bbox_inches='tight', pad_inches=0, transparent=False)
#         buf.seek(0)
        
#         # Convert to RGB array
#         img = Image.open(buf)
#         img_array = np.array(img.convert('RGB'))
#         states_rgb_list.append(img_array)
        
#         # Take a step with the safe action
#         state_next, reward, cost, done, info = env.step(velocity_command_safe)
        
        
#         # Render the next state to an image (after taking the step)
#         ax.clear()
#         ax.set_xlim(-30, 30)
#         ax.set_ylim(-30, 30)
#         ax.set_facecolor('white')  # White background
        
#         # Draw car at new position
#         ax.scatter(state_next[0], state_next[1], color='blue', s=5)
        
#         # Draw obstacle at same position
#         obstacle_circle = plt.Circle(
#             (env.obstacle_position[0], env.obstacle_position[1]), 
#             4, ##hardcode circle radius as 4 because in the env.onstacle_radius will not be 4 as we are vaying it for generating trajectories using cbf
#             color='green', 
#             alpha=0.5
#         )
#         ax.add_artist(obstacle_circle)
        
#         # Remove axis and white space
#         ax.set_axis_off()
#         plt.tight_layout(pad=0)
        
#         # Save figure to memory buffer (after step)
#         buf = BytesIO()
#         fig.savefig(buf, format='png', dpi=16, bbox_inches='tight', pad_inches=0, transparent=False)
#         buf.seek(0)
        
#         # Convert to RGB array
#         img = Image.open(buf)
#         img_array_next = np.array(img.convert('RGB'))
        
#         states_rgb_next_list.append(img_array_next)  # Store as `states_rgb_next

#         # Store data
#         state_list.append(np.concatenate([state, env.obstacle_position]))  # STATE IN DATASET HAS 4 DIMENSIONS but simulator returns only xcar, ycar
#         action_list.append(velocity_command_safe)
#         next_state_list.append(np.concatenate([state_next, env.obstacle_position]))
#         reward_list.append(reward)
#         cost_list.append(cost)
#         done_list.append(float(done))
        
#         # Update state
#         state = state_next
#         step_count += 1
#         current_episode_length += 1
        
#         # If episode is done, reset environment and track episode boundary
#         if done:
#             trajectory_count += 1
#             episode_lengths.append(current_episode_length)
#             if trajectory_count < num_trajectories:  # Only add new start if not the last trajectory
#                 episode_starts.append(step_count)
#             current_episode_length = 0
            
#             if trajectory_count % 100 == 0:
#                 print(f"Completed {trajectory_count} trajectories")
#                 wandb.log({
#                     "dataset_generation_progress": trajectory_count / num_trajectories,
#                     "completed_trajectories": trajectory_count,
#                     "average_episode_length": np.mean(episode_lengths)
#                 })
#             state = env.reset()
    
#     plt.close(fig)  # Close the figure when done
    
#     # Convert states_rgb_list to a numpy array
#     states_rgb = np.array(states_rgb_list)
#     states_rgb_next = np.array(states_rgb_next_list)
    
#     # Convert to numpy arrays and save
#     np.savez(
#         DATASET_PATH, 
#         states=np.array(state_list), 
#         states_rgb=states_rgb,  # Save the RGB images
#         states_rgb_next=states_rgb_next,
#         actions=np.array(action_list), 
#         next_states=np.array(next_state_list),
#         rewards=np.array(reward_list),
#         costs=np.array(cost_list),
#         dones=np.array(done_list),
#         episode_starts=np.array(episode_starts),
#         episode_lengths=np.array(episode_lengths),
#     )
#     print(f"Dataset saved to {DATASET_PATH}")
#     print(f"Dataset shape: {len(state_list)} transitions")
#     print(f"RGB images shape: {states_rgb.shape}")
#     print(f"RGB next images shape: {states_rgb_next.shape}")
#     print(f"Total trajectories: {len(episode_starts)}")
#     print(f"Average cost per transition: {np.mean(cost_list):.4f}")
          
#     # Log dataset statistics to wandb
#     wandb.log({
#         "dataset_generation_completed": True,
#         "dataset_size": len(state_list),
#         "total_trajectories": len(episode_starts),
#         "average_cost": float(np.mean(cost_list)),
#         "safe_transitions_percent": float(np.mean(np.array(cost_list) == 0) * 100),
#         "average_reward": float(np.mean(reward_list)),
#         "average_episode_length": float(np.mean(episode_lengths))
#     })
    
#     return DATASET_PATH


###NOW DOESNT USE CBF TO GENERATE THE DATA.
###NOW DOESNT USE CBF TO GENERATE THE DATA.
###NOW DOESNT USE CBF TO GENERATE THE DATA.
###NOW DOESNT USE CBF TO GENERATE THE DATA.
###NOW DOESNT USE CBF TO GENERATE THE DATA.
###NOW DOESNT USE CBF TO GENERATE THE DATA.
def generate_dataset_images(num_trajectories=1):
    """Generates and saves a dataset of trajectories with rewards, costs, and clean RGB images."""
    print(f"Generating dataset with {num_trajectories} trajectories...")
    wandb.log({"dataset_generation_started": True, "num_trajectories": num_trajectories})

    state_list, action_list, next_state_list = [], [], []
    reward_list, cost_list, done_list = [], [], []
    states_rgb_list, states_rgb_next_list = [], []  # New list to store RGB images
    
    # Track episode boundaries for visualization
    episode_starts = [0]
    episode_lengths = []

    # Initialize with obstacle radius 0
    r = 0###
    
    env = DubinsCarEnv(max_velocity=3.0, dt=0.1, obstacle_radius=r, goal_position=goal_position)
    state = env.reset()
    
    # Setup figure for rendering images without displaying them
    fig, ax = plt.subplots(figsize=(4,4), facecolor='white')
    plt.ion()  # Turn on interactive mode
    
    trajectory_count = 0
    step_count = 0
    current_episode_length = 0
    ##IF RADIUS IS 0 THEN ENVIRONMENT DOESNT USE CBF AND goal_reaching__safe_controller RETURNS NAIVE COMMAND
    while trajectory_count < num_trajectories:
        # Increase obstacle radius every while, this takes effect in the environment generating trajectories
        if (step_count %80000) == 0 and step_count > 0:#traj 453: Increased obstacle radius to 0.5000 MB deduped)
            '''
            dataset not found at safe_rl_dataset_images_ALLUNSAFE_big_obstacle.npz
            Generating new dataset...
            Generating dataset with 500 trajectories...
            Completed 100 trajectories MB uploaded (0.000 MB deduped)
            Completed 200 trajectories MB uploaded (0.000 MB deduped)
            Completed 300 trajectories MB uploaded (0.000 MB deduped)
            traj 308: Increased obstacle radius to 0.4000 MB deduped)
            Completed 400 trajectories MB uploaded (0.000 MB deduped)
            Completed 500 trajectories MB uploaded (0.000 MB deduped)
            Dataset saved to safe_rl_dataset_images_ALLUNSAFE_big_obstacle.npz
            Dataset shape: 66725 transitions
            RGB images shape: (66725, 64, 64, 3)
            Total trajectories: 500
            Average cost per transition: 0.1113
            '''
            '''
            dataset at safe_rl_dataset_images_ALLUNSAFE_big_obstacle_5000traj
            traj 2905: Increased obstacle radius to 1.0
            Completed 3000 trajectories
            Completed 3100 trajectories
            Completed 3200 trajectories
            Completed 3300 trajectories
            Completed 3400 trajectories
            traj 3474: Increased obstacle radius to 1.2
            Completed 3500 trajectories
            Completed 3600 trajectories
            Completed 3700 trajectories
            Completed 3800 trajectories
            Completed 3900 trajectories
            Completed 4000 trajectories
            traj 4036: Increased obstacle radius to 1.4
            Completed 4100 trajectories
            Completed 4200 trajectories
            Completed 4300 trajectories
            Completed 4400 trajectories
            Completed 4500 trajectories
            traj 4594: Increased obstacle radius to 1.5999999999999999
            Completed 4600 trajectories
            Completed 4700 trajectories
            Completed 4800 trajectories
            Completed 4900 trajectories
            Completed 5000 trajectories
            Dataset saved to safe_rl_dataset_images_ALLUNSAFE_big_obstacle_5000traj.npz
            Dataset shape: 699134 transitions
            RGB images shape: (699134, 64, 64, 3)
            Total trajectories: 5000
            Average cost per transition: 0.0536
            '''
            r = min(r + 0.2, 4.2)  # Cap radius at 5 to prevent impossible scenarios. changed this a bit from code that has no images

            env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=goal_position, obstacle_radius=r)
            state = env.reset()
            print(f"traj {trajectory_count}: Increased obstacle radius to {r}")
            wandb.log({"obstacle_radius": r, "trajectory_count": trajectory_count})
        # print(r)
        # env.render() ##NOTE THAT RENDER DEPENDS ON ENV.OBSTACLE_RADIUS. AND NOTE THAT IF R=0 PARAM GIVEN TO ENV THEN IT SETS R TO 4
        # Calculate naive controller action
        velocity_command = env.goal_reaching_controller()
        
        # Apply CBF to make the action safe
        velocity_command_safe = env.goal_reaching__safe_controller(velocity_command)
        # print("velocity_command",velocity_command)
        # print(env.obstacle_radius)
        # print("velocity_command_safe",velocity_command_safe)
        # Render the current state to an image without displaying
        ax.clear()
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_facecolor('white')  # White background
        
        # Draw car as a small blue dot
        
        
        # Draw obstacle as a green circle
        obstacle_circle = plt.Circle(
            (env.obstacle_position[0], env.obstacle_position[1]), 
            4, ##hardcode circle radius as 4 because in the env.onstacle_radius will not be 4 as we are vaying it for generating trajectories using cbf
            color='brown', 
            alpha=1
        )
        ax.add_artist(obstacle_circle)
        ax.scatter(state[0], state[1], color='black', s=300)
        # Remove axis and white space
        ax.set_axis_off()
        plt.tight_layout(pad=0)
        
        # Save figure to memory buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=16, bbox_inches='tight', pad_inches=0, transparent=False)
        buf.seek(0)
        
        # Convert to RGB array
        img = Image.open(buf)
        img_array = np.array(img.convert('RGB'))
        states_rgb_list.append(img_array)
        
        # Take a step with the safe action
        state_next, reward, cost, done, info = env.step(velocity_command_safe)
        
        
        
        
        # Convert to RGB array
        img = Image.open(buf)
        # img_array_next = np.array(img.convert('RGB'))
        
        # states_rgb_next_list.append(img_array_next)  # Store as `states_rgb_next

        # Store data
        state_list.append(np.concatenate([state, env.obstacle_position]))  # STATE IN DATASET HAS 4 DIMENSIONS but simulator returns only xcar, ycar
        action_list.append(velocity_command_safe)
        next_state_list.append(np.concatenate([state_next, env.obstacle_position]))
        reward_list.append(reward)
        cost_list.append(cost)
        done_list.append(float(done))
        
        # Update state
        state = state_next
        step_count += 1
        current_episode_length += 1
        
        # If episode is done, reset environment and track episode boundary
        if done:
            trajectory_count += 1
            episode_lengths.append(current_episode_length)
            if trajectory_count < num_trajectories:  # Only add new start if not the last trajectory
                episode_starts.append(step_count)
            current_episode_length = 0
            
            if trajectory_count % 100 == 0:
                print(f"Completed {trajectory_count} trajectories")
                wandb.log({
                    "dataset_generation_progress": trajectory_count / num_trajectories,
                    "completed_trajectories": trajectory_count,
                    "average_episode_length": np.mean(episode_lengths)
                })
            state = env.reset()
    
    plt.close(fig)  # Close the figure when done
    
    # Convert states_rgb_list to a numpy array
    states_rgb = np.array(states_rgb_list)
    # states_rgb_next = np.array(states_rgb_next_list)
    
    # Convert to numpy arrays and save
    np.savez(
        DATASET_PATH, 
        states=np.array(state_list), 
        states_rgb=states_rgb,  # Save the RGB images
        # states_rgb_next=states_rgb_next,
        actions=np.array(action_list), 
        next_states=np.array(next_state_list),
        rewards=np.array(reward_list),
        costs=np.array(cost_list),
        dones=np.array(done_list),
        episode_starts=np.array(episode_starts),
        episode_lengths=np.array(episode_lengths),
    )
    print(f"Dataset saved to {DATASET_PATH}")
    print(f"Dataset shape: {len(state_list)} transitions")
    print(f"RGB images shape: {states_rgb.shape}")
    # print(f"RGB next images shape: {states_rgb_next.shape}")
    print(f"Total trajectories: {len(episode_starts)}")
    print(f"Average cost per transition: {np.mean(cost_list):.4f}")
          
    # Log dataset statistics to wandb
    wandb.log({
        "dataset_generation_completed": True,
        "dataset_size": len(state_list),
        "total_trajectories": len(episode_starts),
        "average_cost": float(np.mean(cost_list)),
        "safe_transitions_percent": float(np.mean(np.array(cost_list) == 0) * 100),
        "average_reward": float(np.mean(reward_list)),
        "average_episode_length": float(np.mean(episode_lengths))
    })
    
    return DATASET_PATH

def load_dataset():
    """Loads the dataset if it exists."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        
        # Log basic dataset info to wandb
        wandb.log({
            "dataset_loaded": True,
            "dataset_size": len(data["states"]),
            "dataset_path": DATASET_PATH
        })
        
        return data
    else:
        print(f"Dataset not found at {DATASET_PATH}")
        wandb.log({"dataset_found": False, "dataset_path": DATASET_PATH})
        return None

import matplotlib.pyplot as plt

def visualize_dataset_samples(data,traj_index=2900):
    """
    Visualizes a few samples from the dataset, including trajectories and RGB images.
    
    Args:
        data (dict): The dataset loaded from the .npz file.
        num_samples (int): Number of samples to visualize.
    """
    states = data["states"]
    states_rgb = data["states_rgb"]
    # states_rgb_next=data["states_rgb_next"]
    episode_starts = data["episode_starts"]
    episode_lengths = data["episode_lengths"]
    
    # Get the start and end indices of the selected trajectory
    start_idx = episode_starts[traj_index]
    end_idx = start_idx + episode_lengths[traj_index]

    # Plot the trajectory
    plt.figure(figsize=(12, 5))
    plt.subplot(3, 1, 1)  # First subplot for trajectory
    plt.plot(states[start_idx:end_idx, 0], states[start_idx:end_idx, 1], 'b-', label='Trajectory')
    plt.scatter(states[start_idx, 0], states[start_idx, 1], color='green', label='Start')
    plt.scatter(states[end_idx - 1, 0], states[end_idx - 1, 1], color='red', label='End')
    plt.title(f"Trajectory {traj_index + 1}")
    plt.legend()
    plt.grid(True)
    
    # Select 20 evenly spaced frames from the trajectory
    num_frames = 20
    frame_indices = np.linspace(start_idx, end_idx - 1, num_frames, dtype=int)
    print(states[frame_indices])
    # Create a figure for the 20 frames
    plt.figure(figsize=(15, 8))
    for i, frame_idx in enumerate(frame_indices):
        plt.subplot(4, 5, i + 1)  # Arrange in a 4x5 grid
        plt.imshow(states_rgb[frame_idx])
        plt.title(f"Frame {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


    
def main():
    # Initialize wandb
    wandb.init(
        project="ccbf-car-navigation_images",
        name=f"{random_value}",
        config={
            "environment": "DubinsCarEnv",
            "goal_position": goal_position.tolist(),
            "max_velocity": 3.0,
            "dt": 0.1,
            "safe_distance": 4.0,
            "seed": 42,
        }
    )
    
    seed_everything(42)
    # use_mps = torch.backends.mps.is_available()
    # device = "mps" if use_mps else "cpu"
    device="cpu"
    print(f"Using device: {device}")
    wandb.config.update({"device": device})

    # Load existing dataset or generate a new one
    data = load_dataset()
    if data is not None:
        print("Dataset loaded successfully")
        # Visualize dataset
        
        visualize_trajectories(data, num_trajectories=12)
        visualize_dataset_statistics(data)
    else:
        print("Generating new dataset...")
        generate_dataset_images(num_trajectories=3000)
        data = load_dataset()
        
        
    visualize_dataset_samples(data,traj_index=419)  

    # ########BELOW IS COMMENTED UNTIL I FIX VAE,DYNAMICS MODELS.
    # env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=goal_position, obstacle_radius=2)
    # dataset = Dataset(state_car_dim=2, state_obstacles_dim=2, control_dim=2, buffer_size=1000000, safe_distance=4.0)
    
    # # Populate dataset from loaded data
    # if data is not None:
    #     states = data["states"]
    #     actions = data["actions"]
    #     next_states = data["next_states"]
    #     states_rgb=data['states_rgb']
    #     states_rgb_next=data['states_rgb_next']
        
    #     print("Populating dataset object...")
    #     for s, a, s_next,img,img_next in zip(states, actions, next_states, states_rgb, states_rgb_next):
    #         dataset.add_data(s, a, s_next,img, img_next)
    #     print(f"Dataset populated with {len(states)} transitions")
    #     wandb.log({"dataset_populated": True, "dataset_size": len(states)})

    # # Initialize CBF
    # num_hidden_dim=3
    # dim_hidden=32
    # use_cql_actions=False
    # cql_actions_weight=10   ##changed this. run again for 0.001->10 but with detached
    # num_action_samples=10
    # temp=0.7
    # detach=False
    
    # cbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=num_hidden_dim, dim_hidden=dim_hidden)
    # cbf = cbf.to(device)
    
    
    
    # # Update wandb config with training parameters
    # wandb.config.update({
    #     "num_hidden_dim":num_hidden_dim,
    #     "dim_hidden":dim_hidden,
    #     "use_cql_actions": use_cql_actions,

    #     "safe_distance": 4.0,
    #     "eps_safe": 0.3,
    #     "eps_unsafe": 0.3,
    #     "safe_loss_weight": 1,
    #     "unsafe_loss_weight": 1.2,
    #     "action_loss_weight": 1,
    #     "batch_size": 128,
    #     "learning_rate": 1e-4,
    #     "cql_actions_weight":cql_actions_weight,
  
    #     "num_action_samples": 10,
    #     "num_state_samples": 8,
    #     "state_sample_std": 0.001,
    #     "step_count": 5000,
    #     "num_action_samples":10,
    #     "temp":temp,
    #     "detach":detach

        
    # })

    # # Setup trainer with new CQL parameters
    # trainer = Trainer(
    #     cbf, dataset, 
    #     safe_distance=4, 
    #     eps_safe=0.1, 
    #     eps_unsafe=0.3,
    #     safe_loss_weight=1, 
    #     unsafe_loss_weight=1.2, 
    #     action_loss_weight=1,
    #     dt=0.1, 
    #     batch_size=128, 
    #     opt_iter=1, 
    #     lr=1e-4, 
    #     device=device,
    #     # CQL parameters
    #     use_cql_actions=use_cql_actions, 
    #     cql_actions_weight=cql_actions_weight,  # Weight for L_CQL_actions loss
     
    #     num_action_samples=num_action_samples,   # Number of random actions to sample
    #     num_state_samples=8,     # Number of nearby states to sample
    #     state_sample_std=0.1,     # Standard deviation for state sampling
    #     detach=detach
    # )

    # # Create lists to track metrics (extended to track CQL losses)
    # losses = []
    # accuracies = []
    # safe_h_values = []
    # unsafe_h_values = []
    # cql_action_losses = []
    # cql_state_losses = []
        
    # # Train CBF with periodic logging and checkpoint saving
    # print("Training CCBF with Conservative CQL-inspired losses...")
    # step_count=5000
    
    # # Create tables for detailed training logs in wandb
    # wandb.define_metric("step")
    # wandb.define_metric("training/*", step_metric="step")
    
    # for i in range(step_count):
    #     # Log step to wandb
    #     wandb.log({"step": i})
        
    #     # Train one step
    #     acc_np, loss_np, avg_safe_h, avg_unsafe_h = trainer.train_cbf()
        
    #     # Unpack loss components for better logging
    #     safe_loss, unsafe_loss, deriv_loss, cql_actions_loss, cql_states_loss = loss_np
        
    #     # Store metrics for later analysis
    #     losses.append(loss_np)
    #     accuracies.append(acc_np)
    #     safe_h_values.append(avg_safe_h)
    #     unsafe_h_values.append(avg_unsafe_h)
    #     cql_action_losses.append(cql_actions_loss)
    #     cql_state_losses.append(cql_states_loss)
        
    #     # Log metrics to wandb
    #     wandb.log({
    #         "training/acc_safe":acc_np[0],
    #         "training/acc_unsafe":acc_np[1],
    #         "training/total_loss": np.sum(loss_np),
    #         "training/safe_loss": safe_loss,
    #         "training/unsafe_loss": unsafe_loss,
    #         "training/deriv_loss": deriv_loss,
    #         "training/cql_actions_loss": cql_actions_loss,
    #         "training/cql_states_loss": cql_states_loss,
    #         "training/accuracy": acc_np,
    #         "training/safe_h_value": avg_safe_h,
    #         "training/unsafe_h_value": avg_unsafe_h,
    #         "training/h_value_gap": avg_safe_h - avg_unsafe_h,
    #         "training/progress": i / step_count
    #     })
        
    #     # Save checkpoint every 1000 steps (or modify frequency as needed)
    #     if (i+1) % 5000 == 0:
    #         checkpoint = {
    #             'step': i,
    #             'model_state_dict': trainer.cbf.state_dict(),
    #             'loss': loss_np,
    #             'acc': acc_np,
    #             'cql_params': {
    #                 'cql_actions_weight': trainer.cql_actions_weight,
              
    #                 'num_action_samples': trainer.num_action_samples,
    #                 'num_state_samples': trainer.num_state_samples,
    #                 'state_sample_std': trainer.state_sample_std
    #             }
    #         }
    #         checkpoint_filename = f'ccbf_checkpoint_{i}_cql_actions{use_cql_actions}_cql_actions_weight{trainer.cql_actions_weight}_{random_value}.pt'
    #         torch.save(checkpoint, checkpoint_filename)
    #         print(f"Checkpoint saved to {checkpoint_filename}")

    # # Save final metrics for analysis
    # metrics = {
    #     'losses': np.array(losses),
    #     'accuracies': np.array(accuracies),
    #     'safe_h_values': np.array(safe_h_values),
    #     'unsafe_h_values': np.array(unsafe_h_values),
    #     'cql_action_losses': np.array(cql_action_losses),
    #     'cql_state_losses': np.array(cql_state_losses)
    # }
    # torch.save(metrics, 'ccbf_training_metrics.pt')
    
    # print("CCBF Training complete")

if __name__ == "__main__": 
    main()