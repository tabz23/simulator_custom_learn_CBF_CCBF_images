import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from numpy.linalg import norm

class DubinsCarEnv:
    def __init__(self, max_velocity, dt, obstacle_radius=None, goal_position=None,reached_epsilon=0.5):
        self.max_velocity = max_velocity
        self.dt = dt
        self.state = None
        self.history = []
        self.fig, self.ax = None, None
        self.goal_position = goal_position 
        self.reached_epsilon=reached_epsilon

        self.obstacle_position = np.array([10, 11]) 
        self.obstacle_radius = obstacle_radius
        
        # Track collisions and steps
        self.collision = False
        self.step_count = 0
        self.max_steps = 200  # Adding a maximum step limit
        
        self.beta = 0.9  # Discount factor for distance reward
        self.R_goal = 50  # Reward for reaching the goal
        self.previous_distance_to_goal = None

    def reset(self):
        self.state = np.random.uniform(-18, 5, size=2)   # Initial position of the car
        # print(self.state)
        self.history = [self.state]
        self.collision = False
        self.step_count = 0
        
        # Initialize previous distance to goal
        self.previous_distance_to_goal = np.linalg.norm(self.goal_position - self.state)
        
        return self.state

    def goal_reaching_controller(self):
        """
        Computes a refined velocity command toward the goal using a PD controller.
        """
        kp = 1  # Proportional gain
        kd = 0.1  # Derivative gain (tuneable)
        
        error = self.goal_position - self.state
        velocity_command = kp * error
        
        # Compute the velocity error (rate of change)
        velocity_error = velocity_command / self.dt
        refined_velocity_command = velocity_command + kd * velocity_error

        # Limit velocity magnitude
        if np.linalg.norm(refined_velocity_command) > self.max_velocity:
            refined_velocity_command = (refined_velocity_command / np.linalg.norm(refined_velocity_command)) * self.max_velocity
        
        return refined_velocity_command
    
    # def goal_reaching__safe_controller(self, naive_velocity_command):
    #     """
    #     Refines the naive velocity command to avoid obstacles using optimization.
    #     """
    #     closest_obstacle_position = self.obstacle_position
    #     obstacle_radius = self.obstacle_radius

    #     diff = self.state - closest_obstacle_position
    #     B = norm(diff) - obstacle_radius
    #     grad_B = 2 * diff

    #     u = cp.Variable(2)
    #     P = np.eye(2)
    #     q = -2 * naive_velocity_command
    #     B_dot = grad_B @ u
    #     alpha = 5

    #     prob = cp.Problem(cp.Minimize((1/2) * cp.quad_form(u, P) + q.T @ u),
    #                       [B_dot + alpha * B >= 0, cp.norm(u) <= self.max_velocity])
    #     prob.solve()

    #     if prob.status == cp.OPTIMAL:
    #         safe_velocity_command = u.value
    #     else:
    #         safe_velocity_command = naive_velocity_command
    #     return safe_velocity_command
    def goal_reaching__safe_controller(self, naive_velocity_command):
        ##IF OBSTACLE RADIUS 0 RETURNS NAIVE VELOCITY COMMAND AND NO CBF CALCULATED. fixed.
        """
        Refines the naive velocity command to avoid obstacles using optimization.
        """
        closest_obstacle_position = self.obstacle_position
        obstacle_radius = self.obstacle_radius
        # if (obstacle_radius==0): ##commented this on april 26
        #     return naive_velocity_command ##commented this on april 26

        diff = self.state - closest_obstacle_position
        B = norm(diff) - obstacle_radius
        grad_B = 2 * diff

        u = cp.Variable(2)
        P = np.eye(2)
        q = -2 * naive_velocity_command
        
        # Rewrite the constraint in standard QP form
        # B_dot + alpha * B >= 0 becomes grad_B @ u + alpha * B >= 0
        # Rearranged to: grad_B @ u >= -alpha * B
        
        alpha = 5##i used 5 to generate old dataset
        
        rhs = -alpha * B  # This is now a constant
        
        # Create a QP-compatible constraint
        prob = cp.Problem(cp.Minimize((1/2) * cp.quad_form(u, P) + q.T @ u),
                        [grad_B @ u >= rhs,
                            u <= 2.5,               # Upper control limits. fixed 
                            u >= -2.5               # Lower control limits. fixed
                         
                         ]
                        
                        
                        )
        
        prob.solve(solver=cp.OSQP)##SOLVING AS A QP

        if prob.status == cp.OPTIMAL:
            safe_velocity_command = u.value
        else:
            safe_velocity_command = naive_velocity_command
        return safe_velocity_command

    def calculate_reward(self):
        """
        Calculate reward based on distance to goal, as described in the image.
        """
        distance_to_goal = np.linalg.norm(self.goal_position - self.state)
        
        # Distance-based reward
        reward = (self.previous_distance_to_goal - distance_to_goal) * self.beta
        
        # Goal reward
        if distance_to_goal < 0.5:
            reward += self.R_goal
        
        # Update previous distance
        self.previous_distance_to_goal = distance_to_goal
        
        return reward

    def calculate_cost(self):
        """
        Calculate cost: 1 when hitting an obstacle, 0 otherwise.
        """
        return 1.0 if self.check_collision() else 0.0

    def step(self, action):
        """
        Execute one step in the environment.
        Returns state, reward, cost, done, info.
        """
        self.step_count += 1
        
        #ADD SOME NOISE
        # if np.random.rand() < 0.1:##uncomment when generating a dataset. note that some results previously were generated using this not commented
        #this includesseed 3 with constraint True CBF_CHECKPOINT alpha_1: /Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_final306.pth till before it everything

        #     noise = np.random.normal(0, 0.5, size=action.shape)  # Gaussian noise with mean 0 and stddev 0.1
        #     action += noise
        ##check if les than max velocity
        if np.linalg.norm(action) > self.max_velocity:
            action = (action / np.linalg.norm(action)) * self.max_velocity
            
        # Update state   
        self.state = self.state + action * self.dt
        self.history.append(self.state)
        
        # Check if we hit an obstacle
        self.collision = self.check_collision()
        
        # Calculate reward and cost
        reward = self.calculate_reward()
        cost = self.calculate_cost()
        
        # Check termination conditions
        goal_reached = np.linalg.norm(self.goal_position - self.state) < self.reached_epsilon
        timeout = self.step_count >= self.max_steps
        
        done = goal_reached or timeout # or self.collision    ##if collision continue
        
        info = {
            'goal_reached': goal_reached,
            'collision': self.collision,
            'timeout': timeout,
            'distance_to_goal': np.linalg.norm(self.goal_position - self.state)
        }
        # print(self.collision)

        return self.state, reward, cost, done, info
    
    # def step_in_latent_space(self, action):##added this now
    #     """
    #     Execute one step in the environment.
    #     Returns state, reward, cost, done, info.
    #     """
    #     self.step_count += 1
        
    #     #ADD SOME NOISE
    #     # if np.random.rand() < 0.1:##uncomment when generating a dataset. note that some results previously were generated using this not commented
    #     #this includesseed 3 with constraint True CBF_CHECKPOINT alpha_1: /Users/i.k.tabbara/Documents/python directory/best_cbf_VAE_Deterministic_4_cql_final306.pth till before it everything

    #     #     noise = np.random.normal(0, 0.5, size=action.shape)  # Gaussian noise with mean 0 and stddev 0.1
    #     #     action += noise
    #     ##check if les than max velocity
    #     if np.linalg.norm(action) > self.max_velocity:
    #         action = (action / np.linalg.norm(action)) * self.max_velocity
            
    #     # Update state   
    #     self.state = self.state + action * self.dt
    #     self.history.append(self.state)
        
    #     # Check if we hit an obstacle
    #     self.collision = self.check_collision()
        
    #     # Calculate reward and cost
    #     reward = self.calculate_reward()
    #     cost = self.calculate_cost()
        
    #     # Check termination conditions
    #     goal_reached = np.linalg.norm(self.goal_position - self.state) < self.reached_epsilon
    #     timeout = self.step_count >= self.max_steps
        
    #     done = goal_reached or timeout # or self.collision    ##if collision continue
        
    #     info = {
    #         'goal_reached': goal_reached,
    #         'collision': self.collision,
    #         'timeout': timeout,
    #         'distance_to_goal': np.linalg.norm(self.goal_position - self.state)
    #     }
    #     # print(self.collision)

    #     return self.state, reward, cost, done, info

    def check_collision(self):
        # Check collision with the single obstacle
        diff = self.state - self.obstacle_position
        distance_to_obstacle = norm(diff)
        return distance_to_obstacle - 4 <= 0 #hardcoded as 4 as that is what we will use when training controller and labeling unsafe states

    # def render(self):
    #     if self.fig is None or self.ax is None:
    #         self.fig, self.ax = plt.subplots()

    #     self.ax.clear()

    #     trajectory = np.array(self.history)
    #     all_x_traj = trajectory[:, 0]
    #     all_y_traj = trajectory[:, 1]
    #     self.ax.plot(all_x_traj, all_y_traj, 'b-', label="Trajectory")
        
    #     x, y = self.state
    #     self.ax.plot(x, y, 'bo', label="Car Position")

    #     # Render the single obstacle
    #     self.ax.add_patch(plt.Circle(self.obstacle_position, 4, color='g', alpha=0.5, label="Obstacle"))
    #     self.ax.add_patch(plt.Rectangle((self.goal_position[0] - 1.5, self.goal_position[1] - 1.5), width=3, height=3, color='b', alpha=0.5, label="Goal"))

    #     self.ax.set_xlim(-30, 30)
    #     self.ax.set_ylim(-30, 30)
    #     self.ax.set_aspect('equal', adjustable='box')

    #     # Change color when collision occurs
    #     if self.collision:
    #         self.ax.set_title("CBF-based Car Simulation - COLLISION!")
    #         self.ax.set_facecolor((1.0, 0.9, 0.9))  # Light red background
    #     else:
    #         self.ax.set_title("CBF-based Car Simulation")
    #         self.ax.set_facecolor((1.0, 1.0, 1.0))  # White background

    #     self.ax.set_xlabel("X Position")
    #     self.ax.set_ylabel("Y Position")
    #     self.ax.legend()

    #     # plt.pause(0.00000000001)
    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))  # create figure
        self.ax.clear()

        trajectory = np.array(self.history)
        all_x_traj = trajectory[:, 0]
        all_y_traj = trajectory[:, 1]
        self.ax.plot(all_x_traj, all_y_traj, 'b-', label="Trajectory")

        x, y = self.state
        self.ax.plot(x, y, 'bo', label="Car Position")

        # Render the single obstacle
        self.ax.add_patch(plt.Circle(self.obstacle_position, 4, color='g', alpha=0.5, label="Obstacle"))
        self.ax.add_patch(plt.Rectangle((self.goal_position[0] - 1.5, self.goal_position[1] - 1.5), width=3, height=3, color='b', alpha=0.5, label="Goal"))

        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-30, 30)
        self.ax.set_aspect('equal', adjustable='box')

        # Change color when collision occurs
        if self.collision:
            self.ax.set_title("CBF-based Car Simulation - COLLISION!")
            self.ax.set_facecolor((1.0, 0.9, 0.9))  # Light red background
        else:
            self.ax.set_title("CBF-based Car Simulation")
            self.ax.set_facecolor((1.0, 1.0, 1.0))  # White background

        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.legend()

        plt.pause(0.000000000001) # Adjust pause duration based on real-time factor and the simulation speed
        plt.draw()