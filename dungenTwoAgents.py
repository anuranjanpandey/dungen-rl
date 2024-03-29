import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
import ray

class DungeonGame(gym.Env):
    def __init__(self, config):
        self.dungeon = np.array(config["dungeon"])
        self.grid_size = self.dungeon.shape
        self.target = (self.grid_size[0]-1, self.grid_size[1]-1)
        self.cur_pos = np.array([0, 0])
        self.health = config["initial_health"]
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Box(low=0, high=max(self.grid_size)-1, shape=(2,))
        self.action_taken = []

    def reset(self, *, seed=None, options=None):
        self.cur_pos = np.array([0, 0])
        self.health = config["env_config"]["initial_health"]
        self.action_taken = []
        return self.cur_pos, {}

    def step(self, action):
        x, y = self.cur_pos
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size[1] - 1, y + 1)
        
        # Check if the new position is outside the grid
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            # Penalize the agent for going out of the grid
            self.health -= 1000.0
        
        self.cur_pos = np.array([x, y])
        self.health += self.dungeon[x, y]
        terminated = np.all(self.cur_pos == self.target) or self.health <= 0
        truncated = False
        if self.health <= 0:
            reward = -1000.0  
        elif np.all(self.cur_pos == self.target):
            reward = 1.0
        else:
            reward = -0.1
        self.action_taken.append(action)
        return self.cur_pos, reward, terminated, truncated, {}

# Define the training configuration
config = (
    PPOConfig().environment(
        env=DungeonGame,
        env_config={"dungeon": [
            [-2,-3,3],
            [-5,-10,1],
            [10,30,-5]
            ], "initial_health": 50},
        # env_config={"dungeon": [[-69,40,-16,26,12,-83,-96,-38,-60,-8,14,34,-94,-44,-70,47,-2,-35,-63,-83,7,-60,37,50,32,22,-74,-86,-79,-32,-16,-69,37,27,22,-40,-3,39,26,-50,6,46,-65,-1,39,-19,-87,-88,35,47],[-8,-9,-38,-13,-63,49,-67,-64,-17,-9,-68,-1,-10,-49,8,-59,-34,-85,31,-53,-43,42,-8,48,-25,20,-9,-66,-54,-56,31,-92,-49,40,-91,-62,-14,-41,3,-58,-14,-67,4,38,-62,-41,-59,47,25,-10]], "initial_health": 10000}
    )
    .rollouts(num_rollout_workers=3)
)

# Initialize Ray Tune
ray.init()

# Train agent 1
agent1 = config.build()
for i in range(20):
    results = agent1.train()
    print(f"Agent 1 - Iter: {i}; avg. reward={results['episode_reward_mean']}")

# Train agent 2
agent2 = config.build()
for i in range(20):
    results = agent2.train()
    print(f"Agent 2 - Iter: {i}; avg. reward={results['episode_reward_mean']}")

# Output the trained agents' actions
env = DungeonGame({"dungeon": [
    [-2,-3,3],
    [-5,-10,1],
    [10,30,-5]
], "initial_health": 50})

obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0

# Agent 1 actions
print("Agent 1 Actions:")
while not terminated and not truncated:
    action = agent1.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
print(f"Played 1 episode with Agent 1; total-reward={total_reward}")
print("Actions taken:", env.action_taken)

# Reset environment for Agent 2
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0

# Agent 2 actions
print("\nAgent 2 Actions:")
while not terminated and not truncated:
    action = agent2.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
print(f"Played 1 episode with Agent 2; total-reward={total_reward}")
print("Actions taken:", env.action_taken)

# Close Ray Tune
ray.shutdown()
