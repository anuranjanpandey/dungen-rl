# Leetcode: https://leetcode.com/problems/dungeon-game/?envType=list&envId=50izszui
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np

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
        self.cur_pos = np.array([x, y])
        self.health += self.dungeon[x, y]
        terminated = np.all(self.cur_pos == self.target) or self.health <= 0
        truncated = False
        # reward = 1.0 if self.cur_pos == self.target else -0.1
        if self.health <= 0:
            reward = -1000.0  # Huge penalty for health less than zero
        elif np.all(self.cur_pos == self.target):
            reward = 1.0
        else:
            reward = -0.1
        self.action_taken.append(action)
        return self.cur_pos, reward, terminated, truncated, {}

config = (
    PPOConfig().environment(
        env=DungeonGame,
        env_config={"dungeon": [
            [-2,-3,3],
            [-5,-10,1],
            [10,30,-5]
            ], "initial_health": 7},
        # env_config={"dungeon": [[-69,40,-16,26,12,-83,-96,-38,-60,-8,14,34,-94,-44,-70,47,-2,-35,-63,-83,7,-60,37,50,32,22,-74,-86,-79,-32,-16,-69,37,27,22,-40,-3,39,26,-50,6,46,-65,-1,39,-19,-87,-88,35,47],[-8,-9,-38,-13,-63,49,-67,-64,-17,-9,-68,-1,-10,-49,8,-59,-34,-85,31,-53,-43,42,-8,48,-25,20,-9,-66,-54,-56,31,-92,-49,40,-91,-62,-14,-41,3,-58,-14,-67,4,38,-62,-41,-59,47,25,-10]], "initial_health": 10000}
    )
    .rollouts(num_rollout_workers=3)
)
algo = config.build()

for i in range(5):
    results = algo.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

env = DungeonGame({"dungeon": [
    [-2,-3,3],
    [-5,-10,1],
    [10,30,-5]
    ], "initial_health": 7})
    
# env = DungeonGame({"dungeon": [[-69,40,-16,26,12,-83,-96,-38,-60,-8,14,34,-94,-44,-70,47,-2,-35,-63,-83,7,-60,37,50,32,22,-74,-86,-79,-32,-16,-69,37,27,22,-40,-3,39,26,-50,6,46,-65,-1,39,-19,-87,-88,35,47],[-8,-9,-38,-13,-63,49,-67,-64,-17,-9,-68,-1,-10,-49,8,-59,-34,-85,31,-53,-43,42,-8,48,-25,20,-9,-66,-54,-56,31,-92,-49,40,-91,-62,-14,-41,3,-58,-14,-67,4,38,-62,-41,-59,47,25,-10]], "initial_health": 10000})

obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0
while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
print(f"Played 1 episode; total-reward={total_reward}")
print("Actions taken:", env.action_taken)
