import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN

"""
It's always a good idea to:
    - test RL agent code on a simple workspace first (i.e. CartPole)

"""

class Agent:
    def run(self, is_training=True, render=False):

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(env.observation_space.shape[0], num_actions)

        obs, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            
            # Checking if the player is still alive
            if terminated:
                break

        env.close()


