import flappy_bird_gymnasium
import gymnasium
import torch
import torch.nn as nn

# DQN
# DQN - MLP: state --> action - policy network dicates action to be taken

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x):
        



class Agent:
    def run(self, is_training=True, render=False):



        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
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
