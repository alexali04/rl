"""
It's always a good idea to:
    - test RL agent code on a simple workspace first (i.e. CartPole)



Experience Replay:

(state, action, new_state, reward, terminated) --> Python Deque
- as new experiences get added, we kick out old memories
"""



import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = "cude" if torch.cuda.is_available() else "cpu"



class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']


    def run(self, is_training=True, render=False):

        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []


        policy_dqn = DQN(env.observation_space.shape[0], num_actions).to(device


        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init





        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    # random action if < epsilon
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

            

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                # accumulate reward
                episode_reward += reward
                
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                
                # Move to new state
                state = new_state

            rewards_per_episode.append(episode_reward)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)
