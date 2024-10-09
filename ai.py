import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pong_game import PongEnv  # Import the PongEnv you built
from inspect_model import save_model_image  # Import the function to save the model image

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # Input size is 6 (paddle and ball positions and velocities)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=-1)
        return x


class REINFORCE:
    def __init__(self, policy_network, lr=0.0001, gamma=0.95, entropy_coeff=0.01):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        self.entropy_coeff = entropy_coeff

    def select_action(self, state, env):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        # Calculate the discounted rewards
        R = 0
        rewards = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

        # Normalize the rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Calculate the policy gradient loss
        loss = []
        entropy = []
        for log_prob, reward in zip(self.log_probs, rewards):
            loss.append(-log_prob * reward)
            entropy.append(-log_prob * log_prob.exp())

        policy_loss = torch.cat(loss).sum()
        entropy_loss = torch.cat(entropy).sum()
        total_loss = policy_loss - self.entropy_coeff * entropy_loss

        # Backpropagate and update the network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Reset the rewards and log_probs for the next episode
        self.log_probs = []
        self.rewards = []

    
# Save the model
def save_model(model, filename="policy_network.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


# Load the model
def load_model(model, filename="policy_network.pth"):
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from {filename}")


def main():
    env = PongEnv()
    policy_network = PolicyNetwork()

    try:
        load_model(policy_network)
    except FileNotFoundError:
        print("Model not found. Training a new model.")

    agent = REINFORCE(policy_network)

    for episode in range(1000):  # Play 1000 episodes
        state = env.reset()
        episode_reward = 0

        for t in range(10000):  # Limit the number of steps per episode
            action = agent.select_action(state, env)
            state, reward, done = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            env.render()

            if done:
                break

        agent.update_policy()

        if episode % 100 == 0 and episode > 0:
            save_model(policy_network)
            save_model_image(policy_network, episode, state)

        print(f"Episode {episode}, Total Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    main()
