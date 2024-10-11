import torch
import torch.nn as nn
import torch.optim as optim
from pong_game import PongEnv  # Import the PongEnv you built

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # Input size is 6 (paddle and ball positions and velocities)
        # Excercise: Create the missing layer to connect the input to the output
        self.fc3 = nn.Linear(128, 3)  # Output size is 3 (move up, stay, move down)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # ReLU: Rectified Linear Unit
        x = torch.relu(self.<TODO>) # ReLU: Rectified Linear Unit
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


class REINFORCE:
    # Excercise: Play with the learning rate and gamma values
    # Learning rate value is usually between 0.001 and 0.01, but can be smaller or larger
    # Gamma (discount factor) is usually between 0.9 and 0.99, this value determines how much importance we give to future rewards
    def __init__(self, policy_network, lr=<TODO>, gamma=<TODO>):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

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
            # Calculate the discounted reward, gamma is the discount factor
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

        # Normalize the rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Calculate the policy gradient loss
        loss = []
        for log_prob, reward in zip(self.log_probs, rewards):
            loss.append(-log_prob * reward)

        policy_loss = torch.cat(loss).sum()
        total_loss = policy_loss

        # Backpropagate and update the network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Reset the rewards and log_probs for the next episode
        self.log_probs = []
        self.rewards = []

def main():
    env = PongEnv()
    policy_network = PolicyNetwork()

    # Provide the model to the REINFORCE agent
    agent = REINFORCE(policy_network)

    for episode in range(1000):  # Play 1000 episodes
        state = env.reset()
        episode_reward = 0

        for _ in range(10000):  # Limit the number of steps per episode
            action = agent.select_action(state, env)
            state, reward, done = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            env.render()

            if done:
                break

        agent.update_policy()

        print(f"Episode {episode}, Total Reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    main()
