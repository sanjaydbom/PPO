import gymnasium as gym # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from torch import nn # type: ignore
import matplotlib.pyplot as plt # type: ignore
import random

class Agent(nn.Module):
    def __init__(self):
        super(Agent,self).__init__()
        self.input = nn.Linear(4,32)
        self.probabilities = nn.Linear(32,2)
        self.value = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.input(x)
        x = self.relu(x)
        probs = self.probabilities(x)
        vals = self.value(x)
        return probs, vals
    
agent = Agent()
previous_policy = Agent()
previous_policy.load_state_dict(agent.state_dict())
softmax = nn.Softmax(dim = -1)

optimizer = torch.optim.Adam(agent.parameters(), lr = 3.5e-5)
MSELoss = nn.MSELoss(reduction='none')

TOTAL_RUNS = 15000
GAMMA = 0.99
EPSILON = 0.2
C1 = 0.5
C2 = 0.01
NUM_BATCHES = 25
MINI_BATCH_SIZE = 32

rewards_over_time = [0] * TOTAL_RUNS
env = gym.make("CartPole-v1")

def get_action(probs):
    probs = torch.distributions.Categorical(logits=probs)
    action = probs.sample()
    log_prob = probs.log_prob(action)
    return action.item(), log_prob


for i in range(TOTAL_RUNS):
    #C2 = 0.995 * C2
    if i % 100 == 0 and i != 0:
        print(f"{i / TOTAL_RUNS * 100:.2f}% done: Score {rewards_over_time[i-1]}")

    state, _ = env.reset()
    state_actions = []
    total_reward = 0
    while True:
        probs, _ = previous_policy(state)
        action, log_probs = get_action(probs)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state_actions.append((state, action, reward, next_state, terminated or truncated, log_probs))
        if terminated or truncated:
            break
        state = next_state

    for j in range(NUM_BATCHES):
        array = random.sample(state_actions, k = min(MINI_BATCH_SIZE, len(state_actions)))

        state, action, reward, next_state, done, log_probs = zip(*array)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)
        log_probs = torch.tensor(log_probs)

        probs, values = agent(np.array(state))
        _, next_values = agent(np.array(next_state))
        next_values = next_values.masked_fill(done.unsqueeze(1), 0)

        distribution = torch.distributions.Categorical(logits=probs)

        target = reward.unsqueeze(1) + GAMMA * next_values
        advantage = target.detach() - values
        
        ratio = torch.exp(distribution.log_prob(action) - log_probs.detach())
        ratio = ratio.unsqueeze(1)

        l_clip = torch.min(ratio * advantage.detach(), torch.clamp(ratio, min = 1 - EPSILON, max = 1 + EPSILON) * advantage.detach())
        l_vf = MSELoss(values, target)
        l_s = distribution.entropy()
        optimizer.zero_grad()
        loss = -1 * l_clip + C1 * l_vf - C2 * l_s.unsqueeze(1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()


    #print(total_reward)
    rewards_over_time[i] = total_reward
    previous_policy.load_state_dict(agent.state_dict())

plt.plot(rewards_over_time)
plt.grid(True)
plt.title("Rewards for each Training Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

env = gym.make("CartPole-v1", render_mode = "human")

state, _ = env.reset()
total_reward = 0
with torch.no_grad():
    while True:
        probs, _ = agent(state)
        action, _ = get_action(probs)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
        state = next_state

print(total_reward)
env.close()