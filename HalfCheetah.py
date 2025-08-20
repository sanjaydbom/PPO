import gymnasium as gym
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(17,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12)
        )
        self.critic = nn.Sequential(
            nn.Linear(17,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)
    
agent = Agent()

optimizer = torch.optim.Adam(agent.parameters(), 1e-4) 
loss_fn = torch.nn.MSELoss()

GAMMA  = 0.99
LAMBDA = 0.95

EPSILON = 0.2 # decrease
C1      = 0.5 # Increase
C2      = 0.01 # Increase and anneal or decrease

NUM_EPOCHS      = 10000
NUM_BATCHES     = 35 
MINI_BATCH_SIZE = 250 #Increase

#Episode never terminates but is truncated at 1000 steps so we can optimize with set array size
MAX_STEPS = 1000

env = gym.make("HalfCheetah-v5")
rewards_over_time = [0]*NUM_EPOCHS

for i in range(NUM_EPOCHS):
    state, _ = env.reset()

    state_array       = torch.zeros((MAX_STEPS,17))
    action_array      = torch.zeros((MAX_STEPS,6))
    reward_array      = torch.zeros(MAX_STEPS)
    state_value_array = torch.zeros(MAX_STEPS)
    log_prob_array   = torch.zeros_like(action_array)

    pos = 0
    while True:
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float32)
            logit, state_value = agent(state)
            mean = logit[:6]
            std = logit[6:]
            distribution = torch.distributions.normal.Normal(mean, torch.exp(std))
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            state_array[pos]       = state
            action_array[pos]      = action
            state_value_array[pos] = state_value
            log_prob_array[pos]   = log_prob
            state, reward, terminated, truncated, _ = env.step(np.array(torch.tanh(action)))
            reward_array[pos]      = reward
            
            pos += 1

            if truncated:
                break

    rewards_over_time[i] = torch.sum(reward_array)
    if i % 10 == 0 and i != 0:
        print(f"{i / NUM_EPOCHS * 100:.2f}% complete: Avg Score (prev 100 runs) = {np.mean(rewards_over_time[max(i-100,0):i]):.2f}, Last Reward = {rewards_over_time[i-1]:.2f}")

    advantage_array = torch.zeros_like(reward_array)
    with torch.no_grad():
        _, next_state_value = agent(torch.tensor(state, dtype = torch.float32))
    advantage_array[-1] = reward - state_value + GAMMA * next_state_value.item()
    for j in reversed(range(MAX_STEPS-1)):
        TD_error = reward_array[j] + GAMMA * state_value_array[j+1] - state_value_array[j]
        advantage_array[j] = TD_error + GAMMA * LAMBDA * advantage_array[j+1] 

    target_array = advantage_array + state_value_array
    advantage_array = (advantage_array - advantage_array.mean()) / (advantage_array.std() + 1e-8)

    indicies = np.arange(len(advantage_array))
    for _ in range(NUM_BATCHES):
        random.shuffle(indicies)
        for j in range(0,len(indicies), MINI_BATCH_SIZE):
            index = indicies[j:j+MINI_BATCH_SIZE]

            states        = state_array[index]
            old_log_probs = log_prob_array[index]
            actions       = action_array[index]
            advantages    = advantage_array[index]

            targets = target_array[index]

            logits, values = agent(states)
            means     = logits[:,:6]
            stds      = logits[:,6:]

            distribution  = torch.distributions.normal.Normal(means, torch.exp(stds))
            new_log_probs = distribution.log_prob(actions)

            ratio = torch.exp(torch.sum(new_log_probs, dim = -1) - torch.sum(old_log_probs, dim = -1))
            
            l_clip = torch.min(ratio * advantages.detach(), torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages.detach()).mean()
            l_vf   = loss_fn(values.squeeze(), targets.detach())
            l_s    = distribution.entropy().mean()

            optimizer.zero_grad()
            loss = -1 * l_clip + C1 * l_vf - C2 * l_s
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

plt.plot(rewards_over_time)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()