import torch
from torch import nn, optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random


env = gym.make("Ant-v5")

ACTION_SPACE = env.action_space.shape[0]
OBSERVATION_SPACE = env.observation_space.shape[0]
print(OBSERVATION_SPACE)

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(OBSERVATION_SPACE, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64,ACTION_SPACE * 2)
        self.critic = nn.Linear(64,1)

    def forward(self,x):
        common = self.common(x)
        return self.actor(common), self.critic(common)

device = "mps" if torch.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else device

agent = Agent().to(device)
LR = 5e-5
optimizer = optim.Adam(agent.parameters(), LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.75)
loss_fn = nn.MSELoss()

NUM_EPOCHS = 3000
NUM_BATCHES = 15
BATCH_SIZE = 200

EPSILON = 0.2
C1 = 0.5
C2 = 0.01

GAMMA = 0.99
LAMBDA = 0.95


rewards_over_time = []


state_array = []
action_array = []
log_prob_array = []
reward_array = []
state_value_array = []
done_array = []

for epoch in range(NUM_EPOCHS):
    current_state_during_training, _ = env.reset()
    cur_reward = 0

    while True:
        current_state_during_training = torch.tensor(current_state_during_training, dtype = torch.float32, device = device)
        #print(current_state_during_training.shape)
        with torch.no_grad():
            actor_values, critic_values = agent(current_state_during_training)
            #print(actor_values.shape)
            #print(critic_values.shape)
            means_during_training = actor_values[:ACTION_SPACE]
            stds_during_training = actor_values[ACTION_SPACE:]

            distribution_during_training = torch.distributions.normal.Normal(means_during_training, torch.exp(torch.clamp(stds_during_training, -5,2)))

            actions_during_training = distribution_during_training.sample()
            log_probs_during_training = distribution_during_training.log_prob(actions_during_training)

            state_array.append(current_state_during_training)
            action_array.append(actions_during_training)
            log_prob_array.append(log_probs_during_training)
            state_value_array.append(critic_values.detach())

            current_state_during_training, reward_during_training, terminated, truncated, _ = env.step(np.array(torch.tanh(actions_during_training).cpu()))

            reward_array.append(reward_during_training)
            done_array.append(1 if terminated else 0)

            cur_reward += reward_during_training

            if terminated or truncated:
                break

    rewards_over_time.append(cur_reward)
    if epoch % 10 == 0 and epoch != 0:
        print(f"Epoch #{epoch}: Current Reward {rewards_over_time[-1]:.2f}, Average Reward (Last 100 epochs) {np.mean(rewards_over_time[max(0,epoch-100):epoch+1]):.2f}")
    
    
    if len(reward_array) > 2000:
        state_array = torch.stack(state_array).to(device)
        action_array = torch.stack(action_array).to(device)
        reward_array = torch.tensor(reward_array, device = device, dtype = torch.float32)
        log_prob_array = torch.stack(log_prob_array).to(device)
        state_value_array = torch.tensor(state_value_array, device = device, dtype = torch.float32)
        done_array = torch.tensor(done_array, device = device, dtype = torch.float32)

        advantage_array = torch.zeros_like(reward_array, device = device, dtype = torch.float32)

        if done_array[-1] == 0:
            with torch.no_grad():
                _, next_state_value_for_advantage_calculation = agent(torch.tensor(current_state_during_training, dtype = torch.float32, device = device))
        else:
            next_state_value_for_advantage_calculation = torch.tensor(0,device = device)

        advantage_array[-1] = reward_array[-1] - state_value_array[-1] + GAMMA * next_state_value_for_advantage_calculation
        for j in reversed(range(len(advantage_array)-1)):
            TD_Error = reward_array[j] - state_value_array[j] + GAMMA * state_value_array[j+1] * (1 - done_array[j])
            advantage_array[j] = TD_Error + GAMMA * LAMBDA * advantage_array[j+1] * (1 - done_array[j])

        target_array = advantage_array + state_value_array

        advantage_array = (advantage_array - advantage_array.mean()) / (advantage_array.std() + 1e-8)

        indicies = np.arange(len(advantage_array))
        for batch in range(NUM_BATCHES):
            random.shuffle(indicies)
            for MINI_BATCH in range(0, len(indicies), BATCH_SIZE):
                current_indicies = indicies[MINI_BATCH:MINI_BATCH + BATCH_SIZE]

                states_from_training = state_array[current_indicies]
                action_from_training = action_array[current_indicies]
                old_log_probs = log_prob_array[current_indicies]

                advantages = advantage_array[current_indicies]
                targets = target_array[current_indicies]

                #Calculate l_clip
                actor_outputs, values = agent(states_from_training)
                if actor_outputs.isnan().any():
                    for param in agent.common.parameters():
                        print(param)
                    print(torch.isnan(agent.actor).any())
                    print(values)
                distribution = torch.distributions.normal.Normal(actor_outputs[:,:ACTION_SPACE], torch.exp(torch.clamp(actor_outputs[:,ACTION_SPACE:],-5,2)))
                new_log_probs = distribution.log_prob(action_from_training)
                ratio = torch.exp(torch.sum(new_log_probs, dim = -1) - torch.sum(old_log_probs, dim = -1).detach())
                #print(ratio.shape)
                #print(advantages.shape)
                l_clip = torch.min(ratio * advantages.detach(), torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages.detach()).mean()
                #print(values.squeeze(1).shape)
                #print(targets.shape)
                l_vf = loss_fn(values.squeeze(1), targets.detach())

                l_s = distribution.entropy().mean()

                optimizer.zero_grad()
                loss = -1 * l_clip + C1 * l_vf - C2 * l_s
                if torch.isinf(l_s).any():
                    print("1")
                    print(distribution.entropy().mean())
                    print(torch.exp(actor_outputs[:,ACTION_SPACE:]))
                if torch.isinf(l_clip).any():
                    print("IDK")
                if torch.isinf(l_vf).any():
                    print("2")
                if torch.isnan(l_clip).any():
                    print("3")
                    print(ratio)
                    print(advantages)
                if torch.isnan(l_vf).any():
                    print("4")
                if torch.isnan(l_s).any():
                    print("5")
                #print(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                scheduler.step()

        state_array = []
        action_array = []
        log_prob_array = []
        reward_array = []
        state_value_array = []
        done_array = []

plt.plot(rewards_over_time)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()