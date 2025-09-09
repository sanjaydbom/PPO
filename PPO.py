from torch import optim
import gymnasium as gym
import matplotlib.pyplot as plt
import yaml
import numpy as np
import random
import sys

from Architecture import *

hyperparams_file_name = sys.argv[1]
with open(hyperparams_file_name, 'r') as f:
    hyperparams = yaml.safe_load(f)

env = gym.make(hyperparams['env_name'])
obs_dim = env.observation_space.shape[0]

discrete = isinstance(env.action_space, gym.spaces.Discrete)
if discrete:
    action_dim = env.action_space.n
    action_range = action_dim
else:
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high
actor, critic = get_networks(obs_dim, action_dim, hyperparams['actor_hidden'], hyperparams['critic_hidden'], hyperparams['std_clamp_min'], hyperparams['std_clamp_max'], discrete, action_range)
actor_optim = optim.Adam(actor.parameters(), float(hyperparams['actor_lr']))
critic_optim = optim.Adam(critic.parameters(), float(hyperparams['critic_lr']))

GAMMA = hyperparams['GAMMA']
LAMBDA = hyperparams['LAMBDA']
EPSILON = hyperparams['EPSILON']
C1 = hyperparams['C1']
C2 = hyperparams['C2']
NUM_EPOCHS = hyperparams['NUM_EPOCHS']
BATCH_SIZE = hyperparams['BATCH_SIZE']
NUM_BATCHES = hyperparams['NUM_BATCHES']
TRAINING_MIN_SIZE = hyperparams['TRAINING_MIN_SIZE']

LOGGING_FREQ = hyperparams['LOGGING_FREQ']
SLIDING_WINDOW_SIZE = hyperparams['SLIDING_WINDOW_SIZE']

MODEL_FILE_NAME = hyperparams['MODEL_FILE_NAME']
if MODEL_FILE_NAME is None:
    MODEL_FILE_NAME = hyperparams['env_name'].split('-')[0]

loss_func = torch.nn.MSELoss()
best_avg_reward = -100000

rewards_over_time = []
loss_over_time = []
entropy_loss = []
clip_loss = []
value_fn_loss = []

state_array = []
action_array = []
reward_array = []
next_state_array = []
not_terminated_array = []
log_probs_array = []
state_value_array = []

for epoch in range(NUM_EPOCHS):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)


    current_reward = 0
    with torch.no_grad():
        while True:
            action, log_prob = actor(state)
            state_value = critic(state)
            next_state, reward, terminated, truncated, _ = env.step(np.asarray(action[0]))
            next_state = torch.tensor(next_state, dtype = torch.float32)

            state_array.append(state[0])
            action_array.append(action)
            reward_array.append(reward)
            next_state_array.append(next_state)
            not_terminated_array.append(0 if terminated else 1)
            log_probs_array.append(log_prob)
            state_value_array.append(state_value)

            current_reward += reward
            state = next_state.unsqueeze(0)

            if terminated or truncated:
                break
        
    rewards_over_time.append(current_reward)
    if len(reward_array) > TRAINING_MIN_SIZE:
        state_array = torch.stack(state_array)
        action_array = torch.stack(action_array)
        reward_array = torch.tensor(reward_array, dtype = torch.float32).unsqueeze(1)
        next_state_array = torch.stack(next_state_array)
        not_terminated_array = torch.tensor(not_terminated_array).unsqueeze(1)
        log_probs_array = torch.tensor(log_probs_array, dtype = torch.float32).unsqueeze(1)
        state_value_array = torch.tensor(state_value_array, dtype = torch.float32).unsqueeze(1)

        '''print(state_array.shape)
        print(action_array.shape)
        print(reward_array.shape)
        print(next_state_array.shape)
        print(not_terminated_array.shape)
        print(log_probs_array.shape)
        print(state_value_array.shape)'''

        with torch.no_grad():
            if not_terminated_array[-1]:
                next_state_value = critic(next_state_array[-1])
            else:
                next_state_value = torch.tensor(0)

            advantage_array = torch.zeros_like(reward_array)
            advantage_array[-1] = reward_array[-1] + GAMMA * next_state_value - state_value_array[-1]

            for i in reversed(range(len(reward_array)-1)):
                TD_error = reward_array[i] - state_value_array[i] + GAMMA * state_value_array[i+1] * not_terminated_array[i]
                advantage_array[i] = TD_error + GAMMA * LAMBDA * advantage_array[i+1] * not_terminated_array[i]

            target_array = advantage_array + state_value_array
            advantage_array = (advantage_array - advantage_array.mean()) / (advantage_array.std() + 1e-8)

        indicies = np.arange(len(reward_array))
        #print(len(reward_array))
        loss_over_time.append([])
        entropy_loss.append([])
        clip_loss.append([])
        value_fn_loss.append([])
        for batch in range(NUM_BATCHES):
            random.shuffle(indicies)

            for k in range(0,len(indicies), BATCH_SIZE):
                index = indicies[k:k + BATCH_SIZE]
                
                batch_log_probs = log_probs_array[index].squeeze()
                batch_advantage = advantage_array[index]
                batch_target = target_array[index]
                batch_actions = action_array[index].squeeze()

                batch_states = state_array[index]

                new_log_probs, new_entropy = actor.get_log_probs(batch_states, batch_actions)
                new_state_values = critic(batch_states)

                ratio = torch.exp(new_log_probs - batch_log_probs.detach())
                l_clip = torch.minimum(ratio * batch_advantage.detach(), torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * batch_advantage.detach()).mean()
                l_s = new_entropy.mean()
                l_vf = loss_func(new_state_values, batch_target.detach()).mean()

                actor_optim.zero_grad()
                critic_optim.zero_grad()

                actor_loss = -l_clip - C2 * l_s
                critic_loss = l_vf * C1
                actor_loss.backward()
                critic_loss.backward()

                actor_optim.step()
                critic_optim.step()

                entropy_loss[-1].append(l_s.item())
                clip_loss[-1].append(l_clip.item())
                value_fn_loss[-1].append(l_vf.item())

        state_array = []
        action_array = []
        reward_array = []
        next_state_array = []
        not_terminated_array = []
        log_probs_array = []
        state_value_array = []

    if epoch % LOGGING_FREQ == 0 and epoch != 0:
        print(f"Epoch {epoch} / {NUM_EPOCHS}: Last Reward, {rewards_over_time[-1]:.2f}    Avg. Reward (Last {SLIDING_WINDOW_SIZE}), {np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_SIZE):]):.2f}")
        if np.mean(rewards_over_time[epoch-SLIDING_WINDOW_SIZE:]) > best_avg_reward:
            best_avg_reward = np.mean(rewards_over_time[epoch-SLIDING_WINDOW_SIZE:])
            torch.save(actor.state_dict(), MODEL_FILE_NAME + ".pt")

plt.plot(rewards_over_time, label = "Rewards", color = "blue")
plt.plot([np.mean(rewards_over_time[max(0,epoch-SLIDING_WINDOW_SIZE):epoch+1]) for epoch in range(NUM_EPOCHS)], label = "Average Rewards", color = "red")
plt.legend()
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.title("Rewards During Training")
plt.savefig(MODEL_FILE_NAME + "RewardsGraph.png")

plt.clf()

mean_loss_per_epoch = [np.mean(epoch_loss) for epoch_loss in loss_over_time]
mean_entropy_loss_per_epoch = [np.mean(epoch_loss) for epoch_loss in entropy_loss]
mean_clip_loss_per_epoch = [np.mean(epoch_loss) for epoch_loss in clip_loss]
mean_vf_loss_per_epoch = [np.mean(epoch_loss) for epoch_loss in value_fn_loss]

plt.plot((mean_entropy_loss_per_epoch - np.min(mean_entropy_loss_per_epoch)) / (np.max(mean_entropy_loss_per_epoch) - np.min(mean_entropy_loss_per_epoch)), label = "Scaled Entropy Loss", color = "blue")
plt.plot((mean_clip_loss_per_epoch - np.min(mean_clip_loss_per_epoch)) / (np.max(mean_clip_loss_per_epoch) - np.min(mean_clip_loss_per_epoch)), label = "Scaled Clip Loss", color = "red")
plt.plot((mean_vf_loss_per_epoch - np.min(mean_vf_loss_per_epoch)) / (np.max(mean_vf_loss_per_epoch) - np.min(mean_vf_loss_per_epoch)), label = "Scaled Value Function Loss", color = "green")  
plt.legend()
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Scaled loss")
plt.title("Scaled Loss")
plt.savefig(MODEL_FILE_NAME + "ScaledLoss.png")

plt.clf()

plt.plot(mean_loss_per_epoch, label = "Total Loss", color = 'purple')
plt.plot(-C2 * np.array(mean_entropy_loss_per_epoch), label = "Entropy Loss", color = "blue")
plt.plot(-1 * np.array(mean_clip_loss_per_epoch), label = "Clip Loss", color = "red")
plt.plot(C1 * np.array(mean_vf_loss_per_epoch), label = "Value Function Loss", color = "green")  
plt.legend()
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss During Training")
plt.savefig(MODEL_FILE_NAME + "Loss.png")

print("DONE!!!")
