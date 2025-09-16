import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_layers = [128,128], std_clamp_min = -12, std_clamp_max = 2, discrete = False, action_range = 1):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.clamp_min = std_clamp_min
        self.clamp_max = std_clamp_max
        self.action_range = action_range
        self.is_discrete = discrete
        
        self.head = nn.Linear(obs_dim, hidden_layers[0])
        self.hidden_layer = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
        if self.is_discrete:
            self.tail = nn.Linear(hidden_layers[-1], action_dim)
        else:
            self.tail = nn.Linear(hidden_layers[-1], action_dim * 2)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.head(state)
        x = self.relu(x)
        for layer in self.hidden_layer:
            x = layer(x)
            x = self.relu(x)
        x = self.tail(x)

        if self.is_discrete:
            distribution = Categorical(logits = x)
        else:
            distribution = Normal(x[:,:self.action_dim], torch.exp(torch.clamp(x[:,self.action_dim:], self.clamp_min, self.clamp_max)))

        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        
        if not self.is_discrete:
            actions = self.action_range * torch.tanh(actions)
            log_probs -= torch.log(self.action_range * (1 - (actions / self.action_range).pow(2) + 1e-6)).sum(dim=-1, keepdim=True)

        if self.training:
            return actions, log_probs.sum(dim = -1, keepdim=True)
        else:
            return actions
    
    def get_log_probs(self, state, action):
        x = self.head(state)
        x = self.relu(x)
        for layer in self.hidden_layer:
            x = layer(x)
            x = self.relu(x)
        x = self.tail(x)
        action_clamped = torch.clamp(action / self.action_range, -1.0 + 1e-6, 1.0 - 1e-6)
        action_pre_tanh = torch.atanh(action_clamped)
        if self.is_discrete:
            distribution = Categorical(logits = x)
        else:
            distribution = Normal(x[:,:self.action_dim], torch.exp(torch.clamp(x[:,self.action_dim:], self.clamp_min, self.clamp_max)))

        log_prob = distribution.log_prob(action_pre_tanh) - torch.log(self.action_range * (1 - action_clamped.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

        return log_prob.sum(dim = -1, keepdim=True), distribution.entropy()
    
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_layers = [128,128]):
        super(Critic, self).__init__()
        self.action_dim = action_dim
        self.head = nn.Linear(obs_dim, hidden_layers[0])
        self.hidden_layer = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
        self.tail = nn.Linear(hidden_layers[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.head(state)
        x = self.relu(x)
        for layer in self.hidden_layer:
            x = layer(x)
            x = self.relu(x)
        x = self.tail(x)
        return x
    
def get_networks(obs_dim, action_dim, actor_hidden_layers = [128,128], critic_hidden_layers = [128,128], std_clamp_min = -12, std_clamp_max = 2, discrete = False, action_range = 1):
    actor = Actor(obs_dim, action_dim, actor_hidden_layers, std_clamp_min, std_clamp_max, discrete, action_range)
    critic = Critic(obs_dim,action_dim, critic_hidden_layers)
    return actor, critic

