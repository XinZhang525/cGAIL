import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, cond_space):
        super(Policy, self).__init__()
        self.prefc1 = nn.Linear(cond_space, obs_shape)
        self.conv1 = nn.Conv2d(6, 20, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(20, 30, 3)
        self.fc1 = nn.Linear(30, 120) 
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, action_space)
        
        self.critic_linear = nn.Linear(action_space, 1)
        
    def forward(self, state, user):
        user = self.prefc1(user).view(user.size(0), -1)
        x = state.view(state.size(0), 5, 5, 5)
        x = torch.cat((state, user), dim=1).view(state.size(0), 6, 5, 5)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pool(F.leaky_relu(self.conv2(x), 0.2))
        x = x.view(-1, 30)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def act(self, state, user):
        action_prob = self.forward(state, user)
        
        value = self.critic_linear(action_prob)
        action = action_prob.multinomial(1)
        logprob = torch.log(action_prob.gather(1, action))
        return value, action, logprob
    
    def targeting_prob(self, state, user, labels):
        action_prob = self.forward(state, user)
        return action_prob.gather(1, labels)

    def get_value(self, state, user):
        action_prob = self.forward(state, user)
        value = self.critic_linear(action_prob)
        return value

    def evaluate_actions(self, state, user, action):
        value, action, action_log_probs = self.act(state, user)
        
        action_prob = self.targeting_prob(state, user, action)
        dist_entropy = (-action_prob*action_log_probs).mean()

        return value, action_log_probs, dist_entropy

