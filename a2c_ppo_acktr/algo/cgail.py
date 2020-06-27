import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, user_dim, device, lr):
        super(Discriminator, self).__init__()

        self.device = device

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        
        self.label_embedding = nn.Embedding(10, 10)
        self.prefc1 = nn.Linear(action_dim, 25)
        
        self.linear = nn.Linear(state_dim*6+action_dim, 81)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 20, 2)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state, user, label):
        label = label.view(label.size(0))
        user = self.prefc1(user).view(user.size(0), -1)
        x = state.view(state.size(0), -1)
        x = torch.cat((state, user), dim=1).view(state.size(0), -1)
        x = torch.cat((x.view(x.size(0), -1), self.label_embedding(label)), dim=1)
        x = self.relu(self.linear(x))
        x = x.view(x.size(0), 1, 9, 9)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, 20)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return torch.sigmoid(x)

    def update(self, expert_loader, rollouts):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_user, policy_action = policy_batch[0], policy_batch[1], policy_batch[2]
            policy_d = self.forward(policy_state, policy_user, policy_action.float())

            expert_state, expert_user, expert_action = expert_batch
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_user = expert_user.view((expert_user.shape[0], -1)).to(self.device)
            expert_action = expert_action.view((expert_state.shape[0], -1)).to(self.device)
            expert_d = self.forward(expert_state, expert_user, expert_action)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d, torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d, torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss

            loss += gail_loss.item()
            n += 1

            self.optimizer.zero_grad()
            gail_loss.backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, user, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.forward(state, user, action.float())
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
