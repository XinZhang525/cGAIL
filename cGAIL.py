import pickle
import pandas as pd
import csv
import re
import time
import os
import numpy as np
import itertools
from numpy import mean
from copy import deepcopy
from matplotlib import pyplot as plt

from scipy import stats
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
import random
import scipy.optimize

import statistics
from a2c_ppo_acktr.arguments import get_args


args = get_args()
tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros

device = torch.device("cuda:0" if args.cuda else "cpu")
STATE_DIM = 25
ACTION_DIM = 10
USER_DIM = 24

train_file_name = os.path.join(
        args.experts_dir, "expert_traj.pkl")
test_file_name = os.path.join(
    args.experts_dir, "test_traj.pkl")
ground_file_name = os.path.join(
    args.experts_dir, "exp_loc.pkl")

expert_st, expert_ur, expert_ac = pickle.load(open(train_file_name, 'rb'))
train_load = data_utils.TensorDataset(torch.from_numpy(np.asarray(expert_st)), 
                                      torch.from_numpy(np.asarray(expert_ur)), 
                                      torch.from_numpy(np.asarray(expert_ac))) 
train_loader = torch.utils.data.DataLoader(train_load, batch_size=args.gail_batch_size, shuffle=True)

test_st, test_ur, test_ac = pickle.load(open(test_file_name, 'rb'))
test_load = data_utils.TensorDataset(torch.from_numpy(np.asarray(test_st)), 
                                      torch.from_numpy(np.asarray(test_ur)), 
                                      torch.from_numpy(np.asarray(test_ac))) 
test_loader = torch.utils.data.DataLoader(test_load, batch_size=args.gail_batch_size, shuffle=True)
exp_loc = pickle.load(open(ground_file_name, 'rb'))

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

class Net(nn.Module):
    def __init__(self, obs_shape, action_space, cond_space):
        super(Net, self).__init__()
        self.prefc1 = nn.Linear(cond_space, obs_shape)
        self.conv1 = nn.Conv2d(6, 20, 2) 
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(20, 30, 2)
        self.fc1 = nn.Linear(30, 120) 
        self.bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, action_space)
        
    def forward(self, state, user):
        user = self.prefc1(user).view(user.size(0), -1)
        x = state.view(state.size(0), 5, 5, 5)
        x = torch.cat((state, user), dim=1).view(state.size(0), 6, 5, 5)
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pool(F.leaky_relu(self.conv2(x), 0.2))
        x = x.view(-1, 30)
        x = F.leaky_relu(self.bn(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def select_action(self, state, user):
        action_prob = self.forward(state, user)
        action = action_prob.multinomial(1)
        return action
    
    def targeting_prob(self, state, user, labels):
        action_prob = self.forward(state, user)
        return action_prob.gather(1, labels)

class Dis(nn.Module):
    def __init__(self, state_dim, action_dim, user_dim, device, lr):
        super(Dis, self).__init__()

        self.device = device
        self.label_embedding = nn.Embedding(10, 10)
        self.prefc1 = nn.Linear(action_dim, 25)
        
        self.linear = nn.Linear(state_dim*6+action_dim, 81)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(2, 20, 3)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(180, 120)
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
        x = x.view(-1, 180)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return torch.sigmoid(x)


def random_sample_inputs2(states, users, user_IDs, length):
    sts = random.sample(states, length)
    urs = []
    ur_IDs = random.sample([23, 5, 48]*1000, length)
    for i in ur_IDs:
        urs.append(user_info[i])
    return torch.from_numpy(np.asarray(sts)), torch.from_numpy(np.asarray(urs))

def cross_entropy(target, ground_truth): # actually the KL-divergence
    epsilon = 1e-12
    ce = 0.
    target = target.copy()
    ground_truth = ground_truth.copy()
    ces = []
    ce2s = []
    for state in range(len(ground_truth)):
        target_prime = np.clip(target[state], epsilon, 1.-epsilon)
        ground_truth[state] = np.clip(ground_truth[state], epsilon, 1.-epsilon)
        t = np.sum(ground_truth[state]*np.log((target_prime/ground_truth[state])))
        ce -= t
        ces.append(t)
        
    return ce/(len(target)), ces

net = Net(STATE_DIM, ACTION_DIM, USER_DIM)
dis = Dis(STATE_DIM, ACTION_DIM, USER_DIM, device, lr=args.D_lr)

lr = 2e-4
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

dtype = torch.float32
torch.set_default_dtype(dtype)
temp_diff = 2.2
temp_net = 0
temp_dis = 0
for epoch in range(2000):
    running_loss = 0.
    dis_loss = 0.
    for i, data in enumerate(train_loader, 0):
        '''expert inputs , user info, and labels(actions)'''
        inputs, user, labels = data
        inputs = inputs.float()
        user = user.float()
        labels = labels.long()
        
        batch_size = inputs.size(0)
        
        '''generate actions'''
        fak_labels = net.select_action(inputs, user)
        fak_prob = net.targeting_prob(inputs, user, fak_labels)

        
        '''updating net'''
        optimizer.zero_grad()
        loss = (-torch.log(fak_prob)).mean()
        loss.backward()
        optimizer.step()
                    
        
        running_loss += loss.item()
        if i % 150 == 149:
            print('[{}, {}] generator loss: {}'.format((epoch+1), i+1, running_loss/150))
            print('--------------------')
            running_loss = 0.        
            if epoch % 5 == 4:
                out_loc = {}
                for i, data in enumerate(test_loader, 0):
                    inputs, user, labels = data
                    inputs = inputs.float()
                    user = user.float()
                    labels = labels.long()
                    output = net.select_action(inputs, user).tolist()
                    for i in range(inputs.size(0)):
                        x = int(inputs[i][0].item())
                        y = int(inputs[i][1].item())

                        if (x, y) not in out_loc:
                            out_loc[(x, y)] = np.zeros(10)
                            out_loc[(x, y)][output[i]] += 1
                        else:
                            out_loc[(x, y)][output[i]] += 1
                target = []
                ground = []
                for key in out_loc:
                    o1 = out_loc[key].copy()
                    o1 /= sum(o1)
                    if key in exp_loc:
                        o2 = np.zeros(10)
                        for b, w in exp_loc[key].items():
                            o2[b] += w
                        o2 /= sum(o2)
                        target.append(o1)
                        ground.append(o2)

                k, kls = cross_entropy(target, ground)
                print(k)
