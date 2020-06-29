import os
import numpy as np
import torch
import random
import pickle

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

exp_dict = pickle.load(open('/home/xzhang17/urban_computing/cGAIL/expert_traj/exp_dict.pkl', 'rb'))

class Env():
    def __init__(self, states, users):
        self.states = states
        self.users = users
        self.user = None
        self.action_space = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    def seed(self, seed=None):
        return np.random.seed(seed)
    def reset(self):
        state, user = random_sample_inputs(self.states, self.users, 1)
        self.user = user
        return state, user
    def step(self, state, action):
        return decide_next_state(action, state, 1)

class Envs():
    def __init__(self, envs):
        self.envs = envs
    def reset(self):
        states, users = [], []
        for i in range(len(self.envs)):
            stemp, utemp = self.envs[i].reset()
            states.append(stemp)
            users.append(utemp)
        return torch.stack(states), torch.stack(users)
    def step(self, state, action):
        return decide_next_state(action, state, 1)
    
def make_env(states, users, seed, rank):
#     def _thunk():
    env = Env(states, users)
    env.seed(seed + rank)
    return env
#     return _thunk


def make_vec_envs(states, users, seed, num_processes, gamma, device):
    envs_temp = [make_env(states, users, seed, i) for i in range(num_processes)]
    return Envs(envs_temp)

class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
        
def decide_next_state(dirc, current_state, timestep=1):
    day = current_state[3]
    if current_state[2] < 289:
        new_time = current_state[2] + timestep
    else:
        new_time = (current_state[2]+timestep)%288
        day += 1
    day = day%7
    
    if dirc == 0:
        new_step = [current_state[0], current_state[1]+1, new_time, day]
    if dirc == 1: 
        new_step = [current_state[0]+1, current_state[1]+1, new_time, day] 
    if dirc == 2:
        new_step = [current_state[0]+1, current_state[1], new_time, day]
    if dirc == 3: 
        new_step = [current_state[0]+1, current_state[1]-1, new_time, day]
    if dirc == 4:
        new_step = [current_state[0], current_state[1]-1, new_time, day]
    if dirc == 5: 
        new_step = [current_state[0]-1, current_state[1]-1, new_time, day]
    if dirc == 6:
        new_step = [current_state[0]-1, current_state[1], new_time, day]
    if dirc == 7:
        new_step = [current_state[0]-1, current_state[1]+1, new_time, day]
    if dirc == 8:
        new_step = [current_state[0], current_state[1], new_time, day]
    if tuple(new_step) in exp_dict:
        return torch.from_numpy(np.asarray(exp_dict[tuple(new_step)]))
    else:
        return None
    
def random_sample_inputs(states, users, length):
    os = random.sample(states, length)
    ou = random.sample(users, length)
    return torch.from_numpy(np.asarray(os)), torch.from_numpy(np.asarray(ou))
