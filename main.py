import copy
import glob
import os
import time
from collections import deque
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import cgail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.envs import decide_next_state
from a2c_ppo_acktr.envs import make_vec_envs
from evaluation import cross_entropy
import torch.utils.data as data_utils

STATE_DIM = 25
ACTION_DIM = 10
USER_DIM = 24
def main():
    args = get_args()

    torch.manual_seed(args.seed)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    envs = make_vec_envs(args.seed, args.num_processes,
                         args.gamma, device, False)
    
    actor_critic = Policy(STATE_DIM, ACTION_DIM, USER_DIM)
    actor_critic.to(device)

    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch,
        args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
        lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)
    if args.cgail:
        discr = cgail.Discriminator(
            STATE_DIM, ACTION_DIM, USER_DIM, device, lr=args.D_lr)
    else:
        discr = gail.Discriminator(
            STATE_DIM, ACTION_DIM,
            device, lr=args.D_lr)
        
    train_file_name = os.path.join(
        args.experts_dir, "expert_traj.pkl")
    test_file_name = os.path.join(
        args.experts_dir, "test_traj.pkl")
    
    expert_st, expert_ur, expert_ac = pickle.load(open(train_file_name, 'rb'))
    train_load = data_utils.TensorDataset(torch.from_numpy(np.asarray(expert_st)), 
                                          torch.from_numpy(np.asarray(expert_ur)), 
                                          torch.from_numpy(np.asarray(expert_ac))) 
    gail_train_loader = torch.utils.data.DataLoader(train_load, batch_size=args.gail_batch_size, shuffle=True)
    
    test_st, test_ur, test_ac = pickle.load(open(test_file_name, 'rb'))
    test_load = data_utils.TensorDataset(torch.from_numpy(np.asarray(test_st)), 
                                          torch.from_numpy(np.asarray(test_ur)), 
                                          torch.from_numpy(np.asarray(test_ac))) 
    test_loader = torch.utils.data.DataLoader(test_load, batch_size=args.gail_batch_size, shuffle=True)



    rollouts = RolloutStorage(args.num_steps, args.num_processes, STATE_DIM*5, ACTION_DIM)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    result_log = []

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step], rollouts.user[step])

            # Obser reward and next obs
            if action.item() != 9:
                obs, reward, done, infos = decide_next_state(action)
                rollouts.insert(obs, user, action, action_log_prob, value)

                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.user[-1]).detach()

        gail_epoch = args.gail_epoch
        if j < 10:
            gail_epoch = 100  # Warm up
        for _ in range(gail_epoch):
            discr.update(gail_train_loader, rollouts)

        for step in range(args.num_steps):
            if cgail:
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.user[step], rollouts.actions[step], args.gamma)
            else:
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo, str(args.lr), str(args.gail_batch_size), "entropy_" + str(args.entropy_coef), "D_lr"+str(args.D_lr))
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(actor_critic, os.path.join(save_path, args.env_name + "_ac_{}.pt".format(j)))
            torch.save(discr, os.path.join(save_path, args.env_name + "_D_{}.pt".format(j)))
            pickle.dump(result_log, open(os.path.join(save_path, args.env_name + "_result_log.pkl"), 'wb'))

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            
            result_log.append([np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)])
            out_loc = {}
            for i, data in enumerate(test_loader, 0):
                inputs, user, labels = data
                inputs = inputs.float()
                user = user.float()
                labels = labels.long()
                output = actor_critic.act(inputs, user).tolist()

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
            k, c, kls = cross_entropy(target, ground)
            print(k, c)
            diff = difference(target, ground)


if __name__ == "__main__":
    main()
