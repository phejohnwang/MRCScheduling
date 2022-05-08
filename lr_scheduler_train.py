# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:47:42 2019

@author: pheno

Supervised training
"""

import copy
import os
import pickle
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hetnet import ScheduleNet4Layer
from utils import ReplayMemory, Transition, action_helper_rollout
from utils import SchedulingEnv, hetgraph_node_helper, build_hetgraph

'''
Fill memory buffer with demonstration data set
    use minDG
'''
def fill_demo_data(folder, start_no, end_no, gamma_d):
    memory = ReplayMemory(1000*20)

    total_no = end_no - start_no + 1
    gurobi_count = 0
    
    for graph_no in range(start_no, end_no+1):
        print('Loading.. {}/{}'.format(graph_no, total_no), end='\r')
        fname = folder + '/%05d' % graph_no
        env = SchedulingEnv(fname)

        # check if the graph is feasible for Gurobi
        solname = folder + 'v9/%05d' % graph_no
        solname_w = solname +'_w.txt'
        
        if os.path.isfile(solname_w):
            gurobi_count += 1
            
            optimals = []
            for i in range(env.num_robots):
                if os.path.isfile(solname+'_%d.txt' % i):
                    optimals.append(np.loadtxt(solname+'_%d.txt' % i, dtype=np.int32))
                else:
                    optimals.append([])
                
            optimalw = np.loadtxt(solname_w, dtype=np.int32)
        else:
            continue
    
        '''
        generate transitions of the problem
        '''
        state_graphs = []
        partials = []
        partialw = []
        actions_task = []
        actions_robot = []
        rewards = []
        terminates = []
        
        state_graphs.append(copy.deepcopy(env.halfDG))
        partials.append(copy.deepcopy(env.partials))
        partialw.append(copy.deepcopy(env.partialw))
        terminates.append(False)

        for i in range(env.num_tasks):
            for j in range(env.num_robots):
                if optimalw[i] in optimals[j]:
                    rj = j
                    break
            
            act_chosen = optimalw[i]
            #print('step: %d, action: [%d, %d]' % (t, act_chosen, rj))
            
            # insert the node, update state, and get reward
            rt, reward, done = env.insert_robot(act_chosen, rj)
            #print(rt, reward, done, env.min_makespan)
            
            state_graphs.append(copy.deepcopy(env.halfDG))
            partials.append(copy.deepcopy(env.partials))
            partialw.append(copy.deepcopy(env.partialw))
            actions_task.append(act_chosen)
            actions_robot.append(rj)
            rewards.append(reward)
            terminates.append(done)
    
        '''
        save transitions into memory buffer
        '''
        for t in range(env.num_tasks):
            curr_g = copy.deepcopy(state_graphs[t])
            curr_partials = copy.deepcopy(partials[t])
            curr_partialw = copy.deepcopy(partialw[t])
            act_task = actions_task[t]
            act_robot = actions_robot[t]
            # calculate discounted reward
            reward_n = 0.0
            for j in range(t, env.num_tasks):
                reward_n += (gamma_d**(j-t)) * rewards[j]
            next_g = copy.deepcopy(state_graphs[t+1])
            next_partials = copy.deepcopy(partials[t+1])
            next_partialw = copy.deepcopy(partialw[t+1])
            next_done = terminates[t+1]
            
            locs = copy.deepcopy(env.loc)
            durs = copy.deepcopy(env.dur)
            
            memory.push(curr_g, curr_partials, curr_partialw,
                        locs, durs,
                        act_task, act_robot,
                        reward_n, next_g, next_partials, 
                        next_partialw, next_done)
    
    print('Gurobi feasible found: {}/{}'.format(gurobi_count, total_no))
    print('Memory buffer size: {}'.format(len(memory)))
    return memory
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--path-to-train', default='./gen/r2t20_001', type=str)
    parser.add_argument('--num-robots', default=2, type=int)
    parser.add_argument('--train-start-no', default=1, type=int)
    parser.add_argument('--train-end-no', default=1000, type=int)
    parser.add_argument('--steps', default=30000, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-5, type=float)
    parser.add_argument('--resume-training', default=False, action='store_true')
    parser.add_argument('--path-to-checkpoint', default='./sltrain02/checkpoint_07000.tar', type=str)
    parser.add_argument('--load-memory', default=False, action='store_true')
    parser.add_argument('--path-to-replay-buffer', default='./buffer/buffer_half_C3.pkl', type=str)
    parser.add_argument('--checkpoint-interval', default=1000, type=int)
    parser.add_argument('--save-replay-buffer-to', default=None, type=str)
    parser.add_argument('--cpsave', default='./cp', type=str)
    args = parser.parse_args()

    resume_training = args.resume_training
    load_memory = args.load_memory
        
    GAMMA = args.gamma
    BATCH_SIZE = args.batch_size
    total_steps = args.steps
    
    loss_history = []

    device = torch.device("cpu") if args.cpu else torch.device("cuda")

    in_dim = {'task': 6,
              'loc': 1,
              'robot': 1,
              'state': 4,
              'value': 1
              }

    hid_dim = {'task': 64,
               'loc': 64,
               'robot': 64,
               'state': 64,
               'value': 64
               }

    out_dim = {'task': 32,
               'loc': 32,
               'robot': 32,
               'state': 32,
               'value': 1
               }

    cetypes = [('task', 'temporal', 'task'),
               ('task', 'located_in', 'loc'), ('loc', 'near', 'loc'),
               ('task', 'assigned_to', 'robot'), ('robot', 'com', 'robot'),
               ('task', 'tin', 'state'), ('loc', 'lin', 'state'),
               ('robot', 'rin', 'state'), ('state', 'sin', 'state'),
               ('task', 'tto', 'value'), ('robot', 'rto', 'value'),
               ('state', 'sto', 'value'), ('value', 'vto', 'value'),
               ('task', 'take_time', 'robot'), ('robot', 'use_time', 'task')]
    
    num_heads = 8
    num_robots = args.num_robots
    map_width = 2
    loc_dist_threshold = 1
    
    policy_net = ScheduleNet4Layer(in_dim, hid_dim, out_dim, cetypes, num_heads).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=1e-5, weight_decay=1e-6)
    #loss_fn = torch.nn.MSELoss()
    lr_scheduler = ReduceLROnPlateau(optimizer,'min', factor = 0.1, 
                                     patience = 800)
    
    if resume_training:
        trained_checkpoint = args.path_to_checkpoint
        cp = torch.load(trained_checkpoint)
        policy_net.load_state_dict(cp['policy_net_state_dict'])
        #target_net.load_state_dict(cp['target_net_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        training_steps_done = cp['training_steps']
        start_step = training_steps_done + 1
    else:
        start_step = 1
        training_steps_done = 0
    
    if load_memory:
        # load replay buffer
        bname = args.path_to_replay_buffer
        with open(bname, 'rb') as f: # open file with read-mode  
            memory = pickle.load(f) # serialize and save object
        print('Memory loaded, length: %d' % len(memory))
    else:
        folder = args.path_to_train
        start_no = args.train_start_no
        end_no = args.train_end_no
        memory = fill_demo_data(folder, start_no, end_no, GAMMA)
    
    print('Initialization done')

    '''
    Training phase
    '''
    #transitions = memory.sample(BATCH_SIZE)
    #batch = Transition(*zip(*transitions))
    for i_step in range(start_step, total_steps+1):
        start_t = time.time()
        policy_net.train()
        print('training no. %d' % i_step)
        
        transitions = memory.sample(BATCH_SIZE)
        #transitions = copy.deepcopy(memory.memory[11:19])
        batch = Transition(*zip(*transitions))
        loss = torch.tensor(0.0).to(device)

        for i in range(BATCH_SIZE):
            num_tasks = batch.curr_g[i].number_of_nodes() - 2
            unsch_tasks = np.array(action_helper_rollout(num_tasks, batch.curr_partialw[i]),
                                   dtype=np.int64)
            
            g = build_hetgraph(batch.curr_g[i], num_tasks, num_robots, batch.durs[i],
                               map_width, np.array(batch.locs[i], dtype=np.int64),
                               loc_dist_threshold, batch.curr_partials[i], unsch_tasks, 
                               batch.act_robot[i], unsch_tasks)
            g = g.to(device)
            
            num_actions = len(unsch_tasks)
            feat_dict = hetgraph_node_helper(batch.curr_g[i].number_of_nodes(), 
                                             batch.curr_partialw[i], 
                                             batch.curr_partials[i],
                                             batch.locs[i], batch.durs[i], 
                                             map_width, num_robots, num_actions)
            
            feat_dict_tensor = {}
            for key in feat_dict:
                feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device)

            outputs = policy_net(g, feat_dict_tensor)
            q_pre = outputs['value']
            
            '''
            Calculate TD loss & LfD loss at the same time
            '''
            if num_actions > 1:
                # q value for alterantive actions                
                offset = 5.0
                target_list = np.full((num_actions, 1), 
                                      batch.reward_n[i] - offset, dtype=np.float32)
            
                LfD_weights = np.full((num_actions, 1), 
                                      0.9/(num_actions-1), dtype=np.float32)
                     
                q_s_a_alt_target1 = q_pre.clone().detach()
                q_s_a_alt_target2 = torch.tensor(target_list).to(device)
                q_s_a_alt_target = torch.min(q_s_a_alt_target1, q_s_a_alt_target2)
                
                # q value for expert action
                expert_idx = 0
                for j in range(num_actions):
                    if unsch_tasks[j] == batch.act_task[i]:
                        expert_idx = j
                        break
                q_s_a_alt_target[expert_idx, 0] = batch.reward_n[i]
                LfD_weights[expert_idx, 0] = 1.0
            else:
                # num_actions == 1
                target_list = np.full((1, 1), batch.reward_n[i], dtype=np.float32)
            
                LfD_weights = np.full((1, 1), 1.0, dtype=np.float32)
                q_s_a_alt_target = torch.tensor(target_list).to(device)
                
            loss_SL = F.mse_loss(q_pre, q_s_a_alt_target, reduction='none')
            LfD_weights = torch.Tensor(LfD_weights).to(device)
            loss_SL = loss_SL * LfD_weights
            loss += loss_SL.sum() / BATCH_SIZE

        loss_batch = loss.data.cpu().numpy()
        
        if i_step > 1:
            lr_scheduler.step(loss_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # tune offset (as in spreadsheet)
        
        loss_history.append(loss_batch)
        end_t = time.time()
        print('[step {}] Loss {:.4f}, time: {:.4f} s'
              .format(i_step, loss_batch, end_t - start_t))        

        '''
        Save checkpoints
        '''
        if i_step % args.checkpoint_interval == 0:
            checkpoint_path = args.cpsave + '/checkpoint_{:05d}.tar'.format(i_step)
            torch.save({
                'training_steps': i_step,
                'policy_net_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history
            }, checkpoint_path)
            print('checkpoint saved')

    # save replay buffer
    if args.save_replay_buffer_to is not None:
        with open(args.save_replay_buffer_to, 'wb') as f:  # open file with write-mode
            pickle.dump(memory, f)  # serialize and save object