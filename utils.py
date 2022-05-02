# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:58:41 2020

@author: pheno

Utility functions

1. Replace floyd_warshall with Johnson's for STN preprocessing
"""


import copy
import random
from collections import Counter
from collections import namedtuple

import dgl
import networkx as nx
import numpy as np
import torch

from benchmark.JohnsonUltra import johnsonU


def build_hetgraph(halfDG, num_tasks, num_robots, dur, map_width, locs, loc_dist_threshold,
                   partials, unsch_tasks, selected_robot, valid_tasks):
    """
    Helper function for building HetGraph
    Q nodes are built w.r.t selected_robot & unsch_tasks
        valid_tasks: available tasks filtered from unsch_tasks
        
    Args:
        loc_dist_threshold: Distance threshold for two locations to be connected by an edge
    """

    num_locs = map_width * map_width
    num_values = len(valid_tasks)
    
    num_nodes_dict = {'task': num_tasks + 2,
                      'loc': num_locs,
                      'robot': num_robots,
                      'state': 1,
                      'value': num_values}

    # Serializing [x, y] locations to 1D array
    # E.g. [1, 1] => 0 * width + 0 = 0
    #      [2, 1] => 0 * width + 1 = 1
    serialized_locs = [(locs[i, 1] - 1) * map_width + locs[i, 0] - 1 for i in range(locs.shape[0])]

    # Sort the nodes and assign an index to each one
    task_name_to_idx = {node: idx for idx, node in enumerate(sorted(halfDG.nodes))}
    task_edge_to_idx = {(from_node, to_node): idx for idx, (from_node, to_node) in enumerate(halfDG.edges)}

    loc_near_data = []
    # Find neighbors of each location
    for i in range(num_locs):
        loc_near_data.append((i, i))
        for j in range(i + 1, num_locs):
            i_x, i_y = i % map_width, i // map_width
            j_x, j_y = j % map_width, j // map_width
            if (i_x - j_x) ** 2 + (i_y - j_y) ** 2 <= loc_dist_threshold ** 2:
                loc_near_data.append((i, j))
                loc_near_data.append((j, i))

    # List of (task id, robot id) tuples
    task_to_robot_data = []

    for rj in range(num_robots):
        # add f0
        task_to_robot_data.append((0, rj))
        # add si (including s0)
        for i in range(len(partials[rj])):
            ti = partials[rj][i].item()
            task_id = ti + 1
            task_to_robot_data.append((task_id, rj))

    unsch_task_to_robot = []
    for rj in range(num_robots):
        for t in unsch_tasks:
            task_id = t + 1
            unsch_task_to_robot.append((task_id, rj))

    robot_com_data = [(i, j) for i in range(num_robots) for j in range(num_robots)]

    data_dict = {
        ('task', 'temporal', 'task'): (
            # Convert named edges to indexes
            [task_name_to_idx[from_node] for from_node, _ in halfDG.edges],
            [task_name_to_idx[to_node] for _, to_node in halfDG.edges],
        ),
        ('task', 'located_in', 'loc'): (
            list(range(2, num_tasks + 2)),
            serialized_locs,
        ),
        ('loc', 'near', 'loc'): (
            [i for i, _ in loc_near_data],
            [j for _, j in loc_near_data],
        ),
        ('task', 'assigned_to', 'robot'): (
            [task for task, _ in task_to_robot_data],
            [robot for _, robot in task_to_robot_data],
        ),
        ('task', 'take_time', 'robot'): (
            [task for task, _ in unsch_task_to_robot],
            [robot for _, robot in unsch_task_to_robot],
        ),
        ('robot', 'use_time', 'task'): (
            [robot for _, robot in unsch_task_to_robot],
            [task for task, _ in unsch_task_to_robot],
        ),
        ('robot', 'com', 'robot'): (
            [i for i, _ in robot_com_data],
            [j for _, j in robot_com_data],
        ),
        # 4. Add graph summary nodes
        # [task] — [in] — [state]
        ('task', 'tin', 'state'): (
            list(range(num_tasks + 2)),
            np.zeros(num_tasks + 2, dtype=np.int64),
        ),
        # [loc] — [in] — [state]
        ('loc', 'lin', 'state'): (
            list(range(num_locs)),
            np.zeros(num_locs, dtype=np.int64),
        ),
        # [robot] — [in] — [state]
        ('robot', 'rin', 'state'): (
            list(range(num_robots)),
            np.zeros(num_robots, dtype=np.int64),
        ),
        # [state] — [in] — [state] self-loop
        ('state', 'sin', 'state'): (
            [0],
            [0],
        ),
        # 5.1 Q value node
        # [task] — [to] — [value]
        ('task', 'tto', 'value'): (
            valid_tasks + 1,
            list(range(num_values)),
        ),
        # [robot] — [to] — [value]
        ('robot', 'rto', 'value'): (
            np.full(num_values, selected_robot, dtype=np.int64),
            list(range(num_values)),
        ),
        # [state] — [to] — [value]
        ('state', 'sto', 'value'): (
            np.zeros(num_values, dtype=np.int64),
            list(range(num_values)),
        ),
        # [value] — [to] — [value] self-loop
        ('value', 'vto', 'value'): (
            list(range(num_values)),
            list(range(num_values)),
        ),
    }

    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, idtype=torch.int64)

    # Store data of edges by index, as DiGraph.edges.data does not guarantee to have exactly the same
    # ordering as Digraph.edges
    temporal_edge_weights = torch.zeros((len(halfDG.edges), 1), dtype=torch.float32)
    # Unpack indexes of edge weights
    weights_idx = [task_edge_to_idx[from_node, to_node] for from_node, to_node, _ in halfDG.edges.data('weight')]
    # Put weights in tensor according to their indexes
    temporal_edge_weights[weights_idx, :] = torch.tensor([[weight] for _, _, weight in halfDG.edges.data('weight')],
                                                      dtype=torch.float32)
    graph.edges['temporal'].data['weight'] = temporal_edge_weights

    takes_time_weight = torch.zeros((len(unsch_task_to_robot), 1), dtype=torch.float32)
    for idx, (task, robot) in enumerate(unsch_task_to_robot):
        # Subtract 2 because task 1's node id is 2, but has index 0 in dur
        takes_time_weight[idx] = dur[task - 2, robot]
    graph.edges['take_time'].data['t'] = takes_time_weight
    # Ordering of takes_time and uses_time edges are exactly the same
    graph.edges['use_time'].data['t'] = takes_time_weight.detach().clone()

    return graph

def hetgraph_node_helper(number_of_nodes, curr_partialw, curr_partials,
                         locations, durations, map_width, num_robots, num_values):
    """
    Generate initial node features for hetgraph
    The input of hetgraph is a dictionary of node features for each type
        number_of_nodes: numer of nodes in half distance graph (halfDG)
        curr_partialw: partial solution/whole
        curr_partials: partial solution/seperate
        locations: np array locations
        durations: np array task durations
        map_width: map grid size
        num_robots: number of robots
        num_values: number of actions / Q values
    Return
        feat_dict: node features stored in a dict
    """
    feat_dict = {}
    num_locations = map_width * map_width

    # Task features.
    # For scheduled tasks, the feature is [1 0 dur 0 dur 0]
    # For unscheduled ones, the feature is [0 1 min max-min mean std]
    feat_dict['task'] = np.zeros((number_of_nodes, 6))

    max_dur, min_dur = durations.max(axis=1), durations.min(axis=1)
    mean_dur, std_dur = durations.mean(axis=1), durations.std(axis=1)

    # f0
    feat_dict['task'][0, 0] = 1

    # s0~si. s0 has index 1
    for i in range(1, number_of_nodes):
        ti = i-1
        if ti in curr_partialw:
            feat_dict['task'][i, 0] = 1
            if ti > 0:
                # Ignore s0
                for j in range(num_robots):
                    if ti in curr_partials[j]:
                        rj = j
                        break              
                
                feat_dict['task'][i, [2, 4]] = durations[ti-1][rj]
        else:
            feat_dict['task'][i] = [0, 1, min_dur[ti-1], max_dur[ti-1] - min_dur[ti-1], 
                                    mean_dur[ti-1], std_dur[ti-1]]
    
    # [loc]
    feat_dict['loc'] = np.zeros((num_locations, 1))
    serialized_locs = [(locations[i, 1] - 1) * map_width + locations[i, 0] - 1 for i in range(locations.shape[0])]
    loc_counter = Counter(serialized_locs)
    for i in range(num_locations):
        # number of tasks in location
        feat_dict['loc'][i, 0] = loc_counter[i]
    
    # [robot]
    feat_dict['robot'] = np.zeros((num_robots, 1))
    for i in range(num_robots):
        # number of tasks assigned so far
        # including s0
        feat_dict['robot'][i, 0] = len(curr_partials[i])
    
    # [state]
    feat_dict['state'] = np.array((number_of_nodes-1, len(curr_partialw),
                                   num_locations, num_robots)).reshape(1,4)
    
    # [value]
    feat_dict['value'] = np.zeros((num_values, 1))

    return feat_dict


'''
Env class for maintaining current partial solution and updated graph
    during data collection process
'''
class SchedulingEnv(object):
    # read problem info specified by fname
    def __init__(self, fname):
        # load constraints
        self.dur = np.loadtxt(fname+'_dur.txt', dtype=np.int32)
        self.ddl = np.loadtxt(fname+'_ddl.txt', dtype=np.int32)
        self.wait = np.loadtxt(fname+'_wait.txt', dtype=np.int32)
        self.loc = np.loadtxt(fname+'_loc.txt', dtype=np.int32)
        
        self.num_tasks = self.dur.shape[0]
        self.num_robots = self.dur.shape[1]
        
        self.M = self.num_tasks * 10.0 # infeasible reward token
        self.C = 3.0 # discount factor for reward calculation

        # reshape if shape is one-dimension, meaning there is only one constraint
        if len(self.ddl) > 0 and len(self.ddl.shape) == 1:
            self.ddl = self.ddl.reshape(1, -1)
    
        if len(self.wait) > 0 and len(self.wait.shape) == 1:
            self.wait = self.wait.reshape(1, -1)  
        
        self.max_deadline = self.num_tasks * 10

        # initial partial solution with t0
        # t0 appears in all partial schedules
        self.partials = []
        for i in range(self.num_robots):
            self.partials.append(np.zeros(1, dtype=np.int32))
        
        self.partialw = np.zeros(1, dtype=np.int32)
        
        # maintain a graph with min/max duration for unscheduled tasks
        self.g = self.initialize_STN()
        
        # get initial min make span
        success, min_makespan = self.check_consistency_makespan()
        if success:
            self.min_makespan = min_makespan
        else:
            print('Initial STN infeasible.')
    
    def initialize_STN(self):
        # Initialize directed graph    
        DG = nx.DiGraph()
        DG.add_nodes_from(['s000', 'f000'])
        DG.add_edge('s000', 'f000', weight = self.max_deadline)
                
        # Add task nodes
        for i in range(1, self.num_tasks+1):
            # Add si and fi
            si = 's%03d' % i
            fi = 'f%03d' % i
            DG.add_nodes_from([si, fi])
            DG.add_weighted_edges_from([(si, 's000', 0),
                                        ('f000', fi, 0)])
        
        # Add task durations
        for i in range(self.num_tasks):
            si = 's%03d' % (i+1)
            fi = 'f%03d' % (i+1)
            dur_min = self.dur[i].min().item()
            dur_max = self.dur[i].max().item()
            DG.add_weighted_edges_from([(si, fi, dur_max),
                                        (fi, si, -1 * dur_min)])
        
        # Add deadlines
        for i in range(self.ddl.shape[0]):
            ti, ddl_cstr = self.ddl[i]
            fi = 'f%03d' % ti
            DG.add_edge('s000', fi, weight = ddl_cstr)            
            
        # Add wait constraints
        for i in range(self.wait.shape[0]):
            ti, tj, wait_cstr = self.wait[i]
            si = 's%03d' % ti
            fj = 'f%03d' % tj
            DG.add_edge(si, fj, weight = -1 * wait_cstr)
        
        return DG
    
    '''
    Check consistency and get min make span
        Also creates the half min graph
    '''
    def check_consistency_makespan(self, updateDG = True):
        consistent = True
        try:
            p_ultra, d_ultra = johnsonU(self.g)
        except Exception as e:
            consistent = False
            print('Infeasible:', e)
                
        '''
        Makespan
        Only consider the last finish time of scheduled tasks
        '''
        if consistent:        
            if len(self.partialw) == 1:
                min_makespan = 0.0
            else:
                tmp = []
                for i in range(1,len(self.partialw)):
                    ti = self.partialw[i]
                    fi = 'f%03d' % ti
                    tmp.append(-1.0 * d_ultra[fi]['s000'])
    
                tmp_np = np.array(tmp)
                min_makespan = tmp_np.max()
        else:
            min_makespan = self.M
            return consistent, min_makespan
        
        if not updateDG:
            return consistent, min_makespan
        
        '''
        Min distance graph & Half min graph
        '''
        juDG = nx.DiGraph()
        for i in range(0, self.num_tasks+1):
            # Add si and fi
            si = 's%03d' % i
            fi = 'f%03d' % i
            # minDG.add_nodes_from([si, fi])
            if i == 0:
                juDG.add_nodes_from([si, fi])
            else:
                juDG.add_node(si)
        
        # add shortest path distance edges
        for k_start in d_ultra:
            for k_end in d_ultra[k_start]:
                #print(key_start, key_end)
                # check if path is inf
                if d_ultra[k_start][k_end] < 9999:
                    # minDG.add_edge(k_start, k_end, 
                    #                weight = d_ultra[k_start][k_end])
                    if juDG.has_node(k_start) and juDG.has_node(k_end):
                        juDG.add_edge(k_start, k_end,
                                      weight = d_ultra[k_start][k_end])
        
        # self.minDG = minDG
        self.halfDG = juDG
        
        return consistent, min_makespan
    
    '''          
    ti is task number 1~num_tasks
    rj is robot number 0~num_robots-1
    append ti to rj's partial schedule
    also update the STN
    '''
    def insert_robot(self, ti, rj, diff = 1.0, updateDG = True):
        # sanity check
        if rj < 0 or rj >= self.num_robots:
            print('invalid insertion')
            return False        
        
        # find tj and update partial solution
        # tj is the last task of rj's partial schedule
        # insert ti right after tj
        tj = self.partials[rj][-1]
        self.partials[rj] = np.append(self.partials[rj], ti)
        self.partialw = np.append(self.partialw, ti)

        # update graph
        # insert ti after tj, no need to add when tj==0    
        # no need to insert if a wait constraint already exists
        if tj != 0:
            si = 's%03d' % ti
            fj = 'f%03d' % tj
            if not self.g.has_edge(si, fj):
                self.g.add_edge(si, fj, weight = 0)
        
        '''
        [New] Also, replace the task duration of ti with actual duration
        '''
        si = 's%03d' % ti
        fi = 'f%03d' % ti
        ti_dur = self.dur[ti-1][rj]
        # this will rewrite previous edge weights
        self.g.add_weighted_edges_from([(si, fi, ti_dur),
                                        (fi, si, -1 * ti_dur)])
        
        '''
        make sure the start time of all unscheduled tasks is no earlier thant si
        '''
        for k in range(1, self.num_tasks+1):
            if k not in self.partialw:
                # tk starts no earlier than si
                # si <= sk, si-sk<=0, sk->si:0
                si = 's%03d' % ti
                sk = 's%03d' % k
                if not self.g.has_edge(sk, si):
                    self.g.add_edge(sk, si, weight = 0)

        '''
        make sure the start time of all unscheduled tasks that
        are within the allowed distance (diff) happen after fi
        '''
        for k in range(1, self.num_tasks+1):
            if k not in self.partialw:
                xi, yi = self.loc[ti-1]
                xk, yk = self.loc[k-1]
                dist_2 = (xi - xk) * (xi - xk) + (yi - yk) * (yi - yk)               
                
                if dist_2 <= diff * diff:
                    # tk starts after fi
                    # fi <= sk, fi-sk <=0, sk->fi:0
                    fi = 'f%03d' % ti
                    sk = 's%03d' % k
                    if not self.g.has_edge(sk, fi):
                        self.g.add_edge(sk, fi, weight=0)

        # calculate reward for this insertion
        success, reward = self.calc_reward_discount(updateDG)
        # check done/termination
        if success==False:
            done = True
        elif (self.partialw.shape[0]==self.num_tasks+1):
            done = True
        else:
            done = False
        
        return success, reward, done
    
    '''
    Reward R of a state-action pair is defined as the change
        in objective values after taking the action,
        
        R = −1 × (Zt+1 − Zt).
        
        divide Zt by a factor D > 1 if xt is not a termination state

        Z(infeasible) = M
    '''
    def calc_reward_discount(self, updateDG = True):
        success, min_makespan = self.check_consistency_makespan(updateDG)
        # feasible
        if success:
            # if last step
            if self.partialw.shape[0]==(self.num_tasks+1):
                delta = min_makespan - self.min_makespan/self.C
            # disounted delta
            else:
                delta = (min_makespan - self.min_makespan)/self.C
        # infeasible
        else:
            delta = self.M - self.min_makespan/self.C
            min_makespan = self.M
        
        reward = -1.0 * delta
        
        self.min_makespan = min_makespan
        return success, reward

    '''
    Return unscheduled tasks given partialw
    '''
    def get_unscheduled_tasks(self):
        unsch_tasks = []
        for i in range(1, self.num_tasks+1):
            if i not in self.partialw:
                unsch_tasks.append(i)
        
        return np.array(unsch_tasks)

    def get_duration_on_tasks(self, robot, tasks):
        """Returns durations of a robot on a list of tasks.
        Task ids should be 1-indexed, and robot id should be 0-indexed
        """
        assert min(tasks) > 0, 'Tasks should be 1-indexed'
        assert 0 <= robot < self.num_robots, 'Robot should be 0-indexed'

        task_ids = [task - 1 for task in tasks]
        return self.dur[task_ids, robot]

    '''
    Return unscheduled tasks given partialw
        plus checking if the task can starts at current timepoint
    '''
    def get_valid_tasks(self, timepoint):
        valid_tasks = []
        for i in range(1, self.num_tasks+1):
            if i not in self.partialw:
                # check task start time
                # si->s0: A
                # s0 - si <= A
                # si >= -A
                si = 's%03d' % i
                time_si = -1.0 * self.halfDG[si]['s000']['weight']
                # time_si is the earliest time task i can happen
                if time_si <= timepoint:
                    valid_tasks.append(i)
        
        return np.array(valid_tasks)
    
    '''
    Return an updated min robot STN
        with task duration (valid unscheduled tasks) 
        replaced with the task duration of chosen robot
        plus consistency check
    '''
    def get_rSTN(self, robot_chosen, valid_task):
        rSTN = copy.deepcopy(self.g)
        # modify STN
        for i in range(len(valid_task)):
            ti = valid_task[i]
            si = 's%03d' % ti
            fi = 'f%03d' % ti
            ti_dur = self.dur[ti-1][robot_chosen]
            rSTN.add_weighted_edges_from([(si, fi, ti_dur),
                                          (fi, si, -1 * ti_dur)])       
        
        # check consistency
        consistent = True    
        try:
            p_ultra, d_ultra = johnsonU(rSTN)
        except Exception as e:
            consistent = False
            print('Infeasible:', e) 

        if consistent:    
            # get min STN
            min_rSTN = nx.DiGraph()
            for i in range(0, self.num_tasks+1):
                # Add si and fi
                si = 's%03d' % i
                fi = 'f%03d' % i
                min_rSTN.add_nodes_from([si, fi])
            
            # add shortest path distance edges
            for k_start in d_ultra:
                for k_end in d_ultra[k_start]:
                    # check if path is valid
                    if d_ultra[k_start][k_end] < 9999:
                        min_rSTN.add_edge(k_start, k_end, 
                                       weight = d_ultra[k_start][k_end])        
            
            return min_rSTN, True
        else:
            return None, False


'''
Transition for n-step
state
    curr_g: networkx graph updated with current solution
    curr_partials: partial solution as a list of numpy arrays (int)
        [sd0 sd1 ...]
            sd0: partial schedule of robot 0
            sd1: partial schedule of robot 1
            ......
    curr_partialw: partial schedule of all tasks selected
    locations: the location of each task
    durations: the duration of each task
action
    act_task: ti
    act_robot: rj
        append ti to rj's partial schedule
reward
    reward_n: total future discounted rewards
state after 1-step
    next_g: networkx graph
    next_partial: next partial solution
termination
    next_done: if True, means the next state is a termination state
        one episode finishes
        1. finish with feasible solution
        2. stop with infeasible partial
'''
Transition = namedtuple('Transition',
                        ('curr_g', 'curr_partials', 'curr_partialw',
                         'locs', 'durs',
                         'act_task', 'act_robot',
                         'reward_n', 'next_g', 'next_partials',
                         'next_partialw', 'next_done'))

'''
Replay buffer
'''
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # Saves a transition
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
Enumerate all possible insertions (rollout version) based on
    num_tasks: number of total tasks 1~N
    curr_partialw: partial solution
Return
    act_task: list of all possible insertions
'''
def action_helper_rollout(num_tasks, curr_partialw):
    act_task = []
    # pick a task t_i from {unallocated}
    for i in range(1, num_tasks + 1):
        if i not in curr_partialw:
            act_task.append(i)

    return np.array(act_task)
        
if __name__ == '__main__':
    # problem path
    fname = 'gen/r2t20_001/00013'
    solname = 'gen/r2t20_001v9/00013'
    # initialize env
    env = SchedulingEnv(fname)
    # env.g is the original STN
    print(env.g.nodes())
    print(env.g.number_of_edges())
    # env.halfDG is the simplified graph to be used for graph construction
    print(sorted(env.halfDG.nodes()))
    print(env.halfDG.number_of_edges())
    
    # load solution
    optimals = []
    for i in range(env.num_robots):
        optimals.append(np.loadtxt(solname+'_%d.txt' % i, dtype=np.int32))
    optimalw = np.loadtxt(solname+'_w.txt', dtype=np.int32)
    
    for i in range(env.num_robots):
        print(optimals[i])

    print(optimalw)
    
    #optimalw[8] = 12
    #optimalw[9] = 7
    
    print('Initial makespan: ', env.min_makespan)
    # check gurobi solution
    rs = []
    for i in range(len(optimalw)):
        for j in range(env.num_robots):
            if optimalw[i] in optimals[j]:
                rj = j
                break
          
        rt, reward, done = env.insert_robot(optimalw[i], rj)
        rs.append(reward)
        print('Insert %d, %d' % (optimalw[i], rj))
        print('No. Edges: %d' % env.halfDG.number_of_edges())
        print('Returns: ', rt, reward, done, env.min_makespan)
        if not rt:
            print('Infeasible!')
            break
        
    print(env.partialw)
    print(sum(rs))
    print('test passed')    