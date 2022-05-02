# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:05:04 2019

@author: pheno

Utils for EDF
"""

import sys
import numpy as np

sys.path.append('../')
from utils import SchedulingEnv

'''
Task class
'''
class Task(object):
    def __init__(self, t_id, s_time, e_time):
        self.id = t_id
        self.start_time = s_time
        self.end_time = e_time

'''
Robot class
'''
class Robot(object):
    def __init__(self, r_id):
        self.id = r_id
        self.schedule = []
        self.next_available_time = 0

'''
Robot team class
'''
class RobotTeam(object):
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.robots = [Robot(i) for i in range(num_robots)]

    # Return a robot id that is available at a given time point
    # new: return all the available robots
    # Return [] if none is available
    def pick_robot(self, timepoint):
        available = []
        for i in range(self.num_robots):
            if self.robots[i].next_available_time <= timepoint:
                available.append(self.robots[i].id)
        
        return available

    def pick_robot_by_min_dur(self, time, env: SchedulingEnv, version, exclude=[]):
        """Returns the robot with minimum average duration on unscheduled tasks for v1,
        min duration on any one unscheduled task for v2,
        min average duration on valid tasks for v3
        """
        dur_and_robot = []  # List of (average duration, robot id) tuples

        if version == 'v3':
            tasks = env.get_valid_tasks(time)
        else:
            tasks = env.get_unscheduled_tasks()
        if len(tasks) == 0:
            return None

        for i in range(self.num_robots):
            if self.robots[i].id not in exclude:
                if self.robots[i].next_available_time <= time:
                    dur = env.get_duration_on_tasks(self.robots[i].id, tasks)
                    if version == 'v2':
                        dur_and_robot.append((min(dur), self.robots[i].id))
                    else:
                        dur_and_robot.append((sum(dur) / len(dur), self.robots[i].id))

        # No robot is available
        if len(dur_and_robot) == 0:
            return None

        return min(dur_and_robot)[1]

    def __len__(self):
        return len(self.robots)
    
    # Update the status of robot after scheduling the chosen task
    def update_status(self, task_chosen, robot_chosen, task_dur, t):
        self.robots[robot_chosen].schedule.append(Task(task_chosen, t, t + task_dur))
        self.robots[robot_chosen].next_available_time = t + task_dur  

    # print all robots' schedules
    def print_schedule(self):
        for i in range(self.num_robots):
            print('Robot %d' % self.robots[i].id)
            for task in self.robots[i].schedule:
                print('Task (%d,%d,%d)'%(task.id, task.start_time, task. end_time))
                
                
'''
Pick a task that has the earlist deadline
    minDG: APSP graph
    act_task: unscheduled tasks
'''
def pick_task(minDG, act_task, timepoint):
    length = len(act_task)
    if length == 0:
        return -1
    
    tmp = np.zeros(length, dtype=np.float32)
    
    for i in range(length):
        ti = act_task[i]
        #si = 's%03d' % ti
        fi = 'f%03d' % ti
        # pick the task with the earlist possible finish time
        tmp[i] = -1.0 * minDG[fi]['s000']['weight']
    
    idx = np.argmin(tmp)
    task_chosen = act_task[idx]
    
    sk = 's%03d' % task_chosen
    time_sk = -1.0 * minDG[sk]['s000']['weight']
    if time_sk <= timepoint:
        return task_chosen
    else:
        return -1

if __name__ == '__main__':
    t = Task(4, 3, 5)
    print(t.id, t.start_time, t.end_time)

    r = Robot(10)
    print(r.id)