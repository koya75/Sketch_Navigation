import os
import numpy as np
import torch
import mlagents
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import cv2
import random

def map_point(device, num_sketch):
    pattern = []
    for idx in range(num_sketch):
        pattern.append(torch.tensor(np.loadtxt('sketch/map_hsr{}.csv'.format(idx+1), delimiter=","), dtype=torch.float32, device=device))#11
    pattern = torch.stack(pattern, 0)
    waypoint = pattern.detach().clone()
    waypoint[:,:,0] = (waypoint[:,:,0] -  63)/12
    waypoint[:,:,1] = -(waypoint[:,:,1] -  63)/12
    return pattern, waypoint

def indoor_point(device, num_sketch):
    pattern = []
    for idx in range(num_sketch):
        pattern.append(torch.tensor(np.loadtxt('sketch/indoor_bird_hsr{}.csv'.format(idx+1), delimiter=","), dtype=torch.float32, device=device))#11
    pattern = torch.stack(pattern, 0)
    waypoint = pattern.detach().clone()
    waypoint[:,:,0] = (waypoint[:,:,0] -  49)/10
    waypoint[:,:,1] = -(waypoint[:,:,1] -  49)/10
    return waypoint, waypoint


class unity_env:
    def __init__(self, file_name, worker_id, time_scale, seed, device, num_sketch):
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)
        if 'Box' in file_name:
            self.make_point = map_point(device, num_sketch)
        elif 'Indoor' in file_name:
            self.make_point = indoor_point(device, num_sketch)
        file_name = os.path.abspath('..')+'/'+file_name
        self.env = UnityEnvironment(file_name=file_name, seed=seed, worker_id=worker_id, side_channels=[channel])
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.num_envs = decision_steps.obs[0][:].shape[0]
        self.device = device

    def reset(self, rand):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        x_pos = torch.tensor(decision_steps.obs[1][:,2], dtype=torch.float32, device=self.device)
        y_pos = torch.tensor(decision_steps.obs[1][:,3], dtype=torch.float32, device=self.device)
        obs = [self.preproc_img(decision_steps.obs[0][:]), torch.tensor(decision_steps.obs[1][:,0], dtype=torch.float32, device=self.device), x_pos, y_pos]
        spec = self.env.behavior_specs[self.behavior_name]
        #self.image_save(decision_steps.obs[0][0])
        self.info = {'crash_count':0 }
        pattern, self.way = self._make_point(rand)
        return obs, pattern

    def step(self, action, t):
        action_tuple = ActionTuple()
        action_tuple.add_discrete(action.cpu().numpy())
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, _ = self.env.get_steps(self.behavior_name)
        x_pos = torch.tensor(decision_steps.obs[1][:,2], dtype=torch.float32, device=self.device)
        y_pos = torch.tensor(decision_steps.obs[1][:,3], dtype=torch.float32, device=self.device)
        obs = [self.preproc_img(decision_steps.obs[0][:]), torch.tensor(decision_steps.obs[1][:,0], dtype=torch.float32, device=self.device), x_pos, y_pos]

        distance = torch.tensor(decision_steps.obs[1][:,1], dtype=torch.float32, device=self.device)
        crash = decision_steps[0].reward
        
        if crash == -1:
            self.info['crash_count']+=1

        dis_reward =  -1.0 * distance #1.0 / (1.0 + distance*distance)

        x_dis = torch.square(self.way[:,t,0] - x_pos)
        y_dis = torch.square(self.way[:,t,1] - y_pos)

        way_reward = -1.0 * torch.sqrt(x_dis + y_dis)
        #reward = way_reward
        #reward = dis_reward + way_reward

        return obs, dis_reward, way_reward

    def preproc_img(self, img):
        img = img.transpose((0,3,1,2))
        return torch.tensor(img, dtype=torch.float32, device=self.device)

    def image_save(self, img):
        cv2.imwrite('Indoor.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)*255)

    def close(self):
        self.env.close()

    def _make_point(self, rand):
        pattern, way_pattern = self.make_point
        point, waypoint = [], []
        for f in rand:
            point.append(pattern[f.item()][:])
            waypoint.append(way_pattern[f.item()][:])
        points = torch.stack(point, 0)
        way_points = torch.stack(waypoint, 0)
        return way_points.detach().clone(), way_points.detach().clone()