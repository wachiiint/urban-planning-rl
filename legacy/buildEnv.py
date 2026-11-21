import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import cv2
import networkx as nx
from icecream import ic
from gym import Env, spaces
import time
from io import BytesIO
import PIL
from IPython.display import display, Image
from collections import deque
from basicClasses import *
'''
buildEnv = build home and target in map
'''

class buildEnv(Env):
    def __init__(self, shape=(10,10,)):
        super(buildEnv, self).__init__()
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("Environment: Incorrect shape")
        shape = tuple(shape)
        self.observation_shape = shape
        self.observation_space = spaces.Box(low=0, high = 4, shape = tuple(shape), dtype = np.uint8)
        self.obs = self.observation_space.sample() * 0
        max_act_code = np.array([self.observation_shape[0]-1, self.observation_shape[1]-1],dtype= np.uint8)
        self.action_space = spaces.Discrete(self.act_encoder(max_act_code))
        
    def get_obs_meaning(self):
        return {"0":"ground","1":"obstacle","2":"road","3":"home","4":"target"}
    def act_encoder(self,act):
        nActType = 2
        shape = self.observation_shape
        if len(act) != 2:
            raise ValueError("Action should be list of length 2")
        for i in range(2):
            # if isinstance(act[i],np.uint8):
            #     raise ValueError("Action index{} is not int".format(i))
            if(act[i]>=shape[i]):
                raise ValueError("Action index{} out of range".format(i))
            
        message = act[0]*shape[1]+act[1]
        return message
    def act_decoder(self, code):
        nActType = 2
        shape = self.observation_shape
        max_code = self.act_encoder([shape[0]-1, shape[1]-1])
        if (code<0) or (code> max_code):
            raise ValueError("Code Out of Range")
        act = [code//shape[1], code%shape[1]]
        return tuple(act)
    def reset(self, requirements = {'map':None, 'home': 0, 'target': 0}):

        #version3: only obstacle can be initial state
        self.obs = self.observation_space.sample() * 0
        self.total_time = 0
        self.home_list = []
        self.target_list = []
        self.road_list = []
        r,c = self.observation_shape
        area = r*c

        #home and target init
        n_home = int(requirements['home'])
        n_target = int(requirements['target'])
        
        desired_h = 0.13 #probability of requirement h
        desired_t = 0.07 #probability of requirement h
        scale = area//100
        
        if n_home == 0:
            n_home = int(np.random.normal(desired_h*area, scale))
            n_home = max(1,n_home)
        if n_target == 0:
            n_target = int(np.random.normal(desired_t*area, scale))
            n_target = max(1,n_target)

        self.rem_home = n_home
        self.rem_target = n_target
        custom_map = requirements['map'] 
        if isinstance(custom_map, np.ndarray):
            self.obs = np.array(custom_map).astype(np.uint8)
        elif custom_map == None:
            # full random
            desired_obstacle = 0.03
            n_obstacle = int(np.random.normal(desired_obstacle*area, area//100))
            n_obstacle = max(1,n_obstacle)

            pos_vector = [(i,j) for i in range(r) for j in range(c)]
            pos_index = np.arange(len(pos_vector))

            pos_obstacle = np.random.choice(pos_index, n_obstacle, replace = False)
            for ele in pos_obstacle: 
                self.obs[pos_vector[ele]] = 1
                # pos_index = pos_index[pos_index!=ele]
            self.obs = self.obs.astype(np.uint8) 
        elif custom_map == []:
            n_obstacle = 0
        else:
            raise ValueError("custom_map is either numpy.ndarray, [], or None")
        return self.obs
    
    def step(self,actionCode,debug= False):
        
        self.total_time+=1
        area = self.observation_shape[0] * self.observation_shape[1]
        action = self.act_decoder(actionCode)
        row,col = action
        pos = row,col
        reward = 0.0

        info = {}
        is_bad_action = False
        if self.obs[pos] != 0:
            reward = area*(-30.0)
            is_bad_action = True
            info['action'] = 'bad'
        
        else:    
            if self.rem_home >  0: 
                info['action'] = 'home'
                self.obs[row,col] = 3
                self.rem_home -= 1
                self.home_list.append(Home(row,col))
                reward += self.zone_home((row,col))
            elif self.rem_target > 0:
                info['action'] = 'target'
                self.obs[row,col] = 4
                self.rem_target -= 1
                self.target_list.append(Target(row,col))
                reward += self.zone_target((row,col))

        if self.total_time>=1.0*area:
            done = True
            info['status'] = 'timed out'
            if debug: 
                ic('timed out')
        elif self.rem_home * self.rem_target == 0:
            done = True
            info['status'] = 'success'
            if debug:
                ic('finished')
        elif is_bad_action:
            done = False
            info['status'] = 'bad action'
            if debug:
                ic('Bad action',reward,self.total_time)
        elif self.phase == 1:
            done = False
            info['status'] = 'phase 1'
            if debug:
                ic(info['action'], pos, reward,self.total_time)

        return self.obs, reward, done, info

    def render(self,mode = 'RGB'):
        #mode = 'RGB' or 
        v = list(self.observation_shape)
        u = v+[3]
        u = np.array(u)
        u = u*32
        u[2]/=32    
        bg = np.zeros(u)
        img = PIL.Image.fromarray(bg, 'RGB')
        pic0 = PIL.Image.open('../Rendering Image/Grass.png')
        pic1 = PIL.Image.open('../Rendering Image/Obstacle.png')
        pic2 = PIL.Image.open('../Rendering Image/Road.png')
        pic3 = PIL.Image.open('../Rendering Image/Home.png')
        pic4 = PIL.Image.open('../Rendering Image/Building.png')
        pic5 = PIL.Image.open('../Rendering Image/Car.png')
        pic6 = PIL.Image.open('../Rendering Image/CarKee.png')
        for i in range(v[0]):
            for j in range(v[1]):
                if self.obs[i][j] == 0:
                    img.paste(pic0, (32*j,32*i))
                elif self.obs[i][j] == 1:
                    img.paste(pic1, (32*j,32*i))
                elif self.obs[i][j] == 2:
                    img.paste(pic2, (32*j,32*i))
                elif self.obs[i][j] == 3:
                    img.paste(pic3, (32*j,32*i))
                elif self.obs[i][j] == 4:
                    img.paste(pic4, (32*j,32*i))
        for c in self.car_list:
            pos = c.get_pos()
            i,j = pos
            if self.obs[i][j] == 2:
                if len(self.car_space[i][j]) == 1:
                    img.paste(pic5, (32*j,32*i))
                elif len(self.car_space[i][j]) == 2:
                    img.paste(pic6, (32*j,32*i))
        #for PILLOW
        # return img

        #for cv2 in google colab
        open_cv_image = np.array(img) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        return open_cv_image

    def zone_home(self,pos):
        M55 = [(-2, -2), (-2, -1),(-2, 0),(-2, 1),(-2, 2),(-1, -2),(-1, 2),(0, -2),(0, 2),(1, -2),(1, 2),(2, -2),(2, -1),(2, 0),(2, 1),(2, 2)]
        for m in M55:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[0]) and (0<=j and j< self.obs.shape[1]):
                if self.obs[cur] == 3:
                    return 20
        return 0

    def zone_target(self,pos):
        M33 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        M55 = [(-2, -2), (-2, -1),(-2, 0),(-2, 1),(-2, 2),(-1, -2),(-1, 2),(0, -2),(0, 2),(1, -2),(1, 2),(2, -2),(2, -1),(2, 0),(2, 1),(2, 2)]
        M77 = [(-3, -3),(-3, -2),(-3, -1),(-3, 0),(-3, 1),(-3, 2),(-3, 3),(-2, -3),(-2, 3),(-1, -3),(-1, 3),(0, -3),(0, 3),(1, -3),(1, 3),(2, -3),(2, 3),(3, -3),(3, -2),(3, -1),(3, 0),(3, 1),(3, 2),(3, 3)]
        point = 0
        for m in M55:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[0]) and (0<=j and j< self.obs.shape[1]):
                if self.obs[cur] == 4:
                    point += 20
        
        for m in M33:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[0]) and (0<=j and j< self.obs.shape[1]):
                if self.obs[cur] == 3:
                    return point-30
        for m in M55:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[0]) and (0<=j and j< self.obs.shape[1]):
                if self.obs[cur] == 3:
                    return point-20
        for m in M77:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[0]) and (0<=j and j< self.obs.shape[1]):
                if self.obs[cur] == 3:
                    return point-10
        
        return point

