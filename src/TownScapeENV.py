import numpy as np
import networkx as nx
from icecream import ic
from gymnasium import Env, spaces
from io import BytesIO
from PIL import Image
# from IPython.display import display, Image
import IPython.display as ipyd
from collections import deque
from queue import PriorityQueue
import os
import pathlib

ASSET_DIR = pathlib.Path(__file__).resolve().parent / "assets"
'''
this is townScape env
//change -> look map as bool
//change -> force build road to target with nearest walk
'''

class townScape(Env):
    def __init__(self, shape=(10,10,)):
        super(townScape, self).__init__()
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("Environment: Incorrect shape")
        shape = tuple(shape)
        self.observation_shape = (5, *shape)
        self.observation_space = spaces.Box(low=0, high = 1, shape = self.observation_shape, dtype = np.uint8)
        #agent can not destroy building

        max_act_code = np.array([self.observation_shape[1]-1, self.observation_shape[2]-1],dtype= np.uint8)
        self.action_space = spaces.Discrete(self.act_encoder(max_act_code))
        
        # self.obs = self.observation_space.sample() * 0
        # self.roadG = nx.create_empty_copy(nx.grid_2d_graph(*shape, periodic=False, create_using=nx.DiGraph))
    def get_compressed_obs(self):
        maze = self.obs * 1
        for i in range(5):
            maze[i]*=i
        maze = np.sum(maze,axis=0)
        return maze
    def get_obs_meaning(self):
        return {"0":"ground","1":"obstacle","2":"road","3":"home","4":"target"}
    def act_encoder(self,act):
        nActType = 2
        shape = self.observation_shape[1],self.observation_shape[2]
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
        shape = self.observation_shape[1],self.observation_shape[2]
        max_code = self.act_encoder([shape[0]-1, shape[1]-1])
        if (code<0) or (code> max_code):
            raise ValueError("Code Out of Range")
        act = [code//shape[1], code%shape[1]]
        return tuple(act)
    def reset(self, *, seed=None, options=None, requirements=None):
        super().reset(seed=seed)
        if requirements is None:
            requirements = (options or {}).get('requirements', {'map': None, 'home': 0, 'target': 0})

        #version3: only obstacle can be initial state
        self.obs = self.observation_space.sample() * 0
        self.total_road = 0
        self.total_time = 0
        self.total_used_road = 0
        self.phase = 1 #1->build home/target #2->build road

        self.home_list = []
        self.target_list = []
        self.road_list = []
        self.car_list = []
        self.car_space = []
        
        shape = self.observation_shape[1],self.observation_shape[2]
        self.roadG = nx.create_empty_copy(nx.grid_2d_graph(*shape , periodic=False, create_using=nx.DiGraph))

        self.used_road = [[0 for j in range(self.observation_shape[2])] for i in range(self.observation_shape[1])]
        self.has_ent = np.zeros(shape, dtype = bool)

        r,c = self.observation_shape[1],self.observation_shape[2]
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
        self.obs = np.zeros(shape = (r,c))
        if isinstance(custom_map, np.ndarray):
            self.obs = np.array(custom_map, dtype = np.uint8)
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
        elif custom_map == []:
            n_obstacle = 0
        else:
            raise ValueError("custom_map is either numpy.ndarray, [], or None")
        
        self.obs = np.array([self.obs==i for i in range(5)], dtype = np.uint8)
        
        return self.obs, {}
    
    def step(self,actionCode,debug= False):
        buildLimit = 5
        # ic(self.total_time)
        self.total_time+=1
        area = self.observation_shape[1] * self.observation_shape[2]
        action = self.act_decoder(actionCode)
        row = action[0]
        col = action[1]
        pos = row,col
        reward = 0.0
        suc_r = 0
        info = {}
        is_bad_action = False
        if self.obs[0,row,col] == 0 and self.phase == 1: #not empty space
            reward = area*(-30.0)
            is_bad_action = True
            info['action'] = 'bad'
        
        else:    
            if self.rem_home == 0 and self.rem_target == 0:
                if self.phase == 1:
                    self.phase = 2
                    #initialize the graph
                    shape = self.observation_shape[1],self.observation_shape[2]
                    self.roadG = nx.grid_2d_graph(*shape, periodic = False,create_using=nx.DiGraph)
                    toDel = []
                    for e in self.roadG.edges:
                        u,v = e
                        if self.obs[4][u]==1 or self.obs[3][v]==1 or self.obs[1][u] == 1 or self.obs[1][v] ==1:
                            toDel.append(e)
                        elif (self.obs[3][u]==1 and self.obs[4][v]==1) or (self.obs[4][u]==1 and self.obs[3][v]==1):
                            toDel.append(e)
                        else:
                            nx.set_edge_attributes(self.roadG, {e: {'cost':100}})
                    for e in toDel:
                        self.roadG.remove_edge(*e)
                    nx.set_edge_attributes(self.roadG, False, 'isRoad')
                    # pairing
                    ran_t_list = np.random.choice(self.target_list,len(self.target_list),replace = False)
                    for i,h in enumerate(self.home_list):
                        if i<len(self.target_list):
                            t = ran_t_list[i]
                        else:
                            t = np.random.choice(self.target_list)
                        h.set_target(t)
                        t.leaf.append(h)

                    reachable = self.bfs()
                    # ic(reachable)
                    for h,val in reachable:
                        if not val:
                            # del self.roadG.nodes[h.pos]['pointer']
                            self.home_list.remove(h)
                    if debug:
                        ic('done pairing', self.home_list)

                #phase2 normal
                try:
                    min_home = self.home_list[0]
                    for h in self.home_list:
                        if dist(h.pos, (row,col))< dist(min_home.pos, (row,col)):
                            min_home = h
                    if debug: ic(min_home)
                    cur_home = min_home
                    if not cur_home.finish:
                        cnt = 0
                        pathBuild = nx.astar_path(self.roadG, cur_home.pos, cur_home.target.pos, heuristic=dist,weight= 'cost')
                        pathBuild = pathBuild[1:-1]
                        # while i<len(pathBuild):
                        for i,pos in enumerate(pathBuild):
                            pos = pathBuild[i]
                            if self.obs[0][pos] == 1:
                                cnt+=1
                                self.addRoad(*pos)
                                self.total_road+=1
                                self.road_list.append(pos)
                                if cnt==buildLimit:
                                    break
                        
                        for h in self.home_list:
                            if not h.finish:
                                c = Car(h.pos,h.target.pos)
                                if c.get_path(self.roadG)!=None:
                                    h.finish = True
                        info['action'] = f'Road = {cnt}'
                        reward += self.zone_road((row,col))
                    else:
                        info['action'] = 'skip'
                except:
                    info['action'] = 'skip'
            elif self.rem_home >  0: 
                info['action'] = 'home'
                self.obs[:,row,col] = 0
                self.obs[3,row,col] = 1
                self.rem_home -= 1
                self.home_list.append(Home(row,col))
                # nx.set_node_attributes(self.roadG, {(row,col): {'pointer': self.home_list[-1]}})
                reward += self.zone_home((row,col))
            elif self.rem_target > 0:
                info['action'] = 'target'
                self.obs[:,row,col] = 0
                self.obs[4,row,col] = 1
                self.rem_target -= 1
                self.target_list.append(Target(row,col))
                # nx.set_node_attributes(self.roadG, {(row,col): {'pointer': self.target_list[-1]}})
                reward += self.zone_target((row,col))

        if self.phase == 2 and not is_bad_action:
            if info['action'] == 'skip':
                reward = 0
                if debug:
                    ic('skipped',reward, self.total_time)
            else:
                #Reward calculation 
                tf, suc_c, suc_r, over_road, sim_info = self.simulate()
                web, iso_h_ratio, iso_t_ratio = self.connectivity()
                if(web==0): 
                    web = 10

                if sim_info == "blank" or len(tf) == 0:
                    sim_reward = 0
                elif sim_info == 'square' :
                    sim_reward = 0
                    info['action'] = 'SQUARE'
                    
                else:
                    sim_reward = np.median(np.array(tf))*300+ suc_c*50
                
                con_reward = max(0,6-web)*20 
                con_reward += iso_h_ratio*30+iso_t_ratio*30
                reward += sim_reward + con_reward
                if debug :
                    ic(reward, sim_reward, con_reward, self.total_time)
        truncated = False
        if self.total_time>=0.6*area:
            done = True
            info['status'] = 'timed out'
            if debug: 
                ic('timed out')
        elif info['action'] == 'SQUARE':
            done = True
            reward = -area*10.0
            info['status'] = 'SQUARE'
            if debug:
                ic('square!!!')
                display_img(self.render())
        elif suc_r==1.0 :
            reward+=area*10.0
            done = True
            info['status'] = 'success'
            if debug:
                ic('success')
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
        else:
            done = False
            info['status'] = 'phase 2'
        #observation reward terminate truncuate info
        '''terminated= True if environment terminates (eg. due to task completion, failure etc.)
        truncated= True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.'''

        return self.obs, reward, done, truncated, info

    def render(self,mode = 'RGB'):
        #mode = 'RGB' or 
        v = (self.observation_shape[1], self.observation_shape[2])
        u = np.array([v[0], v[1], 3])
        u[0:2] *=32
        
        img = Image.new('RGB', (int(u[1]), int(u[0])))
        pic0 = Image.open(ASSET_DIR / 'Grass.png')
        pic1 = Image.open(ASSET_DIR / 'Obstacle.png')
        pic2 = Image.open(ASSET_DIR / 'Road.png')
        pic3 = Image.open(ASSET_DIR / 'Home.png')
        pic4 = Image.open(ASSET_DIR / 'Building.png')
        pic5 = Image.open(ASSET_DIR / 'Car.png')
        pic6 = Image.open(ASSET_DIR / 'CarKee.png')
        for i in range(v[0]):
            for j in range(v[1]):
                if self.obs[0,i,j] == 1:
                    img.paste(pic0, (32*j,32*i))
                elif self.obs[1,i,j] == 1:
                    img.paste(pic1, (32*j,32*i))
                elif self.obs[2,i,j] == 1:
                    img.paste(pic2, (32*j,32*i))
                elif self.obs[3,i,j] == 1:
                    img.paste(pic3, (32*j,32*i))
                elif self.obs[4,i,j] == 1:
                    img.paste(pic4, (32*j,32*i))
        
        for c in self.car_list:
            pos = c.get_pos()
            i,j = pos
            if self.obs[2,i,j] == 1:
                if len(self.car_space[i][j]) == 1:
                    img.paste(pic5, (32*j,32*i))
                elif len(self.car_space[i][j]) == 2:
                    img.paste(pic6, (32*j,32*i))
        #for PILLOW
        # return img

        #for cv2 in google colab
        open_cv_image = np.array(img) 
        # Convert RGB to BGR 
        # open_cv_image = open_cv_image[:, :, ::-1].copy() 
        return open_cv_image

    def connectivity(self):
        maze = self.obs * 1
        for i in range(5):
            maze[i]*=i
        maze = np.sum(maze,axis=0)

        h_list = [h.pos for h in self.home_list]
        t_list = [t.pos for t in self.target_list]
        r_list = [r for r in self.road_list]
        n_graph = 0
        visited = []
        while len(r_list)>0:
            n_graph += 1
            q = deque()
            q.append(r_list[0])
            r_list.pop(0)
            while len(q)>0:
                cur_r, cur_c = q.popleft()
                visited.append((cur_r, cur_c))
                move = [(-1,0),(0,-1),(0,1),(1,0)]
                # ic(cur)
                for i in move:
                    r,c = cur_r+i[0], cur_c+i[1]
                    # ic(r,c)
                    if (0<=r and r<self.observation_shape[1]) and (0<=c and c<self.observation_shape[2]) and (r,c) not in visited:
                        if maze[r,c] == 2:
                            # ic('Road')
                            maze[r,c] = 0
                            q.append((r,c))
                            if (r,c) in r_list: r_list.remove((r,c))
                        elif maze[r,c] == 3:
                            # ic('home')
                            if (r,c) in h_list: h_list.remove((r,c))
                            visited.append((r,c))
                        elif maze[r,c] == 4:
                            # ic('target')
                            if (r,c) in t_list: t_list.remove((r,c))
                            visited.append((r,c))
        # return {'n_graph':n_graph, 'iso_h_ratio': len(h_list)/len(self.home_list), 'iso_t_ratio':len(t_list)/len(self.target_list)}
        if len(self.home_list) == 0:
            iso_h_rate = 0
        else:
            iso_h_rate = len(h_list)/len(self.home_list)
        if len(self.target_list) == 0:
            iso_t_rate = 0
        else:
            iso_t_rate = len(t_list)/len(self.target_list)
        return n_graph, iso_h_rate, iso_t_rate

    def simulate(self,render=False,debug = False):    
        time = 0
        self.car_list = []
        traffic_flow = []
        success_cnt = 0 #number of car that can be travel to
        img_list = []
        # nopath_factor = -0.2
        # blankArea_factor = -0.5
        
        if len(self.home_list)*len(self.target_list) == 0:
            return 0, 0, 0, 0, 'blank'
        elif self.phase == 1:
            return 0, 0, 0, 0, 'phase1'

        self.car_space = [[list() for j in range(self.observation_shape[2])] for i in range(self.observation_shape[1])]
        for i,h in enumerate(self.home_list):

            # t = random.choice(self.target_list)

            # c = Car(tuple(h.pos), tuple(h.target))
            c = Car(h.pos, h.target.pos)

            if c.get_path(self.roadG) != None:
                self.car_list.append(c)
                self.car_space[h.pos[0]][h.pos[1]].append(c)
            else:
                continue
        try:
            f = 0
            while(len(self.car_list)>0):
                time+=1
                # ic(time)
                tf = 0.0
                temp = []
                moved = 0
                for i,c in enumerate(self.car_list):
                    c.get_path(self.roadG)
                    # willremove = False
                    cur = c.path[0]
                    # ic(c.pos, c.path)
                    if(cur == c.target): #finish
                        self.car_space[cur[0]][cur[1]].remove(c)
                        success_cnt+=1
                    else:
                        # ic(c.path)
                        next = c.path[1]
                        if len(self.car_space[next[0]][next[1]]) == 0:
                            c.move()
                            if(self.used_road[next[0]][next[1]]==0 and self.obs[2,next[0],next[1]] == 1):
                                self.used_road[next[0]][next[1]]= 1 
                                self.total_used_road+=1
                                # ic(next[0],next[1])
                            moved+=1
                            self.car_space[cur[0]][cur[1]].remove(c)
                            self.car_space[next[0]][next[1]].append(c)
                            # c.get_path(self.obs)
                        elif len(self.car_space[next[0]][next[1]])==1 and self.obs[4,next[0]][next[1]] ==0:
                            other_c = self.car_space[next[0]][next[1]][0]
                            # ic(c.pos, other_c.pos)
                            # img = self.render()
                            # cv2_imshow(img)
                            # ic(self.obs)
                            # ic(c.path, other_c.path)
                            
                            if ask_direction(other_c.path[1],other_c.path[0])*ask_direction(c.path[1],c.path[0])==-2: #ERROR: index out of range
                                c.move()
                                if(self.used_road[next[0]][next[1]]==0 and self.obs[2,next[0],next[1]] == 1):
                                    self.used_road[next[0]][next[1]]= 1 
                                    self.total_used_road+=1
                                    # ic(next[0],next[1])
                                moved+=1
                                self.car_space[cur[0]][cur[1]].remove(c)
                                self.car_space[next[0]][next[1]].append(c)
                                # c.get_path(self.obs)
                    # if not willremove:
                        temp.append(c)
                tf += moved/len(self.car_list)
                if not self.total_road :
                    tf*= len(self.car_list) / 1
                else:
                    tf *= len(self.car_list) / self.total_road
                
                self.car_list = temp
                traffic_flow.append(tf)
                if render:
                    img = self.render()
                    img_list.append(img)
                    
                if(moved == 0) and len(self.car_list)>1: #DEBUG
                    if f == 1:
                        # for c in self.car_list:
                        #     ic(c.pos, c.path)
                        # display_img(self.render())
                        # ic(self.roadG.edges)
                        raise RuntimeError("Timed out no movement")
                    else:
                        f = 1
                else: f = 0
        except:
            if render:
                return traffic_flow, success_cnt, success_cnt/len(self.home_list), img_list, 'square'
            else:
                return traffic_flow, success_cnt, success_cnt/len(self.home_list), 0, 'square'
        over_road = self.total_road-self.total_used_road
        if render:
            return [traffic_flow, success_cnt, success_cnt/len(self.home_list), img_list, None]
        return [traffic_flow, success_cnt, success_cnt/len(self.home_list), over_road, None]
    def bfs(self):
        # Please USE NETWORKX
        result = []
        for h in self.home_list:
            s = h.pos
            t = h.target.pos
            try:
                pathBuild = nx.astar_path(self.roadG, h.pos, h.target.pos, heuristic=dist,weight= 'cost')
                result.append((h, True))
            except:
                result.append((h, False))
        
        return result
        # maze = self.get_compressed_obs()
        # result = []
        # M = [(-1,0), (0,-1),(0,1),(1,0)]
        # for h in self.home_list:
        #     q = PriorityQueue()
        #     s = h.pos
        #     t = h.target.pos
        #     q.put((0,s))
        #     visited = [s]
        #     f = 0
        #     while not q.empty() and f==0:
        #         g,cur =  q.get()
        #         for i,j in M:
        #             r, c = cur[0]+i,  cur[1]+j
        #             if 0<=r and r<self.observation_shape[1] and (0<=c and c<self.observation_shape[2]) and (r,c) not in visited:
        #                 if (r,c) == t:
        #                     result.append((h, g))
        #                     f = 1
        #                     break
        #                 else:
        #                     if maze[r,c]==2:
        #                         q.put((g, (r,c)))
        #                         visited.append((r,c))
        #                     elif maze[r,c]==0:
        #                         q.put((g+1, (r,c)))
        #                         visited.append((r,c))
        #     if f==0:
        #         result.append((h, -1))
        # return result
    def set_target(self, t_list):
        if len(t_list)!=len(self.home_list):
            raise ValueError("unmatched t_list to home_list")
        for i,h in enumerate(self.home_list):
            h.set_target(Target(*t_list[i]))

    def addRoad(self,r,c):
        pos = r,c
        self.obs[:,r,c] = 0
        self.obs[2][pos] = 1
        shape = self.obs.shape[1], self.obs.shape[2]
        for m in [[-1,0],[0,-1],[0,1],[1,0]]: 
            i,j = pos[0]+m[0], pos[1]+m[1]
            if (0<=i and i< shape[0]) and (0<=j and j< shape[1]):
                if(self.obs[2][i,j]==1):
                    nx.set_edge_attributes(self.roadG, {((i,j),pos): {'cost':1}},)
                    nx.set_edge_attributes(self.roadG, {((i,j),pos): {'isRoad':True}},)
                    nx.set_edge_attributes(self.roadG, {(pos,(i,j)): {'cost':1}},)
                    nx.set_edge_attributes(self.roadG, {(pos,(i,j)): {'isRoad':True}},)
                elif(self.obs[0][i,j]==1):
                    nx.set_edge_attributes(self.roadG, {((i,j),pos): {'cost':50}},)
                elif(self.obs[3][i,j]==1):
                    nx.set_edge_attributes(self.roadG, {((i,j),pos): {'cost':1}},)
                    nx.set_edge_attributes(self.roadG, {((i,j),pos): {'isRoad':True}},)
                elif(self.obs[4][i,j]==1):
                    nx.set_edge_attributes(self.roadG, {(pos,(i,j)): {'cost':1}},)
                    nx.set_edge_attributes(self.roadG, {(pos,(i,j)): {'isRoad':True}},)

    def zone_road(self,pos):
        M = [(-1,0),(0,-1),(0,1),(1,0)]
        point = 0
        for m in M:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[1]) and (0<=j and j< self.obs.shape[2]):
                if not self.has_ent[cur]:
                    self.has_ent[cur] = True
                    point += 10
        return point

    def zone_home(self,pos):
        M55 = [(-2, -2), (-2, -1),(-2, 0),(-2, 1),(-2, 2),(-1, -2),(-1, 2),(0, -2),(0, 2),(1, -2),(1, 2),(2, -2),(2, -1),(2, 0),(2, 1),(2, 2)]
        for m in M55:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[1]) and (0<=j and j< self.obs.shape[2]):
                if self.obs[3,i,j] == 1:
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
            if (0<=i and i< self.obs.shape[1]) and (0<=j and j< self.obs.shape[2]):
                if self.obs[4,i,j] == 1:
                    point += 20
        
        for m in M33:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[1]) and (0<=j and j< self.obs.shape[2]):
                if self.obs[3,i,j] == 1:
                    return point-30
        for m in M55:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[1]) and (0<=j and j< self.obs.shape[2]):
                if self.obs[3,i,j] == 1:
                    return point-20
        for m in M77:
            cur = pos[0] + m[0], pos[1] + m[1]
            i,j = cur
            if (0<=i and i< self.obs.shape[1]) and (0<=j and j< self.obs.shape[2]):
                if self.obs[3,i,j] == 1:
                    return point-10
        
        return point

def display_img(im):
    # im0 = im[:,:,::-1]
    im0 = im
    im2 = Image.fromarray(im0)
    bio = BytesIO()
    im2.save(bio, format='png')
    ipyd.display(ipyd.Image(bio.getvalue(), format='png'))

class Car(object):
    def __init__(self,start, target):
        self.pos = tuple(start)
        self.target = tuple(target)
        self.path = []
        self.distance = 0
    def get_path(self,G):
        if self.target is None:
            raise Exception("No target to get_path")
        # self.path = search(G, self.pos, self.target)
        source = tuple(self.pos)
        target = tuple(self.target)
        try:
            self.path = nx.astar_path(G, source,target, heuristic=dist,weight= lambda u,v,d: d['cost'] if d['isRoad'] else None)
        except:
            self.path = None
        return self.path
    def move(self):
        self.distance+=1
        self.pos = self.path[1]
    def get_pos(self):
        return self.pos
class Home(object):
    def __init__(self,x,y):
        self.pos = (x,y)
        self.target = None
        self.finish = False
    def __repr__(self):
        return f"Home {self.pos}->{self.target}, finish = {self.finish}"
    def get_pos(self):
        return self.pos
    def set_target(self, t):
        if not isinstance(t,Target):
            ic(t)
            raise TypeError("The giving t is not target class")
        self.target = t
        
class Target(object):
    def __init__(self,x,y):
        self.pos = (x,y)
        self.leaf = []
    
    def __repr__(self):
        return f"Target {self.pos}"

def dist(source,target):
    source = tuple(source)
    target = tuple(target)
    if len(source)!=2 or len(target) != 2:
        raise ValueError("Dist: source/target should be 2-tuple")
    return abs(source[0]-target[0])+abs(source[1]-target[1])

def ask_direction(pos, ref):
    dif = [pos[0]-ref[0], pos[1]-ref[1]]
    if dif == [-1,0]:
        return 2 #up
    elif dif == [0,-1]:
        return  1 #left
    elif dif == [0,1]:
        return -2 #right
    elif dif == [1,0]:
        return -1 #down
    #opposite direction -> product<0
    else:
        raise ValueError('Error: direction is incorrect')
