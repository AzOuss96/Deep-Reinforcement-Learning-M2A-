import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
#from PIL import Image as Image
import matplotlib.pyplot as plt
from gym.envs.toy_text import discrete
from itertools import groupby
from operator import itemgetter
# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red

COLORS = {0:[0,0,0], 1:[128,128,128], \
          2:[0,0,255], 3:[0,255,0], \
          5:[255,0,0], 6:[255,0,255], \
          4:[255,255,0]}

class GridworldEnv(discrete.DiscreteEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'], #, 'state_pixels'],
        'video.frames_per_second': 1
    }
    num_env = 0
    plan='gridworldPlans/plan0.txt'

    def __init__(self):
        self._make(GridworldEnv.plan,rewards={0:0,3:1,4:1,5:-1,6:-1})

    def setPlan(self,plan,rewards):
        self._make(plan,rewards)

    def _make(self,plan,rewards):
        self.rewards=rewards
        self.nA = 4
        self.actions={0:[1,0],1:[-1,0],2:[0,-1],3:[0,1]}
        self.nbMaxSteps=1000
        # 0:South
        # 1:North
        # 2:West
        # 3:East
        self.action_space = spaces.Discrete(self.nA)

        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, plan)
        self.obs_shape = [128, 128, 3]
        self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = np.copy(self.start_grid_map)  # current grid map
        self.nbSteps=0
        self.rstates = {}
        self.P=None
        self.nS=0
        self.startPos=self._get_agent_pos(self.current_grid_map)
        self.currentPos=copy.deepcopy(self.startPos)
        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env
        self.verbose=False
        #self.render()
        #plt.pause(1)


    def getMDP(self):
        if self.P is None:
            self.P={}
            self.states={self.start_grid_map.dumps():0}
            self._getMDP(self.start_grid_map, self.startPos)
        return (self.states,self.P)



    def _getMDP(self,gridmap,state):
        cur = gridmap.dumps()
        succs={0:[],1:[],2:[],3:[]}
        self.P[cur]=succs
        self._exploreDir(gridmap,state,[1,0],0,2,3)
        self._exploreDir(gridmap, state, [-1, 0], 1, 2, 3)
        self._exploreDir(gridmap, state, [0, 1], 3, 0, 1)
        self._exploreDir(gridmap, state, [0, -1], 2, 0, 1)


    def _exploreDir(self,gridmap,state,dir,a,b,c):
        cur=gridmap.dumps()
        gridmap = copy.deepcopy(gridmap)
        succs=self.P[cur]
        nstate = copy.deepcopy(state)
        nstate[0]+=dir[0]
        nstate[1] += dir[1]

        if nstate[0]<gridmap.shape[0] and nstate[0]>=0 and nstate[1]<gridmap.shape[1] and nstate[1]>=0 and gridmap[nstate[0],nstate[1]]!=1:
                oldc=gridmap[nstate[0],nstate[1]]
                gridmap[state[0],state[1]] = 0
                gridmap[nstate[0],nstate[1]] = 2
                ng=gridmap.dumps()
                done = (oldc == 3 or oldc == 5)
                if ng in self.states:
                    ns=self.states[ng]
                else:
                    ns=len(self.states)
                    self.states[ng]=ns
                    if not done:
                        self._getMDP(gridmap,nstate)
                r=self.rewards[oldc]

                succs[a].append((0.8, ng,r,done))
                succs[b].append((0.1, ng, r, done))
                succs[c].append((0.1, ng, r, done))
        else:
            succs[a].append((0.8,cur,self.rewards[0],False))
            succs[b].append((0.1, cur, self.rewards[0], False))
            succs[c].append((0.1, cur, self.rewards[0], False))




    def _get_agent_pos(self, grid_map):
        state = list(map(
                 lambda x:x[0] if len(x) > 0 else None,
                 np.where(grid_map == 2)
             ))
        return state


    def step(self, action):
        self.nbSteps += 1
        action = int(action)
        p = np.random.rand()
        if p<0.2:
            p = np.random.rand()
            if action==0 or action==1:
                if p < 0.5:
                    action=2
                else:
                    action=3
            else:
                if p < 0.5:
                    action=0
                else:
                    action=1
        npos = (self.currentPos[0] + self.actions[action][0], self.currentPos[1] + self.actions[action][1])
        rr=-1*(self.nbSteps>self.nbMaxSteps)
        if npos[0] >= self.current_grid_map.shape[0] or npos[0] < 0 or npos[1] >= self.current_grid_map.shape[1] or npos[1] < 0 or self.current_grid_map[npos[0],npos[1]]==1:
            return (self.current_grid_map, self.rewards[0]+rr, self.nbSteps>self.nbMaxSteps, {})
        c=self.current_grid_map[npos]
        r = self.rewards[c]+rr
        done=(c == 3 or c == 5 or self.nbSteps>self.nbMaxSteps)
        self.current_grid_map[self.currentPos[0],self.currentPos[1]] = 0
        self.current_grid_map[npos[0],npos[1]] = 2
        self.currentPos = npos
        return (self.current_grid_map,r,done,{})

    def reset(self):
        self.currentPos = copy.deepcopy(self.startPos)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.nbSteps=0
        return self.current_grid_map

    def _read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
            print(str(grid_map))
        grid_map_array = np.array(
            list(map(
                lambda x: list(map(
                    lambda y: int(y),
                    x.split(' ')
                )),
                grid_map
            ))
        )
        return grid_map_array



    def _gridmap_to_img(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.uint8)
        gs0 = int(observation.shape[0] / grid_map.shape[0])
        gs1 = int(observation.shape[1] / grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1] = np.array(COLORS[grid_map[i, j]])
        return observation

    def render(self, pause=0.00001, mode='human', close=False):
        if not self.verbose :
            return
        img = self._gridmap_to_img(self.current_grid_map)
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(pause)
        return img

    def _close_env(self):
        plt.close(self.this_fig_num)
        return

    def close(self):
        super(GridworldEnv,self).close()
        self._close_env()
    def changeState(self,gridmap):
        self.current_grid_map=gridmap
        self.currentPos=self._get_agent_pos(gridmap)
        self.render()
