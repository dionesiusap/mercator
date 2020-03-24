import numpy as np
import random
import cv2
import math
import utils


class Robot:
    def __init__(self, pos, config, env_img):
        self.pos = pos
        self.config = config
        
        # self.img_map = utils.load_env_from_img(env_img)
        scale = 1
        img = utils.load_env_from_img(env_img)
        img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
        self.img_map = img
    

    def move(self, action_id):
        vec = [np.sin(np.deg2rad(self.pos[2])), np.cos(np.deg2rad(self.pos[2]))]
        vel = self.config[4]
        ang = self.config[5]

        if action_id == 1:
            self.pos[0] -= vel*vec[0]
            self.pos[1] += vel*vec[1]
        if action_id == 2:
            self.pos[0] += vel*vec[0]
            self.pos[1] -= vel*vec[1]
        if action_id == 3:
            self.pos[2] -= ang
            self.pos[2] = self.pos[2] % 360
        if action_id == 4:  
            self.pos[2] += ang
            self.pos[2] = self.pos[2] % 360
        
        if action_id == 5:
            self.pos[1] -= vel
        if action_id == 6:
            self.pos[0] -= vel
        if action_id == 7:
            self.pos[0] += vel
        if action_id == 8:
            self.pos[1] += vel
        
        sig=[0.5,0.5,0.5]
        self.pos[0] += random.gauss(0,sig[0])
        self.pos[1] += random.gauss(0,sig[1])
        self.pos[2] += random.gauss(0,sig[2])


    def measure(self):
        sense_data = []
        inter = (self.config[2] - self.config[1]) / (self.config[0]-1)
        for i in range(self.config[0]):
            theta = self.pos[2] + self.config[1] + i*inter
            sense_data.append(self.laser_scan(np.array((self.pos[0], self.pos[1])), theta))
        return sense_data


    def laser_scan(self, pos, theta):
        end = np.array((pos[0] + self.config[3]*np.cos(np.deg2rad(theta)), pos[1] + self.config[3]*np.sin(np.deg2rad(theta))))

        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = utils.bresenham(x0, x1, y0, y1)
        i = 0
        dist = self.config[3]
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1]<0 or p[0]<0:
                continue
            if self.img_map[p[1], p[0]] < 0.6:
                tmp = math.pow(float(p[0]) - pos[0], 2) + math.pow(float(p[1]) - pos[1], 2)
                tmp = math.sqrt(tmp)
                if tmp < dist:
                    dist = tmp
        return dist