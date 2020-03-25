import numpy as np
import random
import math
import copy
import threading
import utils
from grid_map import OccupancyGridMap


class Particle:
    def __init__(self, pos, robot_config, sensor_config, grid_map):
        self.pos = pos
        self.robot_config = robot_config
        self.sensor_config = sensor_config
        self.grid_map = grid_map


    def control_update(self, action_id, sigma=[0.4,0.4,0.4]):
        vec = [np.sin(np.deg2rad(self.pos[2])), np.cos(np.deg2rad(self.pos[2]))]
        vel = self.robot_config['velocity']
        ang = self.robot_config['omega']

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

        self.pos[0] += random.gauss(0, sigma[0])
        self.pos[1] += random.gauss(0, sigma[1])
        self.pos[2] += random.gauss(0, sigma[2])


    def nearest_distance(self, x, y, wsize, th):
        min_dist = 9999
        min_x = None
        min_y = None
        grid_size = self.grid_map.grid_size
        xx = int(round(x/grid_size))
        yy = int(round(y/grid_size))
        for i in range(xx-wsize, xx+wsize):
            for j in range(yy-wsize, yy+wsize):
                if self.grid_map.calc_grid_probability((i,j)) < th:
                    dist = (i-xx)*(i-xx) + (j-yy)*(j-yy)
                    if dist < min_dist:
                        min_dist = dist
                        min_x = i
                        min_y = j

        return math.sqrt(float(min_dist)*grid_size)


    def measurement_update(self, sensor_data):
        p_hit = 0.9
        p_rand = 0.1
        sig_hit = 3.0
        q = 1
        plist = utils.end_point(self.pos, self.sensor_config, sensor_data)
        for i in range(len(plist)):
            if not (sensor_data[i] > self.sensor_config['max_dist']-1 or sensor_data[i] < 1):
                dist = self.nearest_distance(plist[i][0], plist[i][1], 4, 0.2)
                q = q * (p_hit*utils.gaussian(0,dist,sig_hit) + p_rand/self.sensor_config['max_dist'])
        return q


    def mapping(self, sensor_data):
        inter = (self.sensor_config['end_angle'] - self.sensor_config['start_angle']) / (self.sensor_config['sensor_size']-1)
        for i in range(self.sensor_config['sensor_size']):
            if not (sensor_data[i] > self.sensor_config['max_dist']-1 or sensor_data[i] < 1):
                theta = self.pos[2] + self.sensor_config['start_angle'] + i*inter
                self.grid_map.scan_line(
                    int(self.pos[0]), 
                    int(self.pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
                    int(self.pos[1]),
                    int(self.pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta)))
                )


class ParticleFilter:
    def __init__(self, pos, robot_config, sensor_config, grid_map, size):
        self.size = size
        self.particles = []
        self.weights = np.ones((size), dtype=float) / size
        p = Particle(pos.copy(), robot_config, sensor_config, copy.deepcopy(grid_map))
        for i in range(size):
            self.particles.append(copy.deepcopy(p))
    

    def resampling(self, sensor_data):
        map_rec = np.zeros((self.size))
        re_id = np.random.choice(self.size, self.size, p=list(self.weights))
        new_particles = []
        for i in range(self.size):
            if map_rec[re_id[i]] == 0:
                self.particles[re_id[i]].mapping(sensor_data)
                map_rec[re_id[i]] = 1
            new_particles.append(copy.deepcopy(self.particles[re_id[i]]))
        self.particles = new_particles
        self.weights = np.ones((self.size), dtype=float) / float(self.size)


    def update(self, control, sensor_data):
        field = np.zeros((self.size), dtype=float)
        for i in range(self.size):
            self.particles[i].control_update(control)
            field[i] = self.particles[i].measurement_update(sensor_data)
        print(field)
        if np.sum(field) != 0:
            self.weights = field / np.sum(field)