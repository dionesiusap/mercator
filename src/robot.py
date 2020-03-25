import numpy as np
import random
import cv2
import math
import utils


class Robot:
    def __init__(self, pose, robot_config, sensor_config, env_img):
        self.pose = pose
        self.velocity = robot_config['velocity']
        self.omega = robot_config['omega']
        
        scale = 1
        img = utils.load_env_from_img(env_img)
        img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
        self.environment = img
        self.sensor = LaserScanSensor(sensor_config, self.environment)


    @property
    def config(self):
        config_dict = {
            'velocity': self.velocity,
            'omega': self.omega
        }
        return config_dict


    def move(self, action_id):
        vec = [np.sin(np.deg2rad(self.pose[2])), np.cos(np.deg2rad(self.pose[2]))]

        if action_id == 1:
            self.pose[0] -= self.velocity*vec[0]
            self.pose[1] += self.velocity*vec[1]
        if action_id == 2:
            self.pose[0] += self.velocity*vec[0]
            self.pose[1] -= self.velocity*vec[1]
        if action_id == 3:
            self.pose[2] -= self.omega
            self.pose[2] = self.pose[2] % 360
        if action_id == 4:  
            self.pose[2] += self.omega
            self.pose[2] = self.pose[2] % 360
        
        if action_id == 5:
            self.pose[1] -= self.velocity
        if action_id == 6:
            self.pose[0] -= self.velocity
        if action_id == 7:
            self.pose[0] += self.velocity
        if action_id == 8:
            self.pose[1] += self.velocity
        
        sig=[0.5,0.5,0.5]
        self.pose[0] += random.gauss(0,sig[0])
        self.pose[1] += random.gauss(0,sig[1])
        self.pose[2] += random.gauss(0,sig[2])


    def measure(self):
        sense_data = self.sensor.do_complete_scan(self.pose)
        return sense_data


class LaserScanSensor:
    def __init__(self, sensor_config, environment):
        self.sensor_size = sensor_config['sensor_size']
        self.start_angle = sensor_config['start_angle']
        self.end_angle = sensor_config['end_angle']
        self.max_dist = sensor_config['max_dist']

        self.environment = environment


    @property
    def config(self):
        config_dict = {
            'sensor_size': self.sensor_size,
            'start_angle': self.start_angle,
            'end_angle': self.end_angle,
            'max_dist': self.max_dist
        }
        return config_dict


    def measure_single_beam(self, robot_pose, theta):
        start = np.array([robot_pose[0], robot_pose[1]])
        end = np.array([(robot_pose[0] + self.max_dist * np.cos(np.deg2rad(theta))), (robot_pose[1] + self.max_dist * np.sin(np.deg2rad(theta)))])

        beam_path = utils.bresenham(start, end)

        dist = self.max_dist
        for p in beam_path:
            if p[1] < self.environment.shape[0] and p[0] < self.environment.shape[1] and p[1] >= 0 and p[0] >= 0:
                if self.environment[p[1], p[0]] < 0.6:
                    tmp = math.pow(float(p[0]) - robot_pose[0], 2) + math.pow(float(p[1]) - robot_pose[1], 2)
                    tmp = math.sqrt(tmp)
                    if tmp < dist:
                        dist = tmp
        return dist
    
    
    def do_complete_scan(self, robot_pose):
        sense_data = []
        inter = (self.end_angle - self.start_angle) / (self.sensor_size-1)
        for i in range(self.sensor_size):
            theta = robot_pose[2] + self.start_angle + i*inter
            sense_data.append(self.measure_single_beam(np.array((robot_pose[0], robot_pose[1])), theta))
        return sense_data