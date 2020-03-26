#!/usr/bin/env python3

import numpy as np
import random
import cv2
import math
import copy
import utils
from grid_map import OccupancyGridMap
from particle_filter import Particle, ParticleFilter
from robot import Robot


def Draw(img_map, scale, robot_pos, sensor_data, sensor_config):
    img = img_map.copy()
    img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
    img = utils.get_img_from_map(img)
    sensor_end_points = utils.end_point(robot_pos, sensor_config, sensor_data)
    for pts in sensor_end_points:
        cv2.line(
            img, 
            (int(scale*robot_pos[0]), int(scale*robot_pos[1])), 
            (int(scale*pts[0]), int(scale*pts[1])),
            (255,0,0), 1
        )

    cv2.circle(img,(int(scale*robot_pos[0]), int(scale*robot_pos[1])), int(3*scale), (0,0,255), -1)
    return img


def DrawParticle(img, particles, scale=1.0):
    for p in particles:
        cv2.circle(img,(int(scale*p.pose[0]), int(scale*p.pose[1])), int(2), (0,200,0), -1)
    return img


def SensorMapping(m, robot_pos, sensor_config, sensor_data):
    inter = (sensor_config['end_angle'] - sensor_config['start_angle']) / (sensor_config['sensor_size']-1)
    for i in range(sensor_config['sensor_size']):
        if sensor_data[i] > sensor_config['max_dist']-1 or sensor_data[i] < 1:
            continue
        theta = robot_pos[2] + sensor_config['start_angle'] + i*inter
        m.scan_line(
            int(robot_pos[0]), 
            int(robot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
            int(robot_pos[1]),
            int(robot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta)))
        )


def AdaptiveGetMap(gmap):
    mimg = gmap.calc_map_probability(
        gmap.boundary[0]-20, gmap.boundary[1]+20, 
        gmap.boundary[2]-20, gmap.boundary[3]+20 )
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    return mimg


if __name__ == '__main__':
    # Initialize OpenCV Windows
    cv2.namedWindow('view', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)

    # Load configs
    env_config = utils.load_config('../config/environment.yaml')
    agent_config = utils.load_config('../config/agent.yaml')
    grid_map_config = utils.load_config('../config/grid_map.yaml')

    # Initialize 2D Environment
    env = utils.load_env_from_img(env_config['img_src'])
    env = cv2.resize(env, (round(env_config['scale']*env.shape[1]), round(env_config['scale']*env.shape[0])), interpolation=cv2.INTER_LINEAR)

    # Initialize Agent
    robot_pos = np.array([150.0, 100.0, 0.0])
    robot = Robot(robot_pos, agent_config['robot'], agent_config['sensor'])

    # Initialize Grid Map
    m = OccupancyGridMap(grid_map_config, grid_size=1.0)
    sensor_data = robot.measure(env)
    SensorMapping(m, robot.pose, robot.sensor.config, sensor_data)

    # Initialize Particle
    pf = ParticleFilter(robot_pos.copy(), agent_config['robot'], agent_config['sensor'], copy.deepcopy(m), 10)

    img = Draw(env, 1, robot.pose, sensor_data, robot.sensor.config)
    mimg = AdaptiveGetMap(m)
    cv2.imshow('view',img)
    cv2.imshow('map',mimg)

    # Main Loop
    while(1):
        # Input Control
        action = -1
        k = cv2.waitKey(1)
        if k==ord('w'):
            action = 1
        if k==ord('s'):
            action = 2
        if k==ord('a'):
            action = 3
        if k==ord('d'): 
            action = 4 
        
        if k==ord('i'):
            action = 5
        if k==ord('j'):
            action = 6
        if k==ord('l'):
            action = 7
        if k==ord('k'):
            action = 8
        
        if action > 0:
            robot.move(action)
            sensor_data = robot.measure(env)
            SensorMapping(m, robot.pose, robot.sensor.config, sensor_data)

            img = Draw(env, 1, robot.pose, sensor_data, robot.sensor.config)
            mimg = AdaptiveGetMap(m)
            
            pf.update(action, sensor_data)
            mid = np.argmax(pf.weights)
            imgp0 = AdaptiveGetMap(pf.particles[mid].grid_map)
            
            img = DrawParticle(img, pf.particles)
            cv2.imshow('view',img)
            cv2.imshow('map',mimg)

            cv2.imshow('particle_map',imgp0)
            pf.resampling(sensor_data)

    cv2.destroyAllWindows()