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

    # Initialize 2D Environment
    robot_config = {
        'velocity': 6.0,
        'omega': 6.0
    }

    sensor_config = {
        'sensor_size': 240,
        'start_angle': -30.0,
        'end_angle': 210.0,
        'max_dist': 150.0
    }
    robot_pos = np.array([150.0, 100.0, 0.0])
    robot = Robot(robot_pos, robot_config, sensor_config, '../img/env1.png')

    # Initialize GridMap
    occupancy_map_config = {
        'lo_occupied': 0.4,
        'lo_free': -0.4,
        'lo_max': 5.0,
        'lo_min': -5.0
    }
    m = OccupancyGridMap(occupancy_map_config, grid_size=1.0)
    sensor_data = robot.measure()
    SensorMapping(m, robot.pose, robot.sensor.config, sensor_data)

    img = Draw(robot.environment, 1, robot.pose, sensor_data, robot.sensor.config)
    mimg = AdaptiveGetMap(m)
    cv2.imshow('view',img)
    cv2.imshow('map',mimg)

    # Initialize Particle
    pf = ParticleFilter(robot_pos.copy(), robot_config, sensor_config, copy.deepcopy(m), 10)
    
    # Scan Matching Test
    matching_m = OccupancyGridMap(occupancy_map_config, grid_size=1.0)
    SensorMapping(matching_m, robot.pose, robot.sensor.config, sensor_data)
    matching_pos = np.array([150.0, 100.0, 0.0])

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
            sensor_data = robot.measure()
            SensorMapping(m, robot.pose, robot.sensor.config, sensor_data)

            img = Draw(robot.environment, 1, robot.pose, sensor_data, robot.sensor.config)
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