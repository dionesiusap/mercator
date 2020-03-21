import numpy as np
from math import *


def bresenham(x0, x1, y0, y1):
    rec = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return rec


def gaussian(x, mu, sigma):
    return 1./(sqrt(2.*pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2.)/2)


def end_point(pos, robot_param, sensor_data):
    pts_list = []
    inter = (robot_param[2] - robot_param[1]) / (robot_param[0]-1)
    for i in range(robot_param[0]):
        theta = pos[2] + robot_param[1] + i*inter
        pts_list.append(
            [ pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta)),
              pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta))] )
    return pts_list