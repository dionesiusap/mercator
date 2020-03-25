import numpy as np
import cv2
from math import *


def bresenham(start, end):
    x0 = int(round(start[0]))
    y0 = int(round(start[1]))
    x1 = int(round(end[0]))
    y1 = int(round(end[1]))

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


def load_env_from_img(fname):
        im = cv2.imread(fname)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m


def get_img_from_map(m):
    img = (255*m).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def rot2deg(R):
    cos = R[0,0]
    sin = R[1,0]
    theta = np.rad2deg(np.arccos(np.abs(cos)))
    
    if cos>0 and sin>0:
        return theta
    elif cos<0 and sin>0:
        return 180-theta
    elif cos<0 and sin<0:
        return 180+theta
    elif cos>0 and sin<0:
        return 360-theta
    elif cos==0 and sin>0:
        return 90.0
    elif cos==0 and sin<0:
        return 270.0
    elif cos>0 and sin==0:
        return 0.0
    elif cos<0 and sin==0:
        return 180.0