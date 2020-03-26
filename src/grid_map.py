import numpy as np
import utils


class OccupancyGridMap:
    def __init__(self, map_config, grid_size=1.0):
        self.lo_occupied = map_config['lo_occupied']
        self.lo_free = map_config['lo_free']
        self.lo_max = map_config['lo_max']
        self.lo_min = map_config['lo_min']

        self.grid_map = {}
        self.grid_size = grid_size
        self.boundary = [9999,-9999,9999,-9999]


    @property
    def config(self):
        config_dict = {
            'lo_occupied': self.lo_occupied,
            'lo_free': self.lo_free,
            'lo_max': self.lo_max,
            'lo_min': self.lo_min
        }
        return config_dict


    def calc_grid_probability(self, coord):
        if coord in self.grid_map:
            return np.exp(self.grid_map[coord]) / (1.0 + np.exp(self.grid_map[coord]))
        else:
            return 0.5


    def calc_coordinate_probability(self, pose):
        x, y = int(round(pose[0]/self.grid_size)), int(round(pose[1]/self.grid_size))
        return self.calc_grid_probability((x,y))


    def calc_map_probability(self, x0, x1, y0, y1):
        map_prob = np.zeros((y1-y0, x1-x0))
        idx = 0
        for i in range(x0, x1):
            idy = 0
            for j in range(y0, y1):
                map_prob[idy, idx] = self.calc_grid_probability((i,j))
                idy += 1
            idx += 1
        return map_prob


    def scan_line(self, x0, x1, y0, y1):
        # Scale the position
        start = np.array([int(round(x0/self.grid_size)), int(round(y0/self.grid_size))])
        end = np.array([int(round(x1/self.grid_size)), int(round(y1/self.grid_size))])

        rec = utils.bresenham(start, end)
        for i in range(len(rec)):
            if i < len(rec)-2:
                change = self.lo_occupied
            else:
                change = self.lo_free

            if rec[i] in self.grid_map:
                self.grid_map[rec[i]] += change
            else:
                self.grid_map[rec[i]] = change
                if rec[i][0] < self.boundary[0]:
                    self.boundary[0] = rec[i][0]
                elif rec[i][0] > self.boundary[1]:
                    self.boundary[1] = rec[i][0]
                if rec[i][1] < self.boundary[2]:
                    self.boundary[2] = rec[i][1]
                elif rec[i][1] > self.boundary[3]:
                    self.boundary[3] = rec[i][1]

            if self.grid_map[rec[i]] > self.lo_max:
                self.grid_map[rec[i]] = self.lo_max
            if self.grid_map[rec[i]] < self.lo_min:
                self.grid_map[rec[i]] = self.lo_min