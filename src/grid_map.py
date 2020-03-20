import numpy as np
import utils


class OccupancyGridMap:
    def __init__(self, map_param, grid_size=1.0):
        self.map_param = map_param
        self.grid_map = {}
        self.grid_size = grid_size
        self.boundary = [9999,-9999,9999,-9999]


    def calc_grid_probability(self, pos):
        if pos in self.grid_map:
            return np.exp(self.grid_map[pos]) / (1.0 + np.exp(self.grid_map[pos]))
        else:
            return 0.5


    def calc_coordinate_probability(self, pos):
        x, y = int(round(pos[0]/self.grid_size)), int(round(pos[1]/self.grid_size))
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
        x0, x1 = int(round(x0/self.grid_size)), int(round(x1/self.grid_size))
        y0, y1 = int(round(y0/self.grid_size)), int(round(y1/self.grid_size))

        rec = utils.bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            if i < len(rec)-2:
                change = self.map_param[0]
            else:
                change = self.map_param[1]

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

            if self.grid_map[rec[i]] > self.map_param[2]:
                self.grid_map[rec[i]] = self.map_param[2]
            if self.grid_map[rec[i]] < self.map_param[3]:
                self.grid_map[rec[i]] = self.map_param[3]