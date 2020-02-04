"""
Implementation of distributed sorting algorithm from
The Dynamics of Collective Sorting Robot-Like Ants and Ant-Like Robots by Deneubourg et al.

Possible modifications to work on:
- add randomness: agent confuses two objects with some probability.
- incorporate pheromones and adaptive parameter modifications as in Vizine et al.
- experiment more with number of ants, really high iteration count, density of objects
- modify algorithm so it clusters similar objects together rather than performing a binary classification (i.e. Euclidean norm)
- Vizine et al. seems to use a torus rather than edge boundaries
- use a vision field rather than the memory approximation for f
- move based on pheromone (Ramos and Merelo)
- differentiate individual ants more
- different levels of sight
- more diverse numbers of items
- conditions about clustering outcomes
"""

import random
from collections import deque
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)


class Grid:
    def __init__(self, size, objects, num_agents, k_plus, k_minus, gamma):
        self.size = size
        self.ant_grid = -np.ones((size, size))
        self.object_grid = np.zeros((size, size))
        self.objects = objects
        self.num_agents = num_agents
        self.ant_empty_idx = -1
        self.object_empty_idx = 0
        self.ant_dict = {}
        self._populate(k_plus, k_minus, gamma)

    def _population_helper(self, grid, open_idx):
        open_x, open_y = np.where(grid == open_idx)
        location_index = random.randint(0, len(open_x) - 1)
        x_loc = open_x[location_index]
        y_loc = open_y[location_index]
        return x_loc, y_loc

    def _populate(self, k_plus, k_minus, gamma):
        # Agents
        for idx in range(self.num_agents):
            x_loc, y_loc = self._population_helper(self.ant_grid, self.ant_empty_idx)
            new_ant = Ant(x_loc, y_loc, idx, gamma, k_plus, k_minus)
            self.ant_dict[idx] = new_ant
            self.ant_grid[x_loc, y_loc] = idx
        # Objects
        for object_id, object_count in self.objects.items():
            for obj in range(object_count):
                x_loc, y_loc = self._population_helper(self.object_grid, self.object_empty_idx)
                self.object_grid[x_loc, y_loc] = object_id

    def _move(self, ant):
        x, y = ant.x, ant.y
        options = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        valid_options = []
        for opt in options:
            i, j = opt
            # torus boundary
            i_loc = i % self.size
            j_loc = j % self.size
            if self.ant_grid[i_loc, j_loc] == self.ant_empty_idx:
                valid_options.append((i_loc, j_loc))
        if len(valid_options) > 0:
            new_x, new_y = random.choice(valid_options)
            self.ant_grid[x, y] = self.ant_empty_idx
            self.ant_grid[new_x, new_y] = ant.idx
            ant.x = new_x
            ant.y = new_y

    def f_moore(self, obj_id, x, y):
        num_neighbors = 8
        x_options = [x - 1, x, x + 1]
        y_options = [y - 1, y, y + 1]
        obj_count = 0
        for i in x_options:
            for j in y_options:
                # use mod because of torus boundaries
                i = i % self.size
                j = j % self.size
                if i != x or j != y:
                    # if there is an object at this location
                    if self.object_grid[i, j] == obj_id:
                        obj_count += 1
        return obj_count / num_neighbors

    def f_von_neumann(self, obj_id, x, y):
        num_neighbors = 4
        options = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        obj_count = 0
        for opt in options:
            i, j = opt
            i = i % self.size
            j = j % self.size
            # if there is an object at this location
            if self.object_grid[i, j] == obj_id:
                obj_count += 1
        return obj_count / num_neighbors

    def f(self, num_neighbors, obj_id, x, y):
        if num_neighbors == 8:
            return self.f_moore(obj_id, x, y)
        if num_neighbors == 4:
            return self.f_von_neumann(obj_id, x, y)

    def run(self, iterations):
        for it in range(iterations):
            ant_idxs = list(self.ant_dict.keys())
            random.shuffle(ant_idxs)
            for idx in ant_idxs:
                ant = self.ant_dict[idx]
                x_loc = ant.x
                y_loc = ant.y
                obj = self.object_grid[x_loc, y_loc]
                current_object = ant.object
                if current_object is not None:
                    if obj == self.object_empty_idx:
                        f = self.f(ant.gamma, current_object, x_loc, y_loc)
                        if ant.should_put_down(f):
                            self.object_grid[x_loc, y_loc] = current_object
                            ant.object = None
                else:
                    if obj != self.object_empty_idx:
                        f = self.f(ant.gamma, obj, x_loc, y_loc)
                        if ant.should_pick_up(f):
                            self.object_grid[x_loc, y_loc] = self.object_empty_idx
                            ant.object = obj
                self._move(ant)
            if it % 5000 == 0: print(f'Iteration {it}')
        return self.object_grid


class Ant:
    def __init__(self, x, y, idx, gamma,  k_plus, k_minus):
        self.x = x
        self.y = y
        self.idx = idx
        self.gamma = gamma
        self.k_plus = k_plus
        self.k_minus = k_minus
        self.object = None

    def should_pick_up(self, f):
        prob = self.k_plus / (self.k_plus + f) ** 2
        return random.random() < prob

    def should_put_down(self, f):
        prob = f / (self.k_minus + f) ** 2
        return random.random() < prob


def moore_neighbors(object_grid, obj_id, loc):
    """
    Returns the indices of (up to) 8 neighbors (Moore neighborhood).
    """
    x, y = loc
    size = object_grid.shape[0]
    x_options = [x - 1, x, x + 1]
    y_options = [y - 1, y, y + 1]
    locations = []
    for i in x_options:
        for j in y_options:
            # use mod because of torus boundaries
            i = i % size
            j = j % size
            if i != x or j != y:
                # if there is an object at this location
                if object_grid[i, j] == obj_id:
                    locations.append((i, j))
    return locations


def dfs(object_grid, obj_id, start_x, start_y):
    visited = set()
    cluster_deque = deque([(start_x, start_y)])
    while len(cluster_deque) > 0:
        node = cluster_deque.pop()
        if node not in visited:
            visited.add(node)
            neighbors = moore_neighbors(object_grid, obj_id, node)
            if len(neighbors) > 0:
                for n in neighbors:
                    cluster_deque.append(n)
    return visited


def cluster_metric(object_grid, obj_id):
    size = len(object_grid)
    cluster_grid = np.zeros((size, size))
    cluster_sizes = []
    next_index = 1
    for x in range(size):
        for y in range(size):
            if object_grid[x, y] == obj_id and cluster_grid[x, y] == 0:
                cluster_grid[x, y] = next_index
                cluster_nodes = dfs(object_grid, obj_id, x, y)
                for node in cluster_nodes:
                    xx, yy = node
                    cluster_grid[xx, yy] = next_index
                cluster_sizes.append(len(cluster_nodes))
                next_index += 1
    return next_index - 1, np.mean(cluster_sizes), cluster_grid


def clustering(grid, indices):
    object_clusters = {}
    avg_sizes = []
    for idx in indices:
        num_clusters, avg_size, clustered_grid = cluster_metric(grid, idx)
        avg_sizes.append(avg_size)
        plt.matshow(clustered_grid)
        plt.axis('off')
        plt.title(f'Object {idx}')
        plt.show()
        object_clusters[idx] = (num_clusters, avg_size)
    return np.mean(avg_sizes)


def run_simulation(gridsize, objects_dict, num_agents, iterations, k_plus, k_minus, gamma):
    cmaplist = [(0.9, 0.9, 0.9, 1.0), (0.2, 0.2, 0.7, 1.0), (0.8, 0.2, 0.2, 1.0)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 3)

    grid = Grid(gridsize, objects_dict, num_agents, k_plus, k_minus, gamma)
    grid.run(iterations)

    object_ids = grid.objects.keys()
    avg_cluster_size = clustering(grid.object_grid, object_ids)
    plt.matshow(grid.object_grid, cmap=cmap)
    plt.title(f'{iterations} iterations, $F_c: ${round(float(avg_cluster_size), 2)}')
    plt.axis('off')
    plt.show()
    return grid.object_grid, avg_cluster_size


if __name__ == '__main__':
    final_grid, fitness = run_simulation(gridsize=15,
                                         objects_dict={1: 40, 2: 40},
                                         num_agents=20,
                                         iterations=int(100),
                                         k_plus=0.1,
                                         k_minus=0.1,
                                         gamma=4)
    print(fitness)
