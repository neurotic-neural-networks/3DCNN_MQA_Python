
#TODO Visualization per layer00000000000
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from itertools import product, combinations
import itertools

class Plotter():
    def __init__(self):
        self.fig = plt.figure()
        self.pos = 1
        self.pos2 = 2

    def addplot(self, g):
        self.ax = self.fig.add_subplot(3, 5, self.pos, projection='3d')
        self.pos += 1
        self.draw_boundary()
        plot_protein(g, self.ax)

    def show(self):
        plt.show()


    # draw boundary
    def draw_boundary(self, min=0, max=20):
        r = [min, max]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                self.ax.plot3D(*zip(s, e), color='black', alpha=0.3)

def get_regularization(matrix, min=0, max=1):
    return (max-min)/(matrix.max()-matrix.min())*(matrix-matrix.max())+max

def plot_protein(a, ax):
    a = get_regularization(a)
    for x, y, z in itertools.product(range(a.shape[0]), range(a.shape[1]), range(a.shape[2])):
        if a[x,y,z]>0.01:
            ax.scatter(x, y, z, c='r', marker='o', alpha=a[x,y,z])

def gaussian_blur(a, sigma=1.0):
    x = np.arange(-3, 4, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3, 4, 1)
    z = np.arange(-3, 4, 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    kernel = np.exp(-(xx * 2 + yy * 2 + zz * 2) / (2 * sigma * 2))
    return scipy.signal.convolve(a, kernel, mode="same")


