"""
Quadtree implementation
Nolan Potter and Ethan Chen
"""

import random
from queue import Queue
import matplotlib.pyplot as plt
import numpy as np

class point:

    def __init__(self, x, y, data=[]):
        #point data
        self.x = x
        self.y = y
        self.data = data

    def __repr__(self):
        return f'{{"x": {self.x}, "y": {self.y}}}'

class square:

    def __init__(self, x, y, l, points=[]):
        # square cell initializing
        self.x = x
        self.y = y
        self.l = l
        self.points = points

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.l})'

    def contains(self, point):
        # checks if point falls within a cell
        xcheck = self.x - (self.l / 2) <= point.x and self.x + (self.l / 2) > point.x
        ycheck = self.y - (self.l / 2) <= point.y and self.y + (self.l / 2) > point.y
        return xcheck and ycheck

class quadtree:

    def __init__(self, square, capacity, divided=False):
        #initialize quadtree object
        self.square = square
        self.capacity = capacity
        self.divided = divided
        self.topleft = None
        self.topright = None
        self.botleft = None
        self.botright = None

    def subdivide(self):
        #divide up the current cell
        x, y, l = self.square.x, self.square.y, self.square.l

        topleft = square(x-l/4, y+l/4, l/2, [])
        self.topleft = quadtree(topleft, 1)

        topright = square(x+l/4, y+l/4, l/2, [])
        self.topright = quadtree(topright, 1)

        botleft = square(x-l/4, y-l/4, l/2, [])
        self.botleft = quadtree(botleft, 1)

        botright = square(x+l/4, y-l/4, l/2, [])
        self.botright = quadtree(botright, 1)

        self.divided = True

        for point in self.square.points:
            self.topleft.insert(point)
            self.topright.insert(point)
            self.botleft.insert(point)
            self.botright.insert(point)

        self.points = []

    def insert(self, point):
        #insert a point into the quadtree
        if not self.square.contains(point):
            return
        elif self.divided:
            self.topleft.insert(point)
            self.topright.insert(point)
            self.botleft.insert(point)
            self.botright.insert(point)
        elif len(self.square.points) < self.capacity:
            self.square.points.append(point)
        else:
            self.subdivide()
            self.topleft.insert(point)
            self.topright.insert(point)
            self.botleft.insert(point)
            self.botright.insert(point)

    def printsub(self):
        if self.divided is False and len(self.square.points) > 0:
            print(self.square)
            print(self.square.points)
        else:
            if self.topleft is not None:
                self.topleft.printsub()
            if self.topright is not None:
                self.topright.printsub()
            if self.botleft is not None:
                self.botleft.printsub()
            if self.botright is not None:
                self.botright.printsub()


if __name__ == "__main__":
    sq1 = square(0, 0, 10)
    qtree1 = quadtree(sq1, 1)
    # for x in range(0, 10):
    #     pt = point(random.randint(-5, 5), random.randint(-5,5))
    #     print(pt)
    #     qtree1.insert(pt)