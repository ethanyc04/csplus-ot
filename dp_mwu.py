import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import copy
import glob

class point:

    def __init__(self, x, y, data):
        #point data
        self.x = x
        self.y = y
        self.data = data

    def __repr__(self):
        return f'{{"x": {self.x}, "y": {self.y}}}'

class quadtree:     #quadtree data structure

    def __init__(self, x, y, l):
        #initialize quadtree object
        self.x = x
        self.y = y
        self.l = l
        self.points = []
        self.divided = False
        self.topleft = None
        self.topright = None
        self.botleft = None
        self.botright = None
        self.parent = None
        self.dualweight = []
        self.flow = []
        self.mass = 0
        self.min_cost_child = None
        self.cost_to_parent = 0
        self.augment_cost = 0
        self.augment_path_cost = 0
        self.augment_mass = 0
        self.id = None

    def __repr__(self):
        return f'{{"x": {self.x}, "y": {self.y}, "id": {self.id}}}'

    def contains(self, point):
        # checks if point falls within a cell
        xcheck = self.x - (self.l / 2) <= point.x and self.x + (self.l / 2) >= point.x
        ycheck = self.y - (self.l / 2) <= point.y and self.y + (self.l / 2) >= point.y
        return xcheck and ycheck

    def subdivide(self):
        #divide up the current cell
        x, y, l = self.x, self.y, self.l

        self.topleft = quadtree(x-l/4, y+l/4, l/2)
        self.topleft.parent = self

        self.topright = quadtree(x+l/4, y+l/4, l/2)
        self.topright.parent = self

        self.botleft = quadtree(x-l/4, y-l/4, l/2)
        self.botleft.parent = self

        self.botright = quadtree(x+l/4, y-l/4, l/2)
        self.botright.parent = self

        self.divided = True

        for point in self.points:
            leaf = self.topleft.insert(point)
            if leaf == None:
                leaf = self.topright.insert(point)
            if leaf == None:
                leaf = self.botleft.insert(point)
            if leaf == None:
                leaf = self.botright.insert(point)

        self.points = []

    def insert(self, point):
        #insert a point into the quadtree starting at root
        if not self.contains(point):
            return None
        elif self.divided:
            leaf = self.topleft.insert(point)
            if leaf == None:
                leaf = self.topright.insert(point)
            if leaf == None:
                leaf = self.botleft.insert(point)
            if leaf == None:
                leaf = self.botright.insert(point)
            return leaf
        elif len(self.points) == 0:
            self.points.append(point)
            return self
        else:
            self.subdivide()
            leaf = self.topleft.insert(point)
            if leaf == None:
                leaf = self.topright.insert(point)
            if leaf == None:
                leaf = self.botleft.insert(point)
            if leaf == None:
                leaf = self.botright.insert(point)
            return leaf

    def backward_insert(self, point):
        #insert a point into the quadtree bottom-up recursively starting at leaf

        leaf = self.insert(point)
        if leaf == None:
            return self.parent.backward_insert(point)
        
        return leaf

    def insert_list(qtree, points):
        #insert a list of points into the quadtree
        q = qtree
        for p in points:
            q = q.backward_insert(p)

    def killemptychildren(self):
        #get rid of any cells that do not have points inisde
        if not self.divided and len(self.points) != 0:
            return

        if not self.topleft.divided and len(self.topleft.points) == 0:
            self.topleft = None
        else:
            self.topleft.killemptychildren()
        
        if not self.topright.divided and len(self.topright.points) == 0:
            self.topright = None
        else:
            self.topright.killemptychildren()

        if not self.botleft.divided and len(self.botleft.points) == 0:
            self.botleft = None
        else:
            self.botleft.killemptychildren()

        if not self.botright.divided and len(self.botright.points) == 0:
            self.botright = None
        else:
            self.botright.killemptychildren()

        

    def printsub(self):
        # print subtree
        print(self)
        if len(self.points) != 0:
            print(self.points[0].data)
        if self.divided is False and len(self.points) > 0:
            # print((self.x, self.y, self.l))
            # print(self.points)
            pass
        else:
            if self.topleft is not None:
                self.topleft.printsub()
            if self.topright is not None:
                self.topright.printsub()
            if self.botleft is not None:
                self.botleft.printsub()
            if self.botright is not None:
                self.botright.printsub()

    def getlistofpoints(self, lst):
        #gets list of points from a tree
        #input list in form of [[xcoords], [ycoords], [distribution]
        #colors are hard coded to work with 2 distributinos
        if self.divided is False and len(self.points) > 0:
            lst[0].append(self.points[0].x)
            lst[1].append(self.points[0].y)
            if min(self.points[0].data) > 0:
                lst[2].append(2)
            elif self.points[0].data[0] > 0:
                lst[2].append(0)
            elif self.points[0].data[1] > 0:
                lst[2].append(1)
            else:
                lst[2].append(1000)

            return lst
        if self.topleft is not None:
            lst = self.topleft.getlistofpoints(lst)
        if self.topright is not None:
            lst = self.topright.getlistofpoints(lst)
        if self.botleft is not None:
            lst = self.botleft.getlistofpoints(lst)
        if self.botright is not None:
            lst = self.botright.getlistofpoints(lst)
        
        return lst

    def getcellboundaries(self, lst):
        #returns info wiht correct format to print the line segments of the cells
        #format is list of lists [[[x1, x2],[x3,x4]],[[y1, y2],[y3,y4]]]
        
        
        line1x = [self.x - self.l / 2, self.x - self.l / 2]
        line1y = [self.y - self.l / 2, self.y + self.l / 2]
        lst[0].append(line1x)
        lst[1].append(line1y)

        line2x = [self.x - self.l / 2, self.x + self.l / 2]
        line2y = [self.y + self.l / 2, self.y + self.l / 2]
        lst[0].append(line2x)
        lst[1].append(line2y)

        line3x = [self.x + self.l / 2, self.x + self.l / 2]
        line3y = [self.y + self.l / 2, self.y - self.l / 2]
        lst[0].append(line3x)
        lst[1].append(line3y)

        line4x = [self.x - self.l / 2, self.x + self.l / 2]
        line4y = [self.y - self.l / 2, self.y - self.l / 2]
        lst[0].append(line4x)
        lst[1].append(line4y)

            
        
        if self.topleft is not None:
            lst = self.topleft.getcellboundaries(lst)
        if self.topright is not None:
            lst = self.topright.getcellboundaries(lst)
        if self.botleft is not None:
            lst = self.botleft.getcellboundaries(lst)
        if self.botright is not None:
            lst = self.botright.getcellboundaries(lst)

        return lst

    def plottree(self):
        #plots a quadtree, colors are hard coded for 2 distributions
        lstofpts = self.getlistofpoints([[],[],[]])

        qtreeboundaries = self.getcellboundaries([[],[]])

        for i in range(len(qtreeboundaries[0])):
            print("test")
            plt.plot(qtreeboundaries[0][i], qtreeboundaries[1][i], color="black")

        upperx = (self.x + self.l / 2) + .2 * abs(self.x + self.l / 2)
        uppery = (self.y + self.l / 2) + .2 * abs(self.y + self.l / 2)

        lowerx = (self.x - self.l / 2) - .2 * abs(self.x - self.l / 2)
        lowery = (self.y - self.l / 2) - .2 * abs(self.y - self.l / 2)
        dist1 = [[],[]]
        dist2 = [[],[]]
        bothdist = [[],[]]
        print(lstofpts)
        for i in range(len(lstofpts[0])):
            if lstofpts[2][i] == 0:
                dist1[0].append(lstofpts[0][i])
                dist1[1].append(lstofpts[1][i])
            elif lstofpts[2][i] == 1:
                dist2[0].append(lstofpts[0][i])
                dist2[1].append(lstofpts[1][i])
            else:
                bothdist[0].append(lstofpts[0][i])
                bothdist[1].append(lstofpts[1][i])
        plt.plot(dist1[0], dist1[1], 'ro', color="red")
        plt.plot(dist2[0], dist2[1], 'ro', color="blue")
        plt.plot(bothdist[0], bothdist[1], 'ro', color="purple")
        plt.axis((lowerx, upperx, lowery, uppery))
        plt.show()

    def reset(self):
        #reset quadtree node
        self.dualweight = []
        self.flow = []
        self.mass = 0
        self.min_cost_child = None
        self.cost_to_parent = 0
        self.augment_cost = 0
        self.augment_path_cost = 0
        self.augment_mass = 0

def getboundingbox(lstofpts):
   #gets non randomly shifted minimum bounding box
   minx = float('inf')
   maxx = -float('inf')
   miny = float('inf')
   maxy = -float('inf')

   for pt in lstofpts:
       if pt.x < minx:
           minx = pt.x
       if pt.x > maxx:
           maxx = pt.x
       if pt.y < miny:
           miny = pt.y
       if pt.y > maxy:
           maxy = pt.y

   centerx = (minx + maxx)/2
   centery = (miny + maxy)/2
   length = max(maxx-minx, maxy-miny)

   return centerx, centery, length

def insert_list(qtree, points):
   #insert a list of points into the quadtree
   q = qtree
   for p in points:
       q = q.backward_insert(p)

def is_leaf(qtree):
   #returns true if node is leaf
   if qtree.divided == False:
       return True
   else:
       return False
   
barycenter = {}
cost = 0


def euclidean_dist(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def positive_flow(qtree):    #return number of distributions with positive flow
    return len(np.where(qtree.flow > 0.00000000000001)[0])

def negative_flow(qtree):    #return number of distributions with negative flow
    return len(np.where(qtree.flow < -0.00000000000001)[0])

def initialize(qtree, cost_func, k):
    # first step in DP algorithm: push all mass up
    global cost
    if qtree == None:
        return
    if is_leaf(qtree):
        pt = qtree.points[0]
        qtree.flow = np.array(pt.data)
        # for m in pt.data:
        #     cost += m* cost_func(pt.x, qtree.x, pt.y, qtree.y)
        return

    initialize(qtree.topleft, cost_func, k)
    initialize(qtree.topright, cost_func, k)
    initialize(qtree.botleft, cost_func, k)
    initialize(qtree.botright, cost_func, k)
    qtree.flow = np.zeros(k)
    if qtree.topleft != None:
        qtree.flow += qtree.topleft.flow
        qtree.topleft.cost_to_parent = cost_func(qtree.x, qtree.topleft.x, qtree.y, qtree.topleft.y)
        cost += np.sum(qtree.topleft.flow * qtree.topleft.cost_to_parent)
    if qtree.topright != None:
        qtree.flow += qtree.topright.flow
        qtree.topright.cost_to_parent = cost_func(qtree.x, qtree.topright.x, qtree.y, qtree.topright.y)
        cost += np.sum(qtree.topright.flow * qtree.topright.cost_to_parent)
    if qtree.botleft != None:
        qtree.flow += qtree.botleft.flow
        qtree.botleft.cost_to_parent = cost_func(qtree.x, qtree.botleft.x, qtree.y, qtree.botleft.y)
        cost += np.sum(qtree.botleft.flow * qtree.botleft.cost_to_parent)
    if qtree.botright != None:
        qtree.flow += qtree.botright.flow
        qtree.botright.cost_to_parent = cost_func(qtree.x, qtree.botright.x, qtree.y, qtree.botright.y)
        cost += np.sum(qtree.botright.flow * qtree.botright.cost_to_parent)

def minimize_path_cost(qtree):
    # compute minimum cost children and augmenting path costs
    qtree.augment_path_cost = 0
    if qtree.botleft != None:
        c = qtree.botleft.augment_path_cost + qtree.botleft.augment_cost
        if c < qtree.augment_path_cost:
            qtree.augment_path_cost = c
            qtree.min_cost_child = qtree.botleft
    if qtree.botright != None:
        c = qtree.botright.augment_path_cost + qtree.botright.augment_cost
        if c < qtree.augment_path_cost:
            qtree.augment_path_cost = c
            qtree.min_cost_child = qtree.botright
    if qtree.topleft != None:
        c = qtree.topleft.augment_path_cost + qtree.topleft.augment_cost
        if c < qtree.augment_path_cost:
            qtree.augment_path_cost = c
            qtree.min_cost_child = qtree.topleft
    if qtree.topright != None:
        c = qtree.topright.augment_path_cost + qtree.topright.augment_cost
        if c < qtree.augment_path_cost:
            qtree.augment_path_cost = c
            qtree.min_cost_child = qtree.topright
   

def update_augment_mass(qtree, k):
    # compute mass to be augmented
    qtree.augment_mass = 0
    if qtree.augment_path_cost < 0:
        mask = np.where(qtree.min_cost_child.flow > 0.000000000000001)
        nonzero_flow = qtree.min_cost_child.flow[mask]
        if len(nonzero_flow) == 0:
            return
        min_flow = np.min(nonzero_flow)
        if qtree.min_cost_child.augment_mass == 0:
            qtree.augment_mass = min_flow
        else:
            qtree.augment_mass = min(min_flow, qtree.min_cost_child.augment_mass)

def compute_augmenting_path(qtree, k):
    # compute augmenting path with minimal cost
    if qtree == None:
        return
    if is_leaf(qtree):
        k1 = len(np.where(qtree.flow > 0.000000000000001)[0])     # number of distributions with positive flow
        qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
        return

    compute_augmenting_path(qtree.botleft, k)
    compute_augmenting_path(qtree.botright, k)
    compute_augmenting_path(qtree.topleft, k)
    compute_augmenting_path(qtree.topright, k)

    k1 = len(np.where(qtree.flow > 0.000000000000001)[0])      # number of distributions with positive flow
    qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
    minimize_path_cost(qtree)
    update_augment_mass(qtree, k)

def push_flow(qtree, cost_func, k, push_mass):
    # push flow down the tree and update barycenter
    global barycenter

    if qtree.min_cost_child == None:
        # if is_leaf(qtree):
        #     cost -= push_mass * cost_func(qtree.x, qtree.points[0].x, 
        #                                   qtree.y, qtree.points[0].y) * positive_flow(qtree)
        qtree.mass += push_mass
        if (qtree.x, qtree.y) not in barycenter:
            barycenter[(qtree.x, qtree.y)] = 0
        barycenter[(qtree.x, qtree.y)] += push_mass
        qtree.flow -= push_mass
        k1 = len(np.where(qtree.flow > 0.000000000000001)[0])
        qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
        return

    push_flow(qtree.min_cost_child, cost_func, k, push_mass)

    qtree.flow -= push_mass
    k1 = len(np.where(qtree.flow > 0.000000000000001)[0])
    qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
    qtree.min_cost_child = None
    minimize_path_cost(qtree)
    update_augment_mass(qtree, k)

# def get_barycenter(qtree):
#     global barycenter
   
#     if qtree == None:
#         return
#     if qtree.mass > 0:
#         barycenter[(qtree.x, qtree.y)] = qtree.mass
#     get_barycenter(qtree.topleft)
#     get_barycenter(qtree.topright)
#     get_barycenter(qtree.botleft)
#     get_barycenter(qtree.botright)

def compute_barycenter(qtree, cost_func, k):
    # compute barycenteer with DP algorithm
    global cost
    global barycenter

    initialize(qtree, cost_func, k)
    qtree.mass = 1
    compute_augmenting_path(qtree, k)
    cost += qtree.augment_path_cost*qtree.augment_mass

    while qtree.augment_path_cost < 0 and qtree.mass > 0:
        qtree.mass -= qtree.augment_mass
        push_flow(qtree, euclidean_dist, k, qtree.augment_mass)
        cost += qtree.augment_path_cost*qtree.augment_mass
    if qtree.mass > 0.00000000000001:
        barycenter[(qtree.x, qtree.y)] = qtree.mass


def find_new_root(qtree):
    # find new root for computing dual weights
    if qtree.parent == None:
        res = None
        if qtree.topleft != None:
            res = find_new_root(qtree.topleft)
        if res == None and qtree.topright != None:
            res = find_new_root(qtree.topright)
        if res == None and qtree.botleft != None:
            res = find_new_root(qtree.botleft)
        if res == None and qtree.botright != None:
            res = find_new_root(qtree.botright)
        return res
    if qtree.mass > 0:
        return qtree
    if is_leaf(qtree):
        return None
    res = None
    if qtree.topleft != None:
        res = find_new_root(qtree.topleft)
    if res == None and qtree.topright != None:
        res = find_new_root(qtree.topright)
    if res == None and qtree.botleft != None:
        res = find_new_root(qtree.botleft)
    if res == None and qtree.botright != None:
        res = find_new_root(qtree.botright)
    return res


def DFS_dual_weights(new, parent, cost_func, k):
    # compute dual weights, new is current node, parent is current node's parent in rerooted tree,
    # new.parent is parent of current node in original tree
    global dualweights
    if parent != None and new.parent != parent:
        edgecost = cost_func(new.x, parent.x, new.y, parent.y)
        new.dualweight = np.zeros(k)
        k1 = negative_flow(parent)
        k1rev = positive_flow(parent)
        alpha = (k1 - k1rev)*edgecost

        pos_flow_mask = np.where(parent.flow > 0.00000000000001)
        neg_flow_mask = np.where(parent.flow < -0.00000000000001)
        zero_flow_mask = np.where((parent.flow <= 0.00000000000001) & (parent.flow >= -0.00000000000001))
        new.dualweight[neg_flow_mask] = parent.dualweight[neg_flow_mask] + edgecost
        new.dualweight[pos_flow_mask] = parent.dualweight[pos_flow_mask] - edgecost
        if len(zero_flow_mask[0]) > 0:
            new.dualweight[zero_flow_mask] = parent.dualweight[zero_flow_mask] + min(edgecost, (-np.sum(parent.dualweight) + new.augment_path_cost - alpha)/(k-k1-k1rev))
        
        dualweights[new.id] = new.dualweight
    elif parent != None:
        edgecost = cost_func(new.x, parent.x, new.y, parent.y)
        new.dualweight = np.zeros(k)
        k1 = positive_flow(new)
        k1rev = negative_flow(new)
        alpha = (k1 - k1rev)*edgecost

        pos_flow_mask = np.where(new.flow > 0.00000000000001)
        neg_flow_mask = np.where(new.flow < -0.00000000000001)
        zero_flow_mask = np.where((new.flow <= 0.00000000000001) & (new.flow >= -0.00000000000001))
        new.dualweight[neg_flow_mask] = parent.dualweight[neg_flow_mask] - edgecost
        new.dualweight[pos_flow_mask] = parent.dualweight[pos_flow_mask] + edgecost
        if len(zero_flow_mask) > 0:
            new.dualweight[zero_flow_mask] = parent.dualweight[zero_flow_mask] + min(edgecost, (-np.sum(parent.dualweight) + new.augment_path_cost - alpha)/(k-k1-k1rev))

        dualweights[new.id] = new.dualweight

    if new.topleft != None and new.topleft != parent:
        DFS_dual_weights(new.topleft, new, cost_func, k)
    if new.topright != None and new.topright != parent:
        DFS_dual_weights(new.topright, new, cost_func, k)
    if new.botleft != None and new.botleft != parent:
        DFS_dual_weights(new.botleft, new, cost_func, k)
    if new.botright != None and new.botright != parent:
        DFS_dual_weights(new.botright, new, cost_func, k)
    if new.parent != None and new.parent != parent:
        DFS_dual_weights(new.parent, new, cost_func, k)


def recompute_cstar(new, parent, cost_func, k):
    if is_leaf(new) and parent != None:
        k1 = positive_flow(new)
        new.augment_path_cost = 0
        new.augment_cost = (k-2*k1)*cost_func(new.x, parent.x, new.y, parent.y)
        return

    if new.topleft != None and new.topleft != parent:
        recompute_cstar(new.topleft, new, cost_func, k)
    if new.topright != None and new.topright != parent:
        recompute_cstar(new.topright, new, cost_func, k)
    if new.botleft != None and new.botleft != parent:
        recompute_cstar(new.botleft, new, cost_func, k)
    if new.botright != None and new.botright != parent:
        recompute_cstar(new.botright, new, cost_func, k)
    if new.parent != None and new.parent != parent:
        recompute_cstar(new.parent, new, cost_func, k)

    if parent == new.parent and parent != None:
        k1 = positive_flow(new)
        new.augment_cost = (k-2*k1)*cost_func(new.x, parent.x, new.y, parent.y)
    elif parent != None:
        k1 = negative_flow(parent)
        new.augment_cost = (k-2*k1)*cost_func(new.x, parent.x, new.y, parent.y)

    new.augment_path_cost = 0
    if new.botleft != None and new.botleft != parent:
        c = new.botleft.augment_path_cost + new.botleft.augment_cost
        if c < new.augment_path_cost:
            new.augment_path_cost = c
    if new.botright != None and new.botright != parent:
        c = new.botright.augment_path_cost + new.botright.augment_cost
        if c < new.augment_path_cost:
            new.augment_path_cost = c
    if new.topleft != None and new.topleft != parent:
        c = new.topleft.augment_path_cost + new.topleft.augment_cost
        if c < new.augment_path_cost:
            new.augment_path_cost = c
    if new.topright != None and new.botright != parent:
        c = new.topright.augment_path_cost + new.topright.augment_cost
        if c < new.augment_path_cost:
            new.augment_path_cost = c
    if new.parent != None and new.parent != parent:
        c = new.parent.augment_path_cost + new.parent.augment_cost
        if c < new.augment_path_cost:
            new.augment_path_cost = c

# dualweightsum = 0
# def printdualweights(qtree):
#     global dualweightsum
#     if is_leaf(qtree):
#         for i in range(len(qtree.dualweight)):
#             dualweightsum += qtree.dualweight[i]*qtree.points[0].data[i]

#     #print(qtree.augment_path_cost)
#     if qtree.topleft != None:
#         printdualweights(qtree.topleft)
#     if qtree.topright != None:
#         printdualweights(qtree.topright)
#     if qtree.botleft != None:
#         printdualweights(qtree.botleft)
#     if qtree.botright != None:
#         printdualweights(qtree.botright)


def compute_dual_weights(qtree, cost_func, k):
   global dualweights
   newroot = find_new_root(qtree)
   newroot.dualweight = np.zeros(k)
   dualweights[newroot.id] = newroot.dualweight
   recompute_cstar(newroot, None, cost_func, k)
   DFS_dual_weights(newroot, None, cost_func, k)

# cost = 0
# barycenter = {}

# qtree = quadtree(0, 0, 4)
# qtree.insert(point(1, 1.25, [1.0, 0, 1.0]))
# qtree.insert(point(-1, -1.25, [0, 1.0, 0]))
# qtree.killemptychildren()

# compute_barycenter(qtree, euclidean_dist, 3)
# print("COST", cost)
# print(barycenter)
# #qtree.printsub()
# compute_dual_weights(qtree, euclidean_dist, 3)
# printdualweights(qtree)
# print(dualweightsum)
   

#test on mnist zeros

def normalize_image(im):
    # rescale pixel values to sum to 1
    s = np.sum(im)
    normalized_im = im/s
    return normalized_im

def images_to_points(images):
    # process image into list of points with masses
    points = []
    m = 0
    n = 0
    for im in images:
        if im.shape[0] > m:
            m = im.shape[0]
        if im.shape[1] > n:
            n = im.shape[1]

    for i in range(len(images)):
        row_diff = m - images[i].shape[0]
        col_diff = n - images[i].shape[1]
        images[i] = np.pad(images[i], ((math.floor(row_diff/2), math.ceil(row_diff/2)), (math.floor(col_diff/2), math.ceil(col_diff/2))),
                    'constant', constant_values=0)

    for i in range(m):
        for j in range(n):
            p = point(i, j, [im[i][j] for im in images])
            points.append(p)
            
    return points, (m, n)


def id_nodes(qtree):
    # requires global id counter and id dictionary
    #givve each node in tree a unique id
    global id
    global iddict
    qtree.id = id
    iddict[id] = qtree
    if len(qtree.points) != 0:
        iddict[(qtree.points[0].x, qtree.points[0].y)] = qtree
    id += 1
    if qtree.botleft != None:
        id_nodes(qtree.botleft)
    if qtree.botright != None:
        id_nodes(qtree.botright)
    if qtree.topleft != None:
        id_nodes(qtree.topleft)
    if qtree.topright != None:
        id_nodes(qtree.topright)

# import glob
# from PIL import Image
# zero_images = []
# for filename in glob.glob('testing images/mnist_zeros/*.png'): 
#     im = np.array(Image.open(filename))
#     normalized_im = normalize_image(im)
#     zero_images.append(normalized_im)
# zero_image_points, zero_image_size = images_to_points(zero_images[:3])

# cost = 0
# barycenter = {}
# mnist_sq_x, mnist_sq_y, mnist_sq_l = getboundingbox(zero_image_points)
# mnist_qtree = quadtree(mnist_sq_x, mnist_sq_y, mnist_sq_l)
# insert_list(mnist_qtree, zero_image_points)
# # for p in zero_image_points:
# #     mnist_qtree0.insert(p)
# mnist_qtree.killemptychildren()
# compute_barycenter(mnist_qtree, euclidean_dist, 3)
# print(cost)
# compute_dual_weights(mnist_qtree, euclidean_dist, 3)
# printdualweights(mnist_qtree)
# print(dualweightsum)

edgesdict = {}
def traverse_tree_spanner(qtree, const, id):
    # NOT USED
    global spanner
    if qtree.length <= const:
        if qtree.length not in spanner:
            spanner[qtree.length] = []
        spanner[qtree.length].append(qtree)

    if qtree.botleft != None:
        traverse_tree_spanner(qtree.botleft, const)
    if qtree.botright != None:
        traverse_tree_spanner(qtree.botright, const)
    if qtree.topleft != None:
        traverse_tree_spanner(qtree.topleft, const)
    if qtree.topright != None:
        traverse_tree_spanner(qtree.topright, const)

#CAN IGNORE, supposed to construct a proper spanner but is not used
global spanner
def construct_spanner(qtree, epsilon, spread, level, d):
   if level <= math.log2(spread):
       const = qtree.length*epsilon/(2*d*math.log2(spread))
       traverse_tree_spanner(qtree, const)
   else:
       construct_spanner(qtree.topleft, epsilon, spread, level - 1, d)
       if spanner.isempty():
           construct_spanner(qtree.topright, epsilon, spread, level - 1, d)
       if spanner.isempty():
           construct_spanner(qtree.botleft, epsilon, spread, level - 1, d)
       if spanner.isempty():
           construct_spanner(qtree.botright, epsilon, spread, level - 1, d)

#gets a list of all leaf nodes
def get_leafs(qtree):
   global leafnodes
   if is_leaf(qtree):
       leafnodes.append(qtree)
       return
   if qtree.botleft != None:
       get_leafs(qtree.botleft)
   if qtree.botright != None:
       get_leafs(qtree.botright)
   if qtree.topleft != None:
       get_leafs(qtree.topleft)
   if qtree.topright != None:
       get_leafs(qtree.topright)


global leafnodes
#constructs a complete graph on the leafs
def spanner_with_images(qtree):
   global edgesdict
   global leafnodes
   global numedges
   get_leafs(qtree)
   add_tree_edges(qtree)
   for u in leafnodes:
       if u.id not in edgesdict:
           edgesdict[u.id] = set([])
       for v in leafnodes:
           if u != v:
               numedges += 1
               edgesdict[u.id].add(v.id)


#adds tree edges to the edgesdict and counts the number of edges
def add_tree_edges(qtree):
   global edgesdict
   global numedges
   if qtree.parent != None:

       if qtree.id not in edgesdict:
           edgesdict[qtree.id] = set([])
       if qtree.parent.id not in edgesdict:
           edgesdict[qtree.parent.id] = set([])
       edgesdict[qtree.id].add(qtree.parent.id)
       edgesdict[qtree.parent.id].add(qtree.id)
       numedges += 2

   if is_leaf(qtree):
       return
   if qtree.botleft != None:
       add_tree_edges(qtree.botleft)
   if qtree.botright != None:
       add_tree_edges(qtree.botright)
   if qtree.topleft != None:
       add_tree_edges(qtree.topleft)
   if qtree.topright != None:
       add_tree_edges(qtree.topright)
 

#initializes the adjacency matrix
def construct_adjacency_matrix(qtree, k):
   global edgesdict
   global adjacency_matrix
   spanner_with_images(qtree)
   numnodes = len(edgesdict.keys())
   adjacency_matrix = np.zeros((numnodes, numnodes, k))   

# edgesdict = {0:[2], 1:[0,2], 2:[0,1]}
# construct_adjacency_matrix(5)
# print(adjacency_matrix)
           
#adds the tree flows to the adjacency 
def add_tree_flows_adjmatrix(qtree, k):
   global adjacency_matrix

   if is_leaf(qtree):
       return

   u = qtree.id
   
   #for a given child, add the flow to matrix
   if qtree.botleft != None:
       add_tree_flows_adjmatrix(qtree.botleft, k) #recurse
       v = qtree.botleft.id
       pos_flow_mask = np.where(qtree.botleft.flow > 0.000000000000001)
       neg_flow_mask = np.where(qtree.botleft.flow < - 0.000000000000001)
       adjacency_matrix[v][u][pos_flow_mask] += qtree.botleft.flow[pos_flow_mask]
       adjacency_matrix[u][v][neg_flow_mask] -= qtree.botleft.flow[neg_flow_mask]
    
   #same as previous if
   if qtree.botright != None:
       add_tree_flows_adjmatrix(qtree.botright, k)
       v = qtree.botright.id
       pos_flow_mask = np.where(qtree.botright.flow > 0.000000000000001)
       neg_flow_mask = np.where(qtree.botright.flow < - 0.000000000000001)
       adjacency_matrix[v][u][pos_flow_mask] += qtree.botright.flow[pos_flow_mask]
       adjacency_matrix[u][v][neg_flow_mask] -= qtree.botright.flow[neg_flow_mask]
   
   if qtree.topleft != None:
       add_tree_flows_adjmatrix(qtree.topleft, k)
       v = qtree.topleft.id
       pos_flow_mask = np.where(qtree.topleft.flow > 0.000000000000001)
       neg_flow_mask = np.where(qtree.topleft.flow < - 0.000000000000001)
       adjacency_matrix[v][u][pos_flow_mask] += qtree.topleft.flow[pos_flow_mask]
       adjacency_matrix[u][v][neg_flow_mask] -= qtree.topleft.flow[neg_flow_mask]
   
   if qtree.topright != None:
       add_tree_flows_adjmatrix(qtree.topright, k)
       v = qtree.topright.id
       pos_flow_mask = np.where(qtree.topright.flow > 0.000000000000001)
       neg_flow_mask = np.where(qtree.topright.flow < - 0.000000000000001)
       adjacency_matrix[v][u][pos_flow_mask] += qtree.topright.flow[pos_flow_mask]
       adjacency_matrix[u][v][neg_flow_mask] -= qtree.topright.flow[neg_flow_mask]
               
def leftover_mass_tree(qtree, k, incflows): 
    global ogmass
    if qtree == None:
        return
    if is_leaf(qtree):
        pt = qtree.points[0]
        id = iddict[(pt.x, pt.y)].id
        ogptmass = ogmass[(pt.x, pt.y)]
        leftovermass = np.zeros(k)
        leftovermass = ogptmass - (np.sum(adjacency_matrix[id],0) - incflows[id])
        pt.data = leftovermass

        qtree.reset()
    
    qtree.reset()
    leftover_mass_tree(qtree.botleft, k, incflows)
    leftover_mass_tree(qtree.botright, k, incflows)
    leftover_mass_tree(qtree.topleft, k, incflows)
    leftover_mass_tree(qtree.topright, k, incflows)

def restructure_dual_weights(dualweights):       #for numpy optimization purposes
    m1 = np.tile(dualweights, (len(dualweights), 1, 1))
    m2 = np.transpose(m1, axes=(1, 0, 2))
    return m2 - m1

#core of the mwu alg
def mwu(qtree, cost_func, epsilon, spread, k, numedges, startingguess=None):
    global cost
    global edgesdict #dictionary of all the edges in our graph. Keys are id of nodes and values are list of ids of nodes that have edges to key
    global adjacency_matrix #graph structure, is numedges x numedges x k in size
    global dualweights #dictionary for dual weights, keys are node ids and values are the corresponding dualweights to the node
    global barycenter #dictionary representing the barycenter, keys are coordinates, values are corresponding mass

    global id #counter for ids
    global iddict #dictionary to find the qtree/id for a 

    gstar = cost
    if startingguess != None:
        g = startingguess
    else:
        g = gstar/math.log2(spread)      
    t = (epsilon**-1)*8*((math.log2(spread))**2)*math.log2(k*numedges)
    print(t)
    
    
    while g <= gstar: 
        #initialize the adjacency matrix flows on all edges from the paper
        #adjacency_matrix += np.reciprocal(cost_matrix)*(1/(k*numedges))*g     
        for u in edgesdict:
            for v in edgesdict[u]:
            #    for i in range(k):
            #        adjacency_matrix[u][v][i] = (g/(k*cost_func(iddict[u].x, iddict[v].x, iddict[u].y, iddict[v].y)*numedges))
                adjacency_matrix[u][v] += (g/(k*cost_func(iddict[u].x, iddict[v].x, iddict[u].y, iddict[v].y)*numedges))

        #start iterating                                                                                
        for iteration in range(math.ceil(t)):
            #print(iteration)
            #compute barycenter cost for the leftover mass tree and recompute the dualweights
            incflows = np.sum(adjacency_matrix, 0)
            leftover_mass_tree(qtree, k, incflows)
            cost = 0
            barycenter = {}
            compute_barycenter(qtree, cost_func, k)
            dualweights = np.zeros((len(edgesdict.keys()), k))
            compute_dual_weights(qtree, cost_func, k)
            dual_matrix = restructure_dual_weights(dualweights)     #dual weight differences matrix
            
            #if cost small enough then we have found our solution
            if cost <= epsilon*g:
                add_tree_flows_adjmatrix(qtree, k) #add the flows of our edges in the tree to the adjacency matrix to compute the overall cost
                print(barycenter)
                cost = np.sum(np.abs(adjacency_matrix - np.transpose(adjacency_matrix, (1, 0, 2)))*cost_matrix)/2 #divide by 2 due to double counting
                return
            else:
                #if we didnt find a proper solution perform the mwu
                adjacency_matrix = adjacency_matrix * np.exp(epsilon/(2*(math.log2(spread)**2))*(dual_matrix/cost_matrix))
                
                #recompute the newcost to rescale the flows by g/newcost
                newcost = np.sum(np.abs(adjacency_matrix - np.transpose(adjacency_matrix, (1, 0, 2)))*cost_matrix)/2
                adjacency_matrix = adjacency_matrix * (g/(newcost)) #rescale the flows

        g = (1+epsilon)*g

    return



def construct_cost_matrix(k, cost_func):
    global edgesdict
    global cost_matrix
    numnodes = len(edgesdict.keys())
    cost_matrix = np.ones((numnodes, numnodes, k))  #initialize to 1 to avoid dividing by zero
    for u in edgesdict:
        for v in edgesdict[u]:
            edgecost = cost_func(iddict[u].x, iddict[v].x, iddict[u].y, iddict[v].y)
            cost_matrix[u][v] = np.full(k, edgecost)

            



#plots the barycenter on a graph, you don't need to worry about this
def plotbarycenter(barycenterdict, point_size, image_size):
   coords = barycenterdict.items()
   x = [pt[0][0] for pt in coords]
   y = [pt[0][1] for pt in coords]
   t = [pt[1] for pt in coords]
   # t = np.linspace(0, 1, len(x))
   fig, ax = plt.subplots()
   sc = ax.scatter(y, x, c=t, marker='.', s=point_size)
   fig.colorbar(sc, label="mass")
   plt.xlim(0, image_size[1])
   plt.ylim(0, image_size[0])
   plt.gca().invert_yaxis()
   plt.show()



#run example for distributions at (1,1) - [1, 0, 1] and (1,-1) - [0, 1, 0]

cost = 0
barycenter = {}

id = 0
iddict = {}
dualweights = {}

numedges = 0
edgesdict = {}
adjacency_matrix = []
leafnodes = []
cost_matrix = []


testqtree = quadtree(0, 0, 4)
ptlist = [point(1, 1, [1, 0, 1]), point(1, -1, [0, 1, 0])]
insert_list(testqtree, ptlist)
testqtree.killemptychildren()
id_nodes(testqtree)
# testqtree.printsub()

ogmass = {}
for pt in ptlist:
    ogmass[(pt.x, pt.y)] = copy.deepcopy(pt.data)

compute_barycenter(testqtree, euclidean_dist, 3)
# print("COST", cost)
# print(barycenter)
# #qtree.printsub()
compute_dual_weights(testqtree, euclidean_dist, 3)
# printdualweights(qtree)
# print(dualweightsum)
# print(dualweights)

construct_adjacency_matrix(testqtree, 3)
construct_cost_matrix(3, euclidean_dist)

mwu(testqtree, euclidean_dist, .2, 3, 3, numedges)
print(cost)
print(adjacency_matrix)

# import glob
# zero_images = []
# for filename in glob.glob('testing images/mnist_zeros/*.png'): 
#    im = np.array(Image.open(filename))
#    normalized_im = normalize_image(im)
#    zero_images.append(normalized_im)
# zero_image_points, zero_image_size = images_to_points(zero_images[:3])
# cost = 0
# barycenter = {}
# mnist_sq_x, mnist_sq_y, mnist_sq_l = getboundingbox(zero_image_points)
# mnist_qtree = quadtree(mnist_sq_x, mnist_sq_y, mnist_sq_l)
# insert_list(mnist_qtree, zero_image_points)
# # for p in zero_image_points:
# #     mnist_qtree0.insert(p)

# ogmass = {}
# for pt in zero_image_points:
#     ogmass[(pt.x, pt.y)] = copy.deepcopy(pt.data)

# k = 3

# mnist_qtree.killemptychildren()
# id_nodes(mnist_qtree)
# compute_barycenter(mnist_qtree, euclidean_dist, k)
# plotbarycenter(barycenter, 100, zero_image_size)
# compute_dual_weights(mnist_qtree, euclidean_dist, k)
# construct_adjacency_matrix(mnist_qtree, k)
# construct_cost_matrix(k, euclidean_dist)
# print(cost)
# spread = math.sqrt(27**2+27**2)
# barycenter = {}
# mwu(mnist_qtree, euclidean_dist, .2, spread, k, numedges, zero_image_points,(mnist_sq_x, mnist_sq_y, mnist_sq_l))


# totalmass = 0
# newbc = {}
# for key in barycenter:
#    totalmass += barycenter[key]
#    if barycenter[key] > .0000002:
#        newbc[key] = barycenter[key]
# # plotbarycenter(barycenter, 100, zero_image_size)
# plotbarycenter(newbc, 100, zero_image_size)
# print(totalmass)
# print(cost)
# totalmass = 0
# for key in newbc:
#    totalmass += newbc[key]
# print(totalmass)
