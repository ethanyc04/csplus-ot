import math
import numpy as np

class point:

    def __init__(self, x, y, data):
        #point data
        self.x = x
        self.y = y
        self.data = data

    def __repr__(self):
        return f'{{"x": {self.x}, "y": {self.y}}}'
    
class square:

    def __init__(self, x, y, l):
        # square cell initializing
        self.x = x
        self.y = y
        self.l = l
        self.points = []

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.l})'

    def contains(self, point):
        # checks if point falls within a cell
        xcheck = self.x - (self.l / 2) <= point.x and self.x + (self.l / 2) >= point.x
        ycheck = self.y - (self.l / 2) <= point.y and self.y + (self.l / 2) >= point.y
        return xcheck and ycheck
    
class quadtree:

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

    def __repr__(self):
        return f'{{"x": {self.x}, "y": {self.y}, "l": {self.l}}}'

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
        print(self)
        print(self.augment_path_cost)
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
    n = 0      
    for f in qtree.flow:
        if f > 0.000000000000001:
            n+=1
    return n

def negative_flow(qtree):    #return number of distributions with negative flow
    n = 0      
    for f in qtree.flow:
        if f < -0.000000000000001:
            n+=1
    return n


def initialize(qtree, cost_func, k):
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
    global barycenter

    if qtree.min_cost_child == None:
        # if is_leaf(qtree):
        #     cost -= push_mass * cost_func(qtree.x, qtree.points[0].x, 
        #                                   qtree.y, qtree.points[0].y) * positive_flow(qtree)
        qtree.mass = push_mass
        barycenter[(qtree.x, qtree.y)] = qtree.mass
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
    global cost

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
    if parent != None and new.parent != parent:
        edgecost = cost_func(new.x, parent.x, new.y, parent.y)
        new.dualweight = [0 for i in range(k)]
        k1 = negative_flow(parent)
        k1rev = positive_flow(parent)
        alpha = (k1 - k1rev)*edgecost
        for i in range(k):
            if parent.flow[i] < -.00000000000001:
                new.dualweight[i] = parent.dualweight[i] + edgecost
            elif parent.flow[i] > 0.0000000000001:
                new.dualweight[i] = parent.dualweight[i] - edgecost
            else:
                new.dualweight[i] = parent.dualweight[i] + min(edgecost, (-sum(parent.dualweight) + new.augment_path_cost - alpha)/(k-k1-k1rev))
    elif parent != None:
        edgecost = cost_func(new.x, parent.x, new.y, parent.y)
        new.dualweight = [0 for i in range(k)]
        k1 = positive_flow(new)
        k1rev = negative_flow(new)
        alpha = (k1 - k1rev)*edgecost

        for i in range(k):
            if new.flow[i] > .00000000000001:
                new.dualweight[i] = parent.dualweight[i] + edgecost
            elif new.flow[i] < -.00000000000001:
                new.dualweight[i] = parent.dualweight[i] - edgecost
            else:
                new.dualweight[i] = parent.dualweight[i] + min(edgecost, (-sum(parent.dualweight) + new.augment_path_cost - alpha)/(k-k1-k1rev))

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

dualweightsum = 0
def printdualweights(qtree):
    global dualweightsum
    if is_leaf(qtree):
        for i in range(len(qtree.dualweight)):
            dualweightsum += qtree.dualweight[i]*qtree.points[0].data[i]

    #print(qtree.augment_path_cost)
    if qtree.topleft != None:
        printdualweights(qtree.topleft)
    if qtree.topright != None:
        printdualweights(qtree.topright)
    if qtree.botleft != None:
        printdualweights(qtree.botleft)
    if qtree.botright != None:
        printdualweights(qtree.botright)


def compute_dual_weights(qtree, cost_func, k):
    newroot = find_new_root(qtree)
    newroot.dualweight = [0 for i in range(k)]
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
    s = np.sum(im)
    normalized_im = im/s
    return normalized_im

def images_to_points(images):
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

import glob
from PIL import Image
zero_images = []
for filename in glob.glob('testing images/mnist_zeros/*.png'): 
    im = np.array(Image.open(filename))
    normalized_im = normalize_image(im)
    zero_images.append(normalized_im)
zero_image_points, zero_image_size = images_to_points(zero_images[:3])

cost = 0
barycenter = {}
mnist_sq_x, mnist_sq_y, mnist_sq_l = getboundingbox(zero_image_points)
mnist_qtree = quadtree(mnist_sq_x, mnist_sq_y, mnist_sq_l)
insert_list(mnist_qtree, zero_image_points)
# for p in zero_image_points:
#     mnist_qtree0.insert(p)
mnist_qtree.killemptychildren()
compute_barycenter(mnist_qtree, euclidean_dist, 3)
print(cost)
compute_dual_weights(mnist_qtree, euclidean_dist, 3)
printdualweights(mnist_qtree)
print(dualweightsum)
    

            
            