import math

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

    def __init__(self, square, divided=False):
        #initialize quadtree object
        self.square = square
        self.divided = divided
        self.topleft = None
        self.topright = None
        self.botleft = None
        self.botright = None
        self.parent = None
        self.flow = []
        self.mass = 0
        self.min_cost_child = None
        self.cost_to_parent = 0
        self.augment_cost = 0
        self.augment_path_cost = 0
        self.augment_mass = 0

    def subdivide(self):
        #divide up the current cell
        x, y, l = self.square.x, self.square.y, self.square.l

        topleft = square(x-l/4, y+l/4, l/2)
        self.topleft = quadtree(topleft)
        self.topleft.parent = self

        topright = square(x+l/4, y+l/4, l/2)
        self.topright = quadtree(topright)
        self.topright.parent = self

        botleft = square(x-l/4, y-l/4, l/2)
        self.botleft = quadtree(botleft)
        self.botleft.parent = self

        botright = square(x+l/4, y-l/4, l/2)
        self.botright = quadtree(botright)
        self.botright.parent = self

        self.divided = True

        for point in self.square.points:
            leaf = self.topleft.insert(point)
            if leaf == None:
                leaf = self.topright.insert(point)
            if leaf == None:
                leaf = self.botleft.insert(point)
            if leaf == None:
                leaf = self.botright.insert(point)

        self.square.points = []

    def insert(self, point):
        #insert a point into the quadtree starting at root
        if not self.square.contains(point):
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
        elif len(self.square.points) == 0:
            self.square.points.append(point)
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
        if not self.divided and len(self.square.points) != 0:
            return

        if not self.topleft.divided and len(self.topleft.square.points) == 0:
            self.topleft = None
        else:
            self.topleft.killemptychildren()
        
        if not self.topright.divided and len(self.topright.square.points) == 0:
            self.topright = None
        else:
            self.topright.killemptychildren()

        if not self.botleft.divided and len(self.botleft.square.points) == 0:
            self.botleft = None
        else:
            self.botleft.killemptychildren()

        if not self.botright.divided and len(self.botright.square.points) == 0:
            self.botright = None
        else:
            self.botright.killemptychildren()

        

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

    def getlistofpoints(self, lst):
        #gets list of points from a tree
        #input list in form of [[xcoords], [ycoords], [distribution]
        #colors are hard coded to work with 2 distributinos
        if self.divided is False and len(self.square.points) > 0:
            lst[0].append(self.square.points[0].x)
            lst[1].append(self.square.points[0].y)
            if min(self.square.points[0].data) > 0:
                lst[2].append(2)
            elif self.square.points[0].data[0] > 0:
                lst[2].append(0)
            elif self.square.points[0].data[1] > 0:
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
        
        
        line1x = [self.square.x - self.square.l / 2, self.square.x - self.square.l / 2]
        line1y = [self.square.y - self.square.l / 2, self.square.y + self.square.l / 2]
        lst[0].append(line1x)
        lst[1].append(line1y)

        line2x = [self.square.x - self.square.l / 2, self.square.x + self.square.l / 2]
        line2y = [self.square.y + self.square.l / 2, self.square.y + self.square.l / 2]
        lst[0].append(line2x)
        lst[1].append(line2y)

        line3x = [self.square.x + self.square.l / 2, self.square.x + self.square.l / 2]
        line3y = [self.square.y + self.square.l / 2, self.square.y - self.square.l / 2]
        lst[0].append(line3x)
        lst[1].append(line3y)

        line4x = [self.square.x - self.square.l / 2, self.square.x + self.square.l / 2]
        line4y = [self.square.y - self.square.l / 2, self.square.y - self.square.l / 2]
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

        upperx = (self.square.x + self.square.l / 2) + .2 * abs(self.square.x + self.square.l / 2)
        uppery = (self.square.y + self.square.l / 2) + .2 * abs(self.square.y + self.square.l / 2)

        lowerx = (self.square.x - self.square.l / 2) - .2 * abs(self.square.x - self.square.l / 2)
        lowery = (self.square.y - self.square.l / 2) - .2 * abs(self.square.y - self.square.l / 2)
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


def initialize(qtree, cost_func, k):
    global cost
    if qtree == None:
        return
    if is_leaf(qtree):
        pt = qtree.square.points[0]
        for m in pt.data:
            qtree.flow.append(m)
            cost += m* cost_func(pt.x, qtree.square.x, pt.y, qtree.square.y)
        return
    
    initialize(qtree.topleft, cost_func, k)
    initialize(qtree.topright, cost_func, k)
    initialize(qtree.botleft, cost_func, k)
    initialize(qtree.botright, cost_func, k)
    qtree.flow = [0 for x in range(k)]
    if qtree.topleft != None:
        for i in range(k):
            qtree.flow[i] += qtree.topleft.flow[i]
            qtree.topleft.cost_to_parent = cost_func(qtree.square.x, qtree.topleft.square.x, qtree.square.y, qtree.topleft.square.y)
            cost += qtree.topleft.flow[i] * qtree.topleft.cost_to_parent
    if qtree.topright != None:
        for i in range(k):
            qtree.flow[i] += qtree.topright.flow[i]
            qtree.topright.cost_to_parent = cost_func(qtree.square.x, qtree.topright.square.x, qtree.square.y, qtree.topright.square.y)
            cost += qtree.topright.flow[i] * qtree.topright.cost_to_parent
    if qtree.botleft != None:
        for i in range(k):
            qtree.flow[i] += qtree.botleft.flow[i]
            qtree.botleft.cost_to_parent = cost_func(qtree.square.x, qtree.botleft.square.x, qtree.square.y, qtree.botleft.square.y)
            cost += qtree.botleft.flow[i] * qtree.botleft.cost_to_parent
    if qtree.botright != None:
        for i in range(k):
            qtree.flow[i] += qtree.botright.flow[i]
            qtree.botright.cost_to_parent = cost_func(qtree.square.x, qtree.botright.square.x, qtree.square.y, qtree.botright.square.y)
            cost += qtree.botright.flow[i] * qtree.botright.cost_to_parent

def positive_flow(qtree, k):    #return number of distributions with positive flow
    n = 0      
    for f in qtree.flow:
        if f > 0.000000000000001:
            n+=1
    return n

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
        min_flow = float("inf")
        for i in range(k):
            f = qtree.min_cost_child.flow[i]
            if f > 0.000000000000001:
                if f < min_flow:
                    min_flow = f
        if qtree.min_cost_child.augment_mass == 0:
            qtree.augment_mass = min_flow
        else:
            qtree.augment_mass = min(min_flow, qtree.min_cost_child.augment_mass)

def compute_augmenting_path(qtree, k):
    if qtree == None:
        return
    if is_leaf(qtree):
        k1 = positive_flow(qtree, k)      # number of distributions with positive flow
        qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
        return
    
    compute_augmenting_path(qtree.botleft, k)
    compute_augmenting_path(qtree.botright, k)
    compute_augmenting_path(qtree.topleft, k)
    compute_augmenting_path(qtree.topright, k)

    k1 = positive_flow(qtree, k)      # number of distributions with positive flow
    qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
    minimize_path_cost(qtree)
    update_augment_mass(qtree, k)

def push_flow(qtree, cost_func, k, push_mass):
    global barycenter

    if qtree.min_cost_child == None:
        # if is_leaf(qtree):
        #     cost -= push_mass * cost_func(qtree.square.x, qtree.square.points[0].x, 
        #                                   qtree.square.y, qtree.square.points[0].y)
        qtree.mass = push_mass
        barycenter[(qtree.square.x, qtree.square.y)] = qtree.mass
        for i in range(k):
            qtree.flow[i] -= push_mass
        k1 = positive_flow(qtree, k)
        qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
        return
    
    push_flow(qtree.min_cost_child, cost_func, k, push_mass)

    for i in range(k):
        qtree.flow[i] -= push_mass
    k1 = positive_flow(qtree, k)
    qtree.augment_cost = (k - 2*k1) * qtree.cost_to_parent
    qtree.min_cost_child = None
    minimize_path_cost(qtree)
    update_augment_mass(qtree, k)

# def get_barycenter(qtree):
#     global barycenter
    
#     if qtree == None:
#         return
#     if qtree.mass > 0:
#         barycenter[(qtree.square.x, qtree.square.y)] = qtree.mass
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
 