from queue import Queue

cost = 0
barycenter = {}

def is_leaf(qtree):
    if qtree.divided == False:
        return True
    else:
        return False

def simple_barycenter(qtree, cost_func):
    global cost
    global barycenter
    k = len(qtree.square.points[0].data)   # number of distributions

    if qtree == None:
        return []

    if is_leaf(qtree):
        p = qtree.square.points[0]
        min_mass = min(p.data)
        if min_mass > 0:
            barycenter[(p.x, p.y)] = min_mass
            for i in range(k):
                p.data[i] -= min_mass
        if max(p.data) > 0.00000000000001:
            return qtree.square.points
        else: 
            return []
    
    qtree.square.points = (simple_barycenter(qtree.topleft, cost_func) + simple_barycenter(qtree.topright, cost_func)
                         + simple_barycenter(qtree.botleft, cost_func) + simple_barycenter(qtree.botright, cost_func))
    if qtree.square.points == []:
        return []
    
    mass = [0 for i in range(k)]
    pt_queue = Queue()
    for p in qtree.square.points:
        for i in range(k):
            mass[i] += p.data[i]
            pt_queue.put(p)
    
    min_mass = min(mass)
    barycenter[(qtree.square.x, qtree.square.y)] = min_mass
    mass_needed = [min_mass for i in range(k)]     # mass still to be paired for each distribution
    while max(mass_needed) > 0.00000000000001:
        p = pt_queue.get()
        for i in range(k):
            m = min(p.data[i], mass_needed[i])
            p.data[i] -= m
            mass_needed[i] -= m
    
    return [p for p in qtree.square.points if max[p.data] > 0.00000000000001]