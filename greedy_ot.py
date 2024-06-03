from queue import Queue

cost = 0
transport_plan = {}

def is_leaf(qtree):
    if qtree.divided == False:
        return True
    else:
        return False

def compute_ot(qtree, cost_func):
    global cost
    global transport_plan

    if qtree == None:
        return []

    if is_leaf(qtree): # base case: leaf node
        p = qtree.square.points[0] # there should only be one point 
        val = min(p.data)
        if val > 0: # if both distributions have mass at same point
            transport_plan[(p.x, p.y)] = [((p.x, p.y), val)]
            p.data[0] -= val
            p.data[1] -= val
        if max(p.data) > 0: # if there is still mass at point push up
            return qtree.square.points
        else: return []
    #recursive call
    qtree.square.points = (compute_ot(qtree.topleft, cost_func) + compute_ot(qtree.topright, cost_func)
                         + compute_ot(qtree.botleft, cost_func) + compute_ot(qtree.botright, cost_func))
    if qtree.square.points == []:
        return []
    mass = [0, 0] #only two distributions
    dist1_q = Queue()
    dist2_q = Queue()
    for p in qtree.square.points:
        mass[0] += p.data[0]
        mass[1] += p.data[1]
        if p.data[0] > 0:
            dist1_q.put(p)
        else:                    # each point now should only have mass from one distribution after           
            dist2_q.put(p)       # pairing at leaf nodes
    
    val = min(mass)

    while val > 0.00000000000001:
        while dist1_q.empty() == False and dist2_q.empty() == False:
            p1 = dist1_q.get()
            p2 = dist2_q.get()
            massmatch = min(p1.data[0], p2.data[1])
            val -= massmatch
            p1.data[0] -= massmatch
            p2.data[1] -= massmatch
            if (p1.x, p1.y) not in transport_plan:
                transport_plan[(p1.x, p1.y)] = []
            transport_plan[(p1.x, p1.y)].append(((p2.x, p2.y), massmatch))
            cost += cost_func(p1, p2) * massmatch
            if p1.data[0] > 0.00000000000001:
                dist1_q.put(p1)
            if p2.data[1] > 0.00000000000001:
                dist2_q.put(p2)


        # p1 = dist1_q.get()
        # if (p1.x, p1.y) not in transport_plan:
        #     transport_plan[(p1.x, p1.y)] = []
        # if p1.data[0] > val:   # map all points from dist2 
        #     while dist2_q.empty() == False:
        #         p2 = dist2_q.get()
        #         transport_plan[(p1.x, p1.y)].append(((p2.x, p2.y), p2.data[1]))
        #         cost += cost_func(p1, p2) * p2.data[1]
        #         p2.data[1] = 0
        #     p1.data[0] -= val
        #     val = 0

        # else:   # p1.data[0] <= val
        #     m = p1.data[0]
        #     while m > 0.00000000000001:
        #         p2 = dist2_q.get()
        #         if m >= p2.data[1]:
        #             m -= p2.data[1]
        #             val -= p2.data[1]
        #             transport_plan[(p1.x, p1.y)].append(((p2.x, p2.y), p2.data[1]))
        #             cost += cost_func(p1, p2) * p2.data[1]
        #             p1.data[0] -= p2.data[1]
        #             p2.data[1] = 0
        #         else:   # m < p2.data[1]
        #             val -= m
        #             transport_plan[(p1.x, p1.y)].append(((p2.x, p2.y), m))
        #             cost += cost_func(p1, p2) * m
        #             p1.data[0] = 0
        #             p2.data[1] -= m
        #             dist2_q.put(p2)
        #             m = 0
    
    return [p for p in qtree.square.points if max(p.data) > 0.00000000000001]