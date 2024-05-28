cost = 0
transport_plan = {}

def is_leaf(qtree):
    if qtree.divided == False:
        return True
    else:
        return False

def compute_ot(qtree, cost_func):
    if is_leaf(qtree): # there should only be one point 
        p = qtree.square.points[0]
        val = min(p.data)
        if val > 0: # if both distributions have mass at same point
            transport_plan[(p.x, p.y)] = ((p.x, p.y), val)
            p.data = [m-val for m in p.data]
        if max(p.data) > 0: # if there is still mass at point push up
            return qtree.square.points
        else: return []