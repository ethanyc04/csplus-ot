cost = 0
transport_plan = {}

def is_leaf(qtree):
    if qtree.divided == False:
        return True
    else:
        return False

def compute_ot(qtree, cost_func):
    