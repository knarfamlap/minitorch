import math

def mul(x, y):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    return x * y

def id(x):
    """[summary]

    Args:
        x ([type]): [description]
    """
    return x

def neg(x):
    """[summary]

    Args:
        x ([type]): [description]
    """
    return -x


def add(x, y):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
    """
    return x + y


def lt(x, y):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
    """
    
    return 1.0 if x < y else 0.0

def sigmoid(x):
    """[summary]

    Args:
        x ([type]): [description]
    """
    
    return 1.0 / (1.0  + math.exp(-x))

def relu(x):
    """[summary]

    Args:
        x ([type]): [description]
    """ 
    
    return x if x > 0 else 0.0

def relu_back(x, y):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
    """ 
    
    return y if x > 0 else 0.0


def map(fn):
    """[summary]

    Args:
        fn (function): [description]
    """ 
    
    return lambda fn: fn()

def negList(ls):
    """[summary]

    Args:
        ls ([type]): [description]

    Returns:
        [type]: [description]
    """
    return map(neg)(ls)

