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

    return 1.0 / (1.0 + math.exp(-x))


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

    def _fn(ls):
        return list(map(fn, ls))

    return _fn


def negList(ls):
    """[summary]

    Args:
        ls ([type]): [description]

    Returns:
        [type]: [description]
    """
    return map(neg)(ls)


def zipWith(fn):
    """[summary]

    Args:
        fn (function): [description]
    """

    def _fn(ls1, ls2):
        return list(map(fn, ls1, ls2))

    return _fn


def addLists(ls1, ls2):
    """[summary]

    Args:
        ls1 ([type]): [description]
        ls2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    """[summary]

    Args:
        fn (function): [description]
        start ([type]): [description]
    """

    def _fn(ls):
        return fn(ls[start], _fn(ls[:start - 1]))

    return _fn

def sum(ls):
    """[summary]

    Args:
        ls ([type]): [description]

    Returns:
        [type]: [description]
    """
    return reduce(add, 0)(ls)

def prod(ls):
    """[summary]

    Args:
        ls ([type]): [description]

    Returns:
        [type]: [description]
    """
    return reduce(mul, 0)(ls)