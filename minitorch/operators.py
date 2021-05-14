import math


def mul(x, y):
    return x * y


def id(x):
    return x


def neg(x):
    return -x


def add(x, y):
    return x + y


def lt(x, y):
    return 1.0 if x < y else 0.0


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return x if x > 0 else 0.0


def relu_back(x, y):
    return y if x > 0 else 0.0


def map(fn):
    def _fn(ls):
        return list(map(fn, ls))

    return _fn


def negList(ls):
    return map(neg)(ls)


def zipWith(fn):
    def _fn(ls1, ls2):
        return list(map(fn, ls1, ls2))

    return _fn


def addLists(ls1, ls2):
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    def _fn(ls):
        if len(ls) == 2:
            return fn(ls[0], ls[1]) 
            
        return fn(ls[start], _fn(ls[:start - 1]))

    return _fn


def sum(ls):
    return reduce(add, 0)(ls)


def prod(ls):
    return reduce(mul, 0)(ls)
