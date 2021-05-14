from minitorch import operators
from hypothesis import given
from hypothesis.strategies import lists
from .strategies import small_floats, assert_close
import pytest


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_add_and_mul(x, y):
    assert_close(operators.mul(x, y), x * y)
    assert_close(operators.add(x, y), x + y)
    assert_close(operators.neg(x), -x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert operators.relu(a) == a
    else:
        assert operators.relu(a) == 0.0


# Task 0.2
# Property Testing


@pytest.mark.task0_2
def test_symmetric():
    """
    A test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    a = list(range(-500, 501))
    b = list(range(-500, 501))

    for i, j in zip(a, b):
        assert operators.mul(i, j) == operators.mul(j, i)


@pytest.mark.task0_2
def test_distribute():
    r"""
    A test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """

    a = list(range(-500, 501))
    b = list(range(-500, 501))
    c = list(range(-500, 501))

    for i, j, z in zip(a, b, c):
        assert operators.mul(z, operators.add(i, j)) == operators.add(
            operators.mul(z, i), operators.mul(z, j))


@pytest.mark.task0_2
def test_other():
    """
    Write a test that ensures some other property holds for your functions.
    """
    a = 10.0

    assert operators.neg(a) == -10.0
    assert operators.mul(a, -1) == -1 * a


# HIGHER ORDER


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    assert_close(operators.addLists([a, b], [c, d]), [a + c, b + d])


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_property(ls1, ls2):
    """
    A test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    ls1 = list(range(1, 4))
    ls2 = list(range(1, 4))

    assert operators.addLists(ls1, ls2) == [2, 4, 6]


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(operators.sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(operators.prod([x, y, z]), x * y * z)
