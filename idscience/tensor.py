from collections import Iterable
from functools import reduce
import operator
import random as _random

from idscience.values import InterChars, Null


class Tensor:
    def __init__(self, array, shape):
        self._array = array if isinstance(array, list) else list(array)
        self._shape = shape if isinstance(shape, tuple) else tuple(shape)

    @property
    def size(self):
        return len(self._array)

    def __len__(self):
        if len(self.shape) == 0:
            raise TypeError('len() of unsized object')
        return self.shape[0]

    @property
    def shape(self):
        return self._shape

    def reshape(x, *shape):
        x = tensor(x)
        axes = [i for i, num in enumerate(shape) if (num == -1 or num is None)]
        if len(axes) > 1:
            raise ValueError('can only specify one unknown dimension')
        if len(axes) == 1:
            shape = list(shape)
            shape[axes[0]] = 1
            shape[axes[0]] = x.size // reduce(operator.mul, shape)
        if reduce(operator.mul, shape) != x.size:
            raise ValueError('cannot reshape array of size {} into shape {}'.format(x.size, shape))
        array = list(x._array)
        return Tensor(array, shape)

    def transpose(x, *axes):
        x = tensor(x)
        if len(axes) != len(x.shape):
            raise ValueError("axes don't match tensor")
        if len(set(axes)) != len(axes):
            raise ValueError('repeated axis in transpose')

        array = x._array
        dst_array = [0] * len(array)
        shape = x.shape
        dst_shape = [shape[axis] for axis in axes]
        bases = [reduce(operator.mul, shape[i + 1:]) for i in range(len(shape) - 1)] + [1]
        dst_bases = [reduce(operator.mul, dst_shape[i + 1:]) for i in range(len(dst_shape) - 1)] + [1]
        dst_bases = [base for _, base in sorted(zip(axes, dst_bases))]

        trans_bases = tuple(zip(bases, dst_bases))
        for i, value in enumerate(array):
            dst_idx = 0
            for base, dst_base in trans_bases:
                dst_idx += i // base * dst_base
                i %= base
            dst_array[dst_idx] = value
        return Tensor(dst_array, dst_shape)

    @property
    def T(self):
        axes = tuple(range(len(self.shape)))[::-1]
        return self.transpose(*axes)

    @staticmethod
    def recursion_element(sequence, shape, idxs, is_enter=True):
        if is_enter:
            if not isinstance(idxs, tuple):
                idxs = (idxs, )
            idxs += (slice(None), ) * (len(shape) - 1)
            idxs = [range(axis_n)[idx] if isinstance(idx, slice) else idx
                    for axis_n, idx in zip(shape, idxs)]
            idxs = [[idx if idx >= 0 else (axis_n + idx), ] if isinstance(idx, int) else idx
                    for axis_n, idx in zip(shape, idxs)]

        for axis_n, idx in zip(shape, idxs):
            if (not isinstance(idx, range)) or (len(idx) != axis_n) or idx.step < 0:
                break
        else:
            if is_enter:
                return sequence, idxs
            return sequence

        idx = idxs[0]
        axis_n = shape[0]
        result = []
        if len(shape) == 1:
            for i in idx:
                if i < 0: i += axis_n
                result.append(sequence[i])
        else:
            batch_size = len(sequence) // axis_n
            for i in idx:
                if i < 0: i += axis_n
                sub_sequence = sequence[i * batch_size: (i + 1) * batch_size]
                result.extend(Tensor.recursion_element(sub_sequence, shape[1:], idxs[1:], is_enter=False))

        if is_enter:
            return result, idxs
        return result

    def __getitem__(self, idxs):
        dst_array, idxs = Tensor.recursion_element(self._array, self.shape, idxs)
        dst_shape = [len(idx) for idx in idxs if len(idx) > 1]
        if len(dst_shape) == 0:
            return dst_array[0]
        return Tensor(dst_array, dst_shape)

    def __setitem__(self, idxs, value):
        idxs, _ = Tensor.recursion_element(range(self.size), self._shape, idxs)
        value = tensor(value)
        value_size = value.size
        for i, idx in enumerate(idxs):
            self._array[idx] = value._array[i % value_size]

    def __iter__(self):
        self._iter_n = 0
        return self

    def __next__(self):
        if not hasattr(self, '_iter_n'):
            raise TypeError("'Tensor' object is not an iterator")
        if self._iter_n < len(self):
            result = self[self._iter_n]
            self._iter_n += 1
            return result
        else:
            raise StopIteration

    def tolist(x):
        x = tensor(x)
        if len(x.shape) == 0:
            return x._array[:]
        def recursion_items(array, shape):
            if len(shape) == 1:
                return array
            result = []
            axis_n = shape[0]
            batch_size = len(array) // axis_n
            for i in range(axis_n):
                sub_array = array[i * batch_size: (i + 1) * batch_size]
                result.append(recursion_items(sub_array, shape[1:]))
            return result
        return recursion_items(x._array, x.shape)

    def item(x):
        x = tensor(x)
        if x.size != 1:
            raise ValueError('can only convert an array of size 1 to a Python scalar')
        return x._array[0]

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def _sample_scalar(self):
        return self._array[0]

    def _sample_vector(self, array, shape, size):
        result = []
        if shape[0] >= size:
            array = array[:5] + [InterChars('...')] + array[-5:]
        for item in array:
            if isinstance(item, float):
                item = '{:g}'.format(item)
            elif isinstance(item, complex):
                item = '{:g}'.format(item)
            result.append(item)
        return result

    def _sample_matrix(self, array, shape, row_size, col_size):
        row_idx = range(shape[0]) if shape[0] < row_size else (list(range(5)) + list(range(shape[0] - 5, shape[0])))
        result = []
        items_size = [0] * shape[-1]
        for i in row_idx:
            sub_array = array[i * shape[-1]: (i + 1) * shape[-1]]
            row = self._sample_vector(sub_array, shape[1:], col_size)
            items_size = [max(item_size, len(str(item))) for item_size, item in zip(items_size, row)]
            result.append(row)
        if shape[0] >= row_size:
            result.insert(5, InterChars('... ...'))
        return result, items_size

    def _sample_tensor(self, x, size, indent=0):
        result = ''
        if len(x.shape) == 2:
            samples, items_size = self._sample_matrix(x._array, x.shape, row_size=256, col_size=16)
            for row in samples:
                if not isinstance(row, InterChars):
                    result += '['
                    result += ' '.join(str(item).rjust(item_size) for item_size, item in zip(items_size, row))
                    result += ']'
                else:
                    result += ' ' + str(row)
                result += '\n'
        else:
            idx = range(x.shape[0]) if x.shape[0] < size else (list(range(5)) + list(range(x.shape[0] - 5, x.shape[0])))
            for i in idx:
                result += '['
                result += str(self._sample_tensor(x[i], size, indent=indent+1))
                result += ']\n'
                if x.shape[0] >= size and i == 4:
                    result += '... ... ...\n'
        return result.strip()

    def __str__(self):
        shape = self.shape
        array = self._array
        if len(shape) == 0:
            return str(self._sample_scalar())
        if len(shape) == 1:
            line = self._sample_vector(array, shape, size=256)
            return '[{}]'.format(', '.join(str(item) for item in line))
        result_lines = '[{}]'.format(self._sample_tensor(self, 16)).split('\n')
        for i, line in enumerate(result_lines):
            indent = len(shape) - line.count('[')
            result_lines[i] = ' ' * indent + line + '\n' * (min(1, line.count(']') - 1))
        result = '\n'.join(result_lines).strip()
        return result

    def __repr__(self):
        if len(self.shape) == 0:
            return str(self)
        elif len(self.shape) == 1:
            return 'vector({}, shape: {})'.format(self, self.shape)
        elif len(self.shape) == 2:
            return 'matrix(\n{}, shape: {})'.format(self, self.shape)
        else:
            return 'tensor(\n{}, shape: {})'.format(self, self.shape)

    def map(x, func):
        x = tensor(x)
        array = [0] * x.size
        shape = x.shape
        for i, value in enumerate(x._array):
            array[i] = func(value)
        if len(shape) == 0: return array[0]
        return Tensor(array, shape)

    def map2(x1, x2, func):
        x1 = tensor(x1)
        x2 = tensor(x2)

        shape1 = x1.shape
        m = x1.size
        shape2 = x2.shape
        n = x2.size
        if m == n and len(shape1) == len(shape2) and shape1 != shape2:
            raise ValueError(
                    'operands could not be broadcast together with shapes {} {}'.format(shape1, shape2))
        l = max(m, n)
        array1 = x1._array
        array2 = x2._array
        array = [0] * l
        shape = shape1 if m > n or (m == n and len(shape1) > len(shape2)) else shape2
        k1 = 0
        k2 = 0
        for i in range(l):
            array[i] = func(array1[k1], array2[k2])
            k1 = (k1 + 1) % m
            k2 = (k2 + 1) % n
        if len(shape) == 0: return array[0]
        return Tensor(array, shape)

    def map_outer(x1, x2, func):
        x1 = tensor(x1)
        x2 = tensor(x2)
        array1 = x1._array
        array2 = x2._array
        m, n = len(array1), len(array2)
        size = m * n
        array = [0] * size
        for i in range(size):
            r = i // m
            c = i % m
            array[r * m + c] = func(array1[r], array2[c])
        return Tensor(array, shape=(m, n))

    def matmul(x1, x2):
        x1 = tensor(x1)
        x2 = tensor(x2)
        if len(x1.shape) == 0 or len(x2.shape) == 0:
            raise ValueError("Scalar operands are not allowed, use '*' instead")

        shape1 = x1.shape if len(x1.shape) != 1 else (1, x1.shape[0])
        shape2 = x2.shape if len(x2.shape) != 1 else (x2.shape[0], 1)
        m1, n1 = shape1[-2:]
        m2, n2 = shape2[-2:]
        if n1 != m2:
            raise ValueError(
                    'shapes {} and {} not aligned: {} (dim {}) != {} (dim {})'.format(
                        shape1, shape2, n1, len(shape1) - 1, m2, len(shape2) - 2))
        cell_size1 = m1 * n1
        cell_size2 = m2 * n2
        batch_size1 = x1.size // cell_size1
        batch_size2 = x2.size // cell_size2
        if batch_size1 == batch_size2 and len(shape1) == len(shape2) and shape1[:-2] != shape2[:-2]:
            raise ValueError(
                    'matmul could not be broadcast together with shapes {} {}'.format(shape1, shape2))

        cell_size = m1 * n2
        batch_size = max(batch_size1, batch_size2)
        shape = shape1 if batch_size1 > batch_size2 or (batch_size1 == batch_size2 and len(shape1) > len(shape2)) else shape2
        shape = shape[:-2] + ((m1, n2) if len(x1.shape) != 1 or len(x2.shape) != 1 else tuple())
        l = batch_size * cell_size
        array1 = x1._array
        array2 = x2._array
        array = [0] * l
        for idx in range(l):
            s = idx // cell_size
            s1 = s % batch_size1
            s2 = s % batch_size2
            i = idx % cell_size // n2
            j = idx % n2
            value = 0
            for num in range(n1): # n1 == m2
                k1 = s1 * cell_size1 + i * n1 + num
                k2 = s2 * cell_size2 + num * n2 + j
                value += array1[k1] * array2[k2]
            array[idx] = value
        return Tensor(array, shape)

    __inv__ = lambda x: Tensor.map(x, operator.inv)

    __pos__ = lambda x: Tensor.map(x, operator.pos)
    __neg__ = lambda x: Tensor.map(x, operator.neg)
    __abs__ = lambda x: Tensor.map(x, operator.abs)

    __lt__ = lambda x1, x2: Tensor.map2(x1, x2, operator.lt)
    __le__ = lambda x1, x2: Tensor.map2(x1, x2, operator.le)
    __eq__ = lambda x1, x2: Tensor.map2(x1, x2, operator.eq)
    __ge__ = lambda x1, x2: Tensor.map2(x1, x2, operator.ge)
    __gt__ = lambda x1, x2: Tensor.map2(x1, x2, operator.gt)

    __and__ = lambda x1, x2: Tensor.map2(x1, x2, operator.and_)
    __rand__ = lambda x1, x2: Tensor.map2(x2, x1, operator.and_)

    __or__ = lambda x1, x2: Tensor.map2(x1, x2, operator.or_)
    __ror__ = lambda x1, x2: Tensor.map2(x2, x1, operator.or_)

    __xor__ = lambda x1, x2: Tensor.map2(x1, x2, operator.xor)
    __rxor__ = lambda x1, x2: Tensor.map2(x2, x1, operator.xor)

    __lshift__ = lambda x1, x2: Tensor.map2(x1, x2, operator.lshift)
    __rshift__ = lambda x1, x2: Tensor.map2(x1, x2, operator.rshift)

    __lshift__ = lambda x1, x2: Tensor.map2(x1, x2, operator.lshift)

    __add__ = lambda x1, x2: Tensor.map2(x1, x2, operator.add)
    __radd__ = lambda x1, x2: Tensor.map2(x2, x1, operator.add)

    __sub__ = lambda x1, x2: Tensor.map2(x1, x2, operator.sub)
    __rsub__ = lambda x1, x2: Tensor.map2(x2, x1, operator.sub)

    __mul__ = lambda x1, x2: Tensor.map2(x1, x2, operator.mul)
    __rmul__ = lambda x1, x2: Tensor.map2(x2, x1, operator.mul)

    __truediv__ = lambda x1, x2: Tensor.map2(x1, x2, operator.truediv)
    __rtruediv__ = lambda x1, x2: Tensor.map2(x2, x1, operator.truediv)

    __floordiv__ = lambda x1, x2: Tensor.map2(x1, x2, operator.floordiv)
    __rfloordiv__ = lambda x1, x2: Tensor.map2(x2, x1, operator.floordiv)

    __pow__ = lambda x1, x2: Tensor.map2(x1, x2, operator.pow)
    __rpow__ = lambda x1, x2: Tensor.map2(x2, x1, operator.pow)

    __mod__ = lambda x1, x2: Tensor.map2(x1, x2, operator.mod)
    __rmod__ = lambda x1, x2: Tensor.map2(x2, x1, operator.mod)

    __matmul__ = lambda x1, x2: Tensor.matmul(x1, x2)
    __rmatmul__ = lambda x1, x2: Tensor.matmul(x2, x1)

    __iand__ = lambda x1, x2: Tensor.map2(x1, x2, operator.iand)
    __ior__ = lambda x1, x2: Tensor.map2(x1, x2, operator.ior)
    __ixor__ = lambda x1, x2: Tensor.map2(x1, x2, operator.ixor)
    __ilshift__ = lambda x1, x2: Tensor.map2(x1, x2, operator.ilshift)
    __irshift__ = lambda x1, x2: Tensor.map2(x1, x2, operator.irshift)
    __iadd__ = lambda x1, x2: Tensor.map2(x1, x2, operator.iadd)
    __isub__ = lambda x1, x2: Tensor.map2(x1, x2, operator.isub)
    __imul__ = lambda x1, x2: Tensor.map2(x1, x2, operator.imul)
    __itruediv__ = lambda x1, x2: Tensor.map2(x1, x2, operator.itruediv)
    __ifloordiv__ = lambda x1, x2: Tensor.map2(x1, x2, operator.ifloordiv)
    __ipow__ = lambda x1, x2: Tensor.map2(x1, x2, operator.ipow)
    __imod__ = lambda x1, x2: Tensor.map2(x1, x2, operator.imod)
    __imatmul__ = lambda x1, x2: Tensor.matmul(x1, x2)

    def __contains__(self, x):
        x = tensor(x)
        size1 = self.size
        size2 = x.size
        if size1 < size2 or size1 % size2 != 0:
            return False
        array1 = self._array
        array2 = x._array
        steps = size1 // size2
        for step in range(steps):
            if array1[step * size2: (step + 1) * size2] == array2:
                return True
        return False

    def reduce(x, func, axis=None, keepdims=False):
        x = tensor(x)
        dst_shape = None
        dst_array = None
        if axis is None:
            dst_value = x._array[0]
            array = x._array[1:]
            for value in array:
                dst_value = func(dst_value, value)
            if not keepdims:
                return dst_value
            dst_array = [dst_value]
            dst_shape = [1 for axes in x.shape]
            dst_shape = [axes for axes in dst_shape if axes > 1]
            return Tensor(dst_array, dst_shape)
        if isinstance(axis, int):
            axis = (axis, )
        if isinstance(axis, Iterable) and all(map(lambda x: isinstance(x, int), axis)):
            array = x._array
            shape = x.shape

            dst_shape = list(shape)
            for i in axis:
                dst_shape[i] = 1
            null = Null()
            dst_array = [null] * reduce(operator.mul, dst_shape)

            bases = [reduce(operator.mul, shape[i + 1:]) for i in range(len(shape) - 1)] + [1]
            dst_bases = [reduce(operator.mul, dst_shape[i + 1:]) for i in range(len(dst_shape) - 1)] + [1]

            trans_bases = tuple(zip(bases, dst_bases, dst_shape))
            for i, value in enumerate(array):
                dst_idx = 0
                for base, dst_base, axes in trans_bases:
                    dst_idx += i // base % axes * dst_base
                    i %= base
                dst_value = dst_array[dst_idx]
                dst_array[dst_idx] = func(dst_value, value)
            if not keepdims:
                dst_shape = [axes for axes in dst_shape if axes > 1]
            return Tensor(dst_array, dst_shape)
        raise TypeError('tuple indices must be integers, not {}'.format(axis.__class__))

    def sum(x, axis=None, keepdims=False):
        return Tensor.reduce(x, operator.add, axis=axis, keepdims=keepdims)

    def mean(x, axis=None, keepdims=False):
        s = Tensor.sum(x, axis=axis, keepdims=keepdims)
        n = x.size
        if isinstance(s, Tensor):
            n /= s.size
        return s / n

    def std(x, axis=None, keepdims=False):
        sum_square = Tensor.sum(x**2, axis=axis, keepdims=keepdims)
        n = x.size
        if isinstance(sum_square, Tensor):
            n /= sum_square.size
        square_mean = Tensor.mean(x, axis=axis, keepdims=keepdims)**2
        return (sum_square / n - square_mean)**0.5

    def min(x, axis=None, keepdims=False):
        func = lambda a, b: a if a < b else b
        return Tensor.reduce(x, func, axis=axis, keepdims=keepdims)

    def max(x, axis=None, keepdims=False):
        func = lambda a, b: a if a > b else b
        return Tensor.reduce(x, func, axis=axis, keepdims=keepdims)

    def clip(x, x_min, x_max):
        func = None
        if (x_min is not None) and (x_max is None):
            func = lambda x: max(x_min, x)
        elif (x_min is None) and (x_max is not None):
            func = lambda x: min(x, x_max)
        elif (x_min is not None) and (x_max is not None):
            x_max = max(x_min, x_max)
            func = lambda x: min(max(x_min, x), x_max)
        else:
            raise ValueError('tensor_clip: must set either max or min')
        x = Tensor.map(x, func)
        return x

    def argmax(x, axis=0):
        x = tensor(x)
        if not isinstance(axis, int):
            raise TypeError('{} object cannot be interpreted as an integer'.format(axis.__class__))
        reduce_size = x.shape[axis]
        def func(a, b):
            if isinstance(a, Null):
                return [1, 0, b] # counter, index, value
            else:
                counter = a[0] + 1
                a[0] = counter
                if b > a[2]:
                    a[1] = counter - 1
                    a[2] = b
                if counter < reduce_size:
                    return a
                return a[1]
        return Tensor.reduce(x, func, axis=axis)

    def argmin(x, axis=0):
        x = tensor(x)
        if not isinstance(axis, int):
            raise TypeError('{} object cannot be interpreted as an integer'.format(axis.__class__))
        reduce_size = x.shape[axis]
        def func(a, b):
            if isinstance(a, Null):
                return [1, 0, b] # counter, index, value
            else:
                counter = a[0] + 1
                a[0] = counter
                if b < a[2]:
                    a[1] = counter - 1
                    a[2] = b
                if counter < reduce_size:
                    return a
                return a[1]
        return Tensor.reduce(x, func, axis=axis)

def value_shape(value):
    if not isinstance(value, Iterable)  or isinstance(value, str):
        return tuple()
    shape = (len(value), )
    sub_shape = value_shape(value[0])
    return shape + sub_shape

def value_flatten(value, shape):
    if len(shape) == 0:
        return [value]
    for i in range(len(shape) - 1):
        value = reduce(operator.iconcat, value, [])
    return value

def tensor(value):
    if isinstance(value, Tensor):
        shape = value.shape
        array = list(value._array)
    else:
        shape = value_shape(value)
        array = value_flatten(value, shape)
    return Tensor(array, shape)

def zeros(shape):
    array = [0] * reduce(operator.mul, shape)
    return Tensor(array, shape)

def zeros_like(x):
    return zeros(x.shape)

def ones(shape):
    array = [1] * reduce(operator.mul, shape)
    return Tensor(array, shape)

def ones_like(x):
    return ones(x.shape)

def fill(value, shape):
    array = [value] * reduce(operator.mul, shape)
    return Tensor(array, shape)

def arange(start, stop=None, step=1):
    if stop is None:
        return tensor(range(0, start, step))
    return tensor(range(start, stop, step))

def randrange(start, stop=None, step=1, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.randrange(start, stop, step)
    return Tensor(array, shape or tuple())

def randint(a, b, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.randint(a, b)
    return Tensor(array, shape or tuple())

def choice(seq, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.choice(seq)
    return Tensor(array, shape or tuple())

def random(shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.random()
    return Tensor(array, shape or tuple())

def uniform(a, b, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.uniform(a, b)
    return Tensor(array, shape or tuple())

def triangular(low, high, mode, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.triangular(low, high, mode)
    return Tensor(array, shape or tuple())

def betavariate(alpha, beta, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.betavariate(alpha, beta)
    return Tensor(array, shape or tuple())

def expovariate(lambd, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.expovariate(lambd)
    return Tensor(array, shape or tuple())

def gammavariate(alpha, beta, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.gammavariate(alpha, beta)
    return Tensor(array, shape or tuple())

def gauss(mu, sigma, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.gauss(mu, sigma)
    return Tensor(array, shape or tuple())

def lognormvariate(mu, sigma, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.lognormvariate(mu, sigma)
    return Tensor(array, shape or tuple())

def normalvariate(mu, sigma, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.normalvariate(mu, sigma)
    return Tensor(array, shape or tuple())

def vonmisesvariate(mu, kappa, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.vonmisesvariate(mu, kappa)
    return Tensor(array, shape or tuple())

def paretovariate(alpha, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.paretovariate(alpha)
    return Tensor(array, shape or tuple())

def weibullvariate(alpha, beta, shape=None):
    size = reduce(operator.mul, shape) if shape else 1
    array = [0] * size
    for i in range(size):
        array[i] = _random.weibullvariate(alpha, beta)
    return Tensor(array, shape or tuple())

def concatenate(xs, axis=0):
    dims = 0
    first_shape = None
    after_shape = None
    for i, x in enumerate(xs):
        xs[i] = tensor(x)
        if i == 0:
            dims = len(x.shape)
            first_shape = x.shape[:axis]
            after_shape = x.shape[len(first_shape) + 1:]
        else:
            if (dims != len(x.shape) or
                first_shape != x.shape[:axis] or
                after_shape != x.shape[len(first_shape) + 1:]):
                raise ValueError(
                        'all the input array dimensions except for the concatenation axis must match exactly')
    array = []
    steps = reduce(operator.mul, first_shape + (1, ))
    width = reduce(operator.mul, after_shape + (1, ))
    for step in range(steps):
        for x in xs:
            batch_size = x.shape[axis] * width
            array.extend(x._array[step * batch_size: (step + 1) * batch_size])
    shape = first_shape + (sum([x.shape[axis] for x in xs]), ) + after_shape
    return Tensor(array, shape)

def vstack(xs):
    return concatenate(xs, axis=0)

def hstack(xs):
    return concatenate(xs, axis=-1)
