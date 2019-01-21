import random as _random
import itertools as _itertools
import bisect as _bisect
import math as _math
import cmath as _cmath
import mpmath as _mpmath
import builtins as _builtins

from idscience import Tensor, tensor, zeros_like, vstack


def ceil(x):
    return Tensor.operator_tensor(x, _math.ceil)

def copysign(x, y):
    return Tensor.operator_tensor2tensor(x, y, _math.copysign)

def fabs(x):
    return Tensor.operator_tensor(x, _math.fabs)

def factorial(x):
    return Tensor.operator_tensor(x, _math.factorial)

def floor(x):
    return Tensor.operator_tensor(x, _math.floor)

def fmod(x, y):
    return Tensor.operator_tensor2tensor(x, y, _math.fmod)

def frexp(x):
    return Tensor.operator_tensor(x, _math.frexp)

def gcd(a, b):
    return Tensor.operator_tensor2tensor(a, b, _math.gcd)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    _isclose = lambda a, b: _math.isclose(a, b, rel_tol=1e-09, abs_tol=0.0)
    return Tensor.operator_tensor2tensor(a, b, _isclose)

def isfinite(x):
    return Tensor.operator_tensor(x, _math.isfinite)

def isinf(x):
    return Tensor.operator_tensor(x, _math.isinf)

def isnan(x):
    return Tensor.operator_tensor(x, _math.isnan)

def ldexp(x, i):
    return Tensor.operator_tensor2tensor(x, i, _math.ldexp)

def modf(x):
    return Tensor.operator_tensor(x, _math.modf)

def remainder(x, y):
    return Tensor.operator_tensor2tensor(x, y)

def trunc(x):
    return Tensor.operator_tensor(x, _math.trunc)

def exp(x):
    return Tensor.operator_tensor(x, _math.exp)

def expm1(x):
    return Tensor.operator_tensor(x, _math.expm1)

def log(x, base):
    return Tensor.operator_tensor2tensor(x, base, _math.log)

def log1p(x):
    return Tensor.operator_tensor(x, _math.log1p)

def log2(x):
    return Tensor.operator_tensor(x, _math.log2)

def log10(x):
    return Tensor.operator_tensor(x, _math.log10)

def pow(x, y):
    return Tensor.operator_tensor2tensor(x, y, _math.pow)

def sqrt(x):
    return Tensor.operator_tensor(x, _math.sqrt)

def acos(x):
    return Tensor.operator_tensor(x, _math.acos)

def asin(x):
    return Tensor.operator_tensor(x, _math.asin)

def atan(x):
    return Tensor.operator_tensor(x, _math.atan)

def atan2(x, y):
    return Tensor.operator_tensor2tensor(x, y, _math.atan2)

def cos(x):
    return Tensor.operator_tensor(x, _math.cos)

def hypot(x, y):
    return Tensor.operator_tensor2tensor(x, y, _math.hypot)

def sin(x):
    return Tensor.operator_tensor(x, _math.sin)

def tan(x):
    return Tensor.operator_tensor(x, _math.tan)

def degrees(x):
    return Tensor.operator_tensor(x, _math.degrees)

def radians(x):
    return Tensor.operator_tensor(x, _math.radians)

def acosh(x):
    return Tensor.operator_tensor(x, _math.acosh)

def asinh(x):
    return Tensor.operator_tensor(x, _math.asinh)

def atanh(x):
    return Tensor.operator_tensor(x, _math.atanh)

def cosh(x):
    return Tensor.operator_tensor(x, _math.cosh)

def sinh(x):
    return Tensor.operator_tensor(x, _math.sinh)

def tanh(x):
    return Tensor.operator_tensor(x, _math.tanh)

def erf(x):
    return Tensor.operator_tensor(x, _math.erf)

def erfc(x):
    return Tensor.operator_tensor(x, _math.erfc)

def gamma(x):
    return Tensor.operator_tensor(x, _math.gamma)

def lgamma(x):
    return Tensor.operator_tensor(x, _math.lgamma)

pi = _math.pi
e = _math.e
tau = pi * 2
inf = _math.inf
nan = _math.nan

def round(x, ndigits=None):
    return Tensor.operator_tensor2tensor(x, ndigits, _builtins.round)

def sample(x, weights=None, *, cum_weights=None, size=1):
    random = _random.random
    if cum_weights is None:
        if weights is None:
            _int = int
            total = len(x)
            return [x[_int(random() * total)] for i in range(size)]
        cum_weights = list(_itertools.accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != len(x):
        raise ValueError('The number of weights does not match the x')
    bisect = _bisect.bisect
    total = cum_weights[-1]
    return [x[bisect(cum_weights, random() * total)] for i in range(size)]

def t_test(x, y=None, mu=None):
    if not isinstance(x, Tensor):
        x = tensor(x)
    if y is not None and not isinstance(y, Tensor):
        y = tensor(y)

    if y is None and mu is None:
        raise TypeError('Count of arguments must be 2, but only 1')
    t = 0
    df = 0
    if y is not None:
        n1 = len(x)
        n2 = len(y)
        if n1 <= 1 or n2 <= 1:
            raise ValueError

        mean1 = Tensor.mean(x)
        mean2 = Tensor.mean(y)

        var1 = Tensor.sum((x - mean1)**2) / (n1 - 1)
        var2 = Tensor.sum((y - mean2)**2) / (n2 - 1)

        t =  (mean1 - mean2) / _math.sqrt(var1 / n1 + var2 / n2)
        df = (var1 / n1 + var2 / n2)**2 / (var1**2 / (n1**2 * (n1 - 1)) + var2**2 / (n2**2 * (n2 - 1)))
    if mu is not None:
        n = len(x)
        mean = Tensor.mean(x)
        sd = (Tensor.sum((x - mean)**2) / (n - 1))**0.5

        t = (mean - mu) / (sd / _math.sqrt(n))
        df = n - 1
    p = float(_mpmath.betainc(a=df / 2, b=1 / 2, x2=df / (t**2 + df), regularized=True))
    return t, df, p

def chisq_test(x, correct=True):
    def chi2_uniform_distance(table):
        expected = Tensor.sum(table) / len(table)
        cntrd = table - expected
        return Tensor.sum(cntrd**2) / expected

    def chi2_uniform2d_distance(table, correct):
        row_sums = Tensor.sum(table, axis=1)
        col_sums = Tensor.sum(table, axis=0)
        all_sums = Tensor.sum(row_sums)
        expected = zeros_like(table)
        _correct = False
        if (table.shape[0] - 1) * (table.shape[1] - 1) == 1:
            _correct = True
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                expected_ij = row_sums[i] * col_sums[j] / all_sums
                expected[i, j] = expected_ij
                if expected_ij < 10:
                    _correct = correct and True
        if _correct:
            return Tensor.sum((abs(table - expected) - 0.5)**2 / expected)
        return Tensor.sum((table - expected)**2 / expected)

    def chi2_probability(df, distance):
        return 1 - _mpmath.gammainc(0.5 * df, a=0, b=0.5 * distance, regularized=True)

    if not isinstance(x, Tensor):
        x = tensor(x)
    if len(x.shape) == 1:
        distance = chi2_uniform_distance(x)
        df = len(x) - 1
        prob = float(chi2_probability(df, distance))
        return distance, df, prob
    elif len(x.shape) == 2:
        distance = chi2_uniform2d_distance(x, correct)
        df = (x.shape[0] - 1) * (x.shape[1] - 1)
        prob = float(chi2_probability(df, distance))
        return distance, df, prob
    raise ValueError('Only support vector or matrix')

def mpmatrix2tensor(m):
    rows, cols = m.rows, m.cols
    size = rows * cols
    array = [0] * size
    for i in range(size):
        r = i // cols
        c = i % cols
        array[i] = float(m[r, c])
    return Tensor(array, shape=(rows, cols))

def matrix_inv(x):
    def _matrix_inv(x):
        m = _mpmath.matrix(x)
        u, s, v = _mpmath.svd_r(m)
        u = mpmatrix2tensor(u)
        s = mpmatrix2tensor(s)
        v = mpmatrix2tensor(v)
        s_inv = Tensor.operator_tensor(s, lambda x: 1 / x if x > 1e-7 else x)
        x_inv = v.T * s_inv @ u.T
        return x_inv

    if not isinstance(x, Tensor):
        x = tensor(x)
    if len(x.shape) <= 2:
        x_inv = _matrix_inv(x)
    else:
        src_shape = x.shape
        dst_shape = src_shape[:-2] + src_shape[-2:][::-1]
        ms = x.reshape(-1, src_shape[-2], src_shape[-1])
        for i, m in enumerate(ms):
            ms[i] = _matrix_inv(m)
        x_inv = ms.reshape(*dst_shape)
    return x_inv

def fit_linear(X, y):
    if not isinstance(X, Tensor):
        X = tensor(X)
    if not isinstance(y, Tensor):
        y = tensor(y)
    X_T = X.T
    Beta = matrix_inv(X_T @ X) @ X_T @ y
    return Beta

def fft(x):
    if _math.log2(len(x)) % 1 > 0:
            raise ValueError("size of x must be a power of 2")
    def _fft(x):
        if not isinstance(x, Tensor):
            x = tensor(x)
        N = x.size
        if N < 2: return x
        even = _fft(x[0::2])
        odd =  _fft(x[1::2])
        T = [0] * (N // 2)
        for i in range(N // 2):
            T[i] = _cmath.exp(-2j * _cmath.pi * i / N) * odd[i]
        return vstack([even + T, even - T])
    return _fft(x)

def ifft(x):
    if _math.log2(len(x)) % 1 > 0:
            raise ValueError("size of x must be a power of 2")
    def _ifft(x):
        if not isinstance(x, Tensor):
            x = tensor(x)
        N = x.size
        if N < 2: return x
        even = _ifft(x[0::2])
        odd =  _ifft(x[1::2])
        T= [0] * (N // 2)
        for i in range(N // 2):
            T[i] = _cmath.exp(2j * _cmath.pi * i / N) * odd[i]
        return vstack([even + T, even - T])
    by_ifft = _ifft(x)
    N = len(x)
    return by_ifft / N
