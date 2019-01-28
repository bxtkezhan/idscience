import random as _random
import itertools as _itertools
import bisect as _bisect
import math as _math
import cmath as _cmath
import builtins as _builtins

from idscience import Tensor, tensor, zeros_like, vstack, random, concatenate


def ceil(x):
    return Tensor.map_tensor(x, _math.ceil)

def copysign(x, y):
    return Tensor.map_tensor2tensor(x, y, _math.copysign)

def fabs(x):
    return Tensor.map_tensor(x, _math.fabs)

def factorial(x):
    return Tensor.map_tensor(x, _math.factorial)

def floor(x):
    return Tensor.map_tensor(x, _math.floor)

def fmod(x, y):
    return Tensor.map_tensor2tensor(x, y, _math.fmod)

def frexp(x):
    return Tensor.map_tensor(x, _math.frexp)

def gcd(a, b):
    return Tensor.map_tensor2tensor(a, b, _math.gcd)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    _isclose = lambda a, b: _math.isclose(a, b, rel_tol=1e-09, abs_tol=0.0)
    return Tensor.map_tensor2tensor(a, b, _isclose)

def isfinite(x):
    return Tensor.map_tensor(x, _math.isfinite)

def isinf(x):
    return Tensor.map_tensor(x, _math.isinf)

def isnan(x):
    return Tensor.map_tensor(x, _math.isnan)

def ldexp(x, i):
    return Tensor.map_tensor2tensor(x, i, _math.ldexp)

def modf(x):
    return Tensor.map_tensor(x, _math.modf)

def remainder(x, y):
    return Tensor.map_tensor2tensor(x, y)

def trunc(x):
    return Tensor.map_tensor(x, _math.trunc)

def exp(x):
    return Tensor.map_tensor(x, _math.exp)

def expm1(x):
    return Tensor.map_tensor(x, _math.expm1)

def log(x, base):
    return Tensor.map_tensor2tensor(x, base, _math.log)

def log1p(x):
    return Tensor.map_tensor(x, _math.log1p)

def log2(x):
    return Tensor.map_tensor(x, _math.log2)

def log10(x):
    return Tensor.map_tensor(x, _math.log10)

def pow(x, y):
    return Tensor.map_tensor2tensor(x, y, _math.pow)

def square(x):
    return pow(x, 2)

def sqrt(x):
    return Tensor.map_tensor(x, _math.sqrt)

def acos(x):
    return Tensor.map_tensor(x, _math.acos)

def asin(x):
    return Tensor.map_tensor(x, _math.asin)

def atan(x):
    return Tensor.map_tensor(x, _math.atan)

def atan2(x, y):
    return Tensor.map_tensor2tensor(x, y, _math.atan2)

def cos(x):
    return Tensor.map_tensor(x, _math.cos)

def hypot(x, y):
    return Tensor.map_tensor2tensor(x, y, _math.hypot)

def sin(x):
    return Tensor.map_tensor(x, _math.sin)

def tan(x):
    return Tensor.map_tensor(x, _math.tan)

def degrees(x):
    return Tensor.map_tensor(x, _math.degrees)

def radians(x):
    return Tensor.map_tensor(x, _math.radians)

def acosh(x):
    return Tensor.map_tensor(x, _math.acosh)

def asinh(x):
    return Tensor.map_tensor(x, _math.asinh)

def atanh(x):
    return Tensor.map_tensor(x, _math.atanh)

def cosh(x):
    return Tensor.map_tensor(x, _math.cosh)

def sinh(x):
    return Tensor.map_tensor(x, _math.sinh)

def tanh(x):
    return Tensor.map_tensor(x, _math.tanh)

def erf(x):
    return Tensor.map_tensor(x, _math.erf)

def erfc(x):
    return Tensor.map_tensor(x, _math.erfc)

def gamma(x):
    return Tensor.map_tensor(x, _math.gamma)

def lgamma(x):
    return Tensor.map_tensor(x, _math.lgamma)

pi = _math.pi
e = _math.e
tau = pi * 2
inf = _math.inf
nan = _math.nan

def round(x, ndigits=None):
    return Tensor.map_tensor2tensor(x, ndigits, _builtins.round)

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

def betainc(z, a, b, regularized=False, num_simulations=100, eps=1e-7):
    if _math.isclose(z, 0): return 0.
    if _math.isclose(z, 1): return 1.
    if _math.isclose(b, 1): return z**a
    if _math.isclose(a, 1): return 1 - (1 - x)**b

    C = z**a
    k = 1
    ni = 1
    psb = 1
    z_k = 1
    bei = 1 / a
    error = 1
    while k < num_simulations and error > eps:
        ni *= k
        psb *= (1 - b) + k - 1
        z_k *= z
        num = psb * z_k
        den = ni * (a + k)
        bei_ = bei
        bei += num / den
        error = abs(bei - bei_)
        k += 1

    if not regularized:
        return C * bei
    beta_ab = _math.gamma(a) * _math.gamma(b) / _math.gamma(a + b)
    return C * bei / beta_ab

def t_test(x, y=None, mu=None):
    x = tensor(x)
    if y is not None:
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
    p = betainc(df / (t**2 + df), df / 2, 1 / 2, regularized=True)
    return t, df, p

def gammainc(s, z, regularized=False, num_simulations=100, eps=1e-7):
    C = z**s * _math.exp(-z)
    if _math.isclose(C, 0): return 0.
    k = 0
    num = 1 / z
    den = 1
    gai = 0
    error = 1
    while k < num_simulations and error > eps:
        num *= z
        den *= s + k
        gai_ = gai
        gai += num / den
        error = abs(gai - gai_)
        k += 1
    if not regularized:
        return C * gai
    return C * gai / _math.gamma(s)

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
        return 1 - gammainc(0.5 * df, 0.5 * distance, regularized=True)

    x = tensor(x)
    if len(x.shape) == 1:
        distance = chi2_uniform_distance(x)
        df = len(x) - 1
        prob = chi2_probability(df, distance)
        return distance, df, prob
    elif len(x.shape) == 2:
        distance = chi2_uniform2d_distance(x, correct)
        df = (x.shape[0] - 1) * (x.shape[1] - 1)
        prob = chi2_probability(df, distance)
        return distance, df, prob
    raise ValueError('Only support vector or matrix')

def matrix_inv(A, eps=1e-7):
    Q, R = qr(A)
    I = eye(*R.shape)
    A = concatenate([R, I], axis=-1)
    m, n = A.shape
    for i in range(m):
        a = A[i, i]
        if abs(a) > eps:
            A[i] /= a
    for r in range(m):
        for c in range(r + 1, m):
            A[r] -= A[r, c] * A[c]
    invR = A[:, m:]
    return invR @ Q.T

def fit_linear(X, y):
    X = tensor(X)
    y = tensor(y)
    X_T = X.T
    Beta = matrix_inv(X_T @ X) @ X_T @ y
    return Beta

def fft(x):
    if _math.log2(len(x)) % 1 > 0:
            raise ValueError("size of x must be a power of 2")
    def _fft(x):
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

def norm(x):
    x = tensor(x)
    return sqrt((Tensor.sum(square(x))))

def eye(m, n=None):
    n = n or m
    array = [0] * m * n
    for i in range(m):
        if i >= n: break
        array[i * n + i] = 1
    return Tensor(array, shape=(m, n))

def make_householder(a):
    a = tensor(a)
    v = a / (a[0] + copysign(norm(a), a[0]))
    v[0] = 1
    H = eye(a.shape[0])
    H -= (2 / (v @ v)) * (v.reshape(-1, 1) @ v.reshape(1, -1))
    return H

def qr(A):
    A = tensor(A)
    m, n = A.shape
    Q = eye(m)
    for i in range(n - (m == n)):
        H = eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = Q @ H
        A = H @ A
    return Q, A

def power_iteration(A, num_simulations=100, eps=1e-7):
    A = tensor(A)
    b_k = random((A.shape[1], ))
    k = 0
    error = 1
    while k < num_simulations and error > eps:
        b_k1 = A @ b_k
        b_k1_norm = norm(b_k1)
        b_k_ = b_k
        b_k = b_k1 / b_k1_norm
        k += 1
        error = norm(b_k - b_k_)
    return b_k

def outer(x1, x2):
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
        array[r * m + c] = array1[r] * array2[c]
    return Tensor(array, shape=(m, n))
