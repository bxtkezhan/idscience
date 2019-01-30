from idscience.tensor import Tensor
reshape = Tensor.reshape
transpose = Tensor.transpose
tolist = Tensor.tolist
item = Tensor.item
map = Tensor.map
map2 = Tensor.map2
map_outer = Tensor.map_outer
matmul = Tensor.matmul
reduce = Tensor.reduce
sum = Tensor.sum
mean = Tensor.mean
std = Tensor.std
min = Tensor.min
max = Tensor.max
clip = Tensor.clip
argmax = Tensor.argmax
argmin = Tensor.argmin

from idscience.tensor import tensor, zeros, zeros_like, ones, ones_like, fill, arange
from idscience.tensor import randrange, randint, choice, random, uniform, triangular
from idscience.tensor import (
        betavariate, expovariate, gammavariate, gauss, lognormvariate,
        normalvariate, vonmisesvariate, paretovariate, weibullvariate)
from idscience.tensor import vstack, hstack, concatenate

from idscience import utils

from idscience import functional
from idscience.functional import (
        ceil, copysign, fabs, factorial, floor, fmod, frexp, gcd, isclose,
        isfinite, isinf, isnan, ldexp, modf, remainder, trunc, exp, expm1,
        log, log1p, log2, log10, pow, square, sqrt, acos, asin, atan, atan2, cos,
        hypot, sin, tan, degrees, radians, acosh, asinh, atanh, cosh, sinh,
        tanh, erf, erfc, gamma, lgamma, pi, e, tau, inf, nan, round, eye, outer)
