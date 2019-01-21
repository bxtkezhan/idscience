import idscience as ids
from functools import wraps
import time


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        print('-' * 80)
        return result
    return wrapper

@timethis
def tensr_init():
    s = ids.tensor(7)
    print(s)
    v = ids.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(v)
    m = ids.tensor([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
    print(m)
    t = ids.tensor([
        [[1, 2, 3],
         [1, 2, 3],
         [1, 2, 3]],
        [[2, 3, 4],
         [2, 3, 4],
         [2, 3, 4]],
        [[3, 4, 5],
         [3, 4, 5],
         [3, 4, 5]]])
    print(t)
    tabel = ids.fill(None, (9, 9))
    tabel[0, :] = [
        '#', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
    tabel[1:, 0] = range(8)
    tabel[1:, 1:] = ids.randint(-5, 5, shape=(8, 8))
    print(tabel)

@timethis
def tensor_reshape():
    x1 = ids.zeros((3, 3, 3))
    print(x1)
    x2 = x1.reshape(3, -1)
    print(x2)
    x3 = x2.reshape(-1)
    print(x3)

@timethis
def tensor_transpose():
    t = ids.arange(3 ** 3).reshape(3, 3, 3)
    print(t)
    t = t.T
    print(t)

@timethis
def tensor_get_set_item():
    t = ids.arange(3 ** 3).reshape(3, 3, 3)
    print(t)
    t[0] = ids.ones((3, 3)) * 0
    t[1] = ids.ones((3, 3)) * 1
    t[2] = ids.ones((3, 3)) * 2
    print(t)

@timethis
def tensor_tolist_repr():
    t = ids.arange(3 ** 3).reshape(3, 3, 3).T
    print(t)
    print(t.tolist())

@timethis
def tensor_operator_self():
    t = ids.zeros((3, 3))
    print(t)
    t += 1
    print(t)
    t -= 0.5
    print(t)
    t *= 10
    print(t)
    t /= 5
    print(t)
    t @= t
    print(t)

@timethis
def tensor_operator_t2t():
    t1 = ids.zeros((3, 3)) + 1
    t2 = ids.zeros((3, 3)) + 2
    print(t1)
    print(t2)
    t3 = t1 + t2
    print(t3)
    t3 = t1 - t2
    print(t3)
    t3 = t1 * t2
    print(t3)
    t3 = t1 / t2
    print(t3)
    t3 = t1 @ t2
    print(t3)

@timethis
def tensor_random():
    r = ids.random((10, 10))
    print(r)

@timethis
def tensor_statistics():
    dset1 = ids.arange(3 * 3 * 3).reshape(3, 3, 3)
    print(dset1.sum(), 'vs', sum(dset1._array))
    print(dset1.sum(axis=(1)))
    print('vs')
    print(dset1[:, 0, :] + dset1[:, 1, :] + dset1[:, 2, :])

    dset2 = ids.arange(5 * 5).reshape(5, 5)
    print(dset2)
    print(dset2.mean(axis=0, keepdims=True))
    print(dset2.mean(axis=1, keepdims=True))

    dset3 = ids.random((10, 10))
    print(dset3)
    mean = dset3.mean(axis=0)
    std = (((dset3 - mean)**2).sum(axis=0) / dset3.shape[0])**0.5
    print(std)
    std = dset3.std(axis=0)
    print(std)

@timethis
def functional():
    dset1 = [386.7,396.7,409.8,384.5,394.3,396.2,401.6,392.8,413.5,393.7,
             398.9,404.1,391.3,385.3,411.6,373.5,403.2,395.3,404.4,399.0,
             414.4,383.8,409.8,413.2,395,372.2,399.4,389.5,402.4,397.7]
    print(ids.functional.t_test(dset1, mu=400))
    print(ids.functional.t_test(dset1, ids.tensor(dset1)))

    dset2 = [199809, 200665, 199607, 200270, 199649]
    dset3 = ids.tensor([522573, 244456, 139979,  71531,  21461])
    dset4 = [[8, 6], [5, 7]]
    dset5 = ids.tensor([
        [11, 9, 38, 42],
        [39, 35, 11, 18]])
    print(ids.functional.chisq_test(dset2))
    print(ids.functional.chisq_test(dset3))
    print(ids.functional.chisq_test(dset4))
    print(ids.functional.chisq_test(dset5))

    ms = ids.randint(-5, 5, (3, 3, 4))
    ms_inv = ids.functional.matrix_inv(ms)
    print('ms:')
    print(ms)
    print('ms_inv:')
    print(ms_inv)
    print('ms @ ms_inv:')
    print(ms @ ms_inv)

    X = ids.random((3, 3))
    y = ids.random((3, 1))
    W = ids.functional.fit_linear(X, y)
    print('W(fit):')
    print(W)
    print('y(src):')
    print(y)
    print('y(fit):')
    print(X @ W)

    x = ids.tensor([1., 1., 2., 3., 5., 8., 13., 21.])
    print('x:', x)
    ff = ids.functional.fft(x)
    print('fft(x):', ' '.join('%5.3f' % abs(f) for f in ff))
    iff = ids.functional.ifft(ff)
    print('ifft(fft(x)):', ' '.join('%5.3f' % abs(f) for f in iff))

def utils():
    filename = ids.utils.choose_file()
    print('filename:', filename)
    filename = ids.utils.choose_file(save=True)
    print('filename:', filename)
    directory = ids.utils.choose_dir()
    print('directory:', directory)
    print('-' * 80)


if __name__ == '__main__':
    tensr_init()
    tensor_reshape()
    tensor_transpose()
    tensor_get_set_item()
    tensor_tolist_repr()
    tensor_operator_self()
    tensor_operator_t2t()
    tensor_random()
    tensor_statistics()
    functional()
    # utils()
