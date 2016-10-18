import numpy as np
# import matplotlib.pyplot as plt


def get_sample():
    x = np.random.rand(2)*2-1
    y = np.sin(np.pi*x)
    return x, y


def get_a(x, y):
    px = np.average(x)
    py = np.average(y)
    if np.abs(px) <= np.finfo(float).eps:
        return None
    return py/px


def compute_a_dash(n=1000):
    sum_a = 0

    # computing a dash
    for n in xrange(n):
        x, y = get_sample()
        temp_a = get_a(x, y)
        if temp_a is None:
            print 'wow, it really happend, I am not paranoid!'
            n -= 1
            continue
        sum_a += temp_a
        assert(temp_a/np.pi <= 1.0)
    return sum_a / float(n)


def v_compute_a_dash(n=1000000):
    # same as w/o 'v_' but faster
    x = np.random.rand(n, 2)*2-1
    y = np.sin(np.pi*x)
    avg_x = np.average(x, axis=1)
    avg_y = np.average(y, axis=1)

    sample_a = avg_y/avg_x
    return sample_a.mean()


def v_compute_bias(n=1000000, a=v_compute_a_dash()):
    x = np.random.rand(n)*2-1
    y = np.sin(np.pi*x)
    g = x*a
    return ((y - g)**2).mean()


if __name__ == "__main__":
    a = v_compute_a_dash() # WRONG
    print "a = ", a

    bias = v_compute_bias(a=a) # a is wrong so this is also ...
    print "bias = ", bias
    pass
