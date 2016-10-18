import numpy as np
# import matplotlib.pyplot as plt


def get_sample():
    x = np.random.rand(2)*2-1
    y = np.sin(np.pi*x)
    return x, y


def get_a(x, y):
    return (x[0]*y[0]+x[1]*y[1])/(x[0]**2+x[1]**2)


def compute_a_dash(n=1000):
    sum_a = 0

    # computing a dash
    for n in xrange(n):
        x, y = get_sample()
        temp_a = get_a(x, y)
        sum_a += temp_a
        assert(temp_a/np.pi <= 1.0)
        assert(temp_a > 0)
    return sum_a / float(n)


def compute_var(a_dash, n=100000):
    vec_diff_a = []
    for n in xrange(n):
        x, y = get_sample()
        temp_a = get_a(x, y)
        assert(temp_a/np.pi <= 1.0)
        assert(temp_a > 0)
        diff = temp_a*x - a_dash*x
        e_d_diff_sq = (diff**2).mean()
        vec_diff_a.append(e_d_diff_sq)
    vec_diff_a = np.array(vec_diff_a)
    return vec_diff_a.mean()


def compute_bias(a=compute_a_dash(), n=1000000):
    x = np.random.rand(n)*2-1
    y = np.sin(np.pi*x)
    g = x*a
    return ((y - g)**2).mean()


if __name__ == "__main__":
    a = compute_a_dash(100000)
    print "a = ", a

    bias = compute_bias(a)
    print "bias = ", bias

    print "var = ", compute_var(a)
