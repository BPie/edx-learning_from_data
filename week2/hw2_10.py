import numpy as np
import lra


def foo(x, y):
    return x**2 + y**2 - 0.6


def feature_transformer(data):
    x = data[:, 0]
    y = data[:, 1]
    a0 = np.ones([data.shape[0], 1])
    a1 = np.expand_dims(x, axis=0).T
    a2 = np.expand_dims(y, axis=0).T
    a3 = np.expand_dims(x*y, axis=0).T
    a4 = np.expand_dims(x**2, axis=0).T
    a5 = np.expand_dims(y**2, axis=0).T
    return np.hstack((a0, a1, a2, a3, a4, a5))

if __name__ == "__main__":
    runs_no = 1000
    n = 1000

    supervisor = lra.TargetVectorizedFunction(np.vectorize(foo), 0.1)

    prob = 0
    for run in xrange(runs_no):
        alg = lra.LinearRegressionAlgorithm(n, supervisor, feature_transformer)
        alg.compute_weights()
        prob += alg.get_prob(True)

    error = 1 - prob/float(runs_no)
    print('average: ', error)
