import numpy as np
import lra

def foo(x,y):
    return x**2 + y**2 -0.6


if __name__ == "__main__":
    runs_no = 1000
    n = 1000

    supervisor = lra.TargetVectorizedFunction(np.vectorize(foo), 0.1)

    prob_sum = 0
    for run in xrange(runs_no):
        alg = lra.LinearRegressionAlgorithm(n, supervisor)
        temp_prob = alg.get_prob()
        # print('prob for run ', run, ' = ', temp_prob)
        prob_sum += temp_prob

    print('average: ', 1 - prob_sum/float(runs_no))
