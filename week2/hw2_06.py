import lra

if __name__ == "__main__":
    runs_no = 1000
    n = 100

    prob_sum = 0
    for run in xrange(runs_no):
        alg = lra.LinearRegressionAlgorithm(n)
        temp_prob = alg.get_prob(True)
        print('prob for run ', run, ' = ', temp_prob)
        prob_sum += temp_prob

    print('average: ', 1 - prob_sum/float(runs_no))
