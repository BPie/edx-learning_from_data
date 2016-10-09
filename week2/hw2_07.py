import lra
from pla import Pla2dRunner


if __name__ == "__main__":
    runs_no = 1000
    n = 10
    steps_sum = 0
    for run in xrange(runs_no):
        alg = lra.LinearRegressionAlgorithm(n)
        alg.compute_weights()

        # todo fix
        alg2 = Pla2dRunner(alg._data, alg._w, alg._supervisor)
        alg2.plot()
        raw_input("press any key to quit")
        while True:
            steps_sum += 1
            done, st = alg2.run_iteration()
            if done:
                print("done in: ", st)
                break

    print('average: ', steps_sum/float(runs_no))
