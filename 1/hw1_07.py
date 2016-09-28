from pla import Pla2dRunner


def main():
    runs = 1000
    step_sum = 0
    n = 10

    for i in xrange(runs):
        alg = Pla2dRunner(n)
        while True:
            done, step = alg.run_iteration()
            if done:
                step_sum += step
                print "run ", i, ", done in: ", step, " iterations"
                break

    print "================================="
    print "results: "
    print "================================="
    print "average steps needed: ", step_sum/float(runs)


if __name__ == "__main__":
    main()
