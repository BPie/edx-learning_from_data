from pla import Pla2dRunner


def main():
    runs = 1000
    prob_sum = 0
    prob_testing_set_size = 1000
    n = 100

    for i in xrange(runs):
        alg = Pla2dRunner(n)
        while True:
            done, step = alg.run_iteration()
            if done:
                current_prob = alg.get_prob(prob_testing_set_size)*100
                prob_sum += current_prob
                break

    print "================================="
    print "results: "
    print "================================="
    print "average miss prob: ", prob_sum/float(runs), " % (", prob_sum/float(runs)/100., ")"


if __name__ == "__main__":
    main()
