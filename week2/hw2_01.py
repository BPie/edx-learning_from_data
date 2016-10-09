import coin_flipper

if __name__ == "__main__":
    student = coin_flipper.CoinFlipper()

    steps_no = 100000
    min_sum = 0.
    for step in xrange(steps_no):
        student.run()
        min_sum += student.get_vmin()

    print('average vmin = ', min_sum/float(steps_no))
