import numpy as np


class CoinFlipper:

    def __init__(self, no_of_coins=1000, no_of_flips=10):
        self._no_of_coins = no_of_coins
        self._no_of_flips = no_of_flips

        self.__run = 0
        self.__last_result = None

        self._p1 = 0.5
        self._p0 = 0.5
        assert self._p1 + self._p0 == 1.0

    def run(self):
        self.__run += 1
        # print('running simulation...', self.__run)
        temp_size = [self._no_of_coins, self._no_of_flips]
        self.__last_result = \
            np.random.choice([1., 0.], temp_size, [self._p1, self._p0])

    def get_v1(self):
        if self.__last_result is None:
            self.run()
        return np.sum(self.__last_result[0])/float(self._no_of_flips)

    def get_vmin(self):
        if self.__last_result is None:
            self.run()
        fraction_matrix = np.sum(self.__last_result, axis=1)/self._no_of_flips
        return np.min(fraction_matrix)
