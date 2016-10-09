import numpy as np
import matplotlib.pyplot as plt


def get_random_point():
    return np.random.rand(2)*2-1


def bool_to_int(bool_inp):
    if bool_inp:
        return 1
    else:
        return -1


def get_perp(p1, p2):
    return np.array([
        p2[1] - p1[1],
        p1[0] - p2[0]])


class TargetFunction:
    def __init__(self):
        self.p1 = get_random_point()
        self.p2 = get_random_point()

    def classify(self, point):
        perp_vec = get_perp(self.p1, self.p2)
        point_vec = point-self.p1
        return np.dot(perp_vec, point_vec) > 0

    def group(self, points):
        class_positive = []
        class_negative = []
        for point in points:
            if self.classify(point):
                class_positive.append(point)
            else:
                class_negative.append(point)
        return class_positive, class_negative


class Data:
    def __init__(self, n=10):
        self._n = n
        self._data = []
        for i in xrange(n):
            self._data.append(get_random_point())

    def __iter__(self):
        self._current = 0
        return self

    def next(self):
        if self._current >= self._n:
            raise StopIteration
        else:
            self._current += 1
            return self._data[self._current - 1]

    def __getitem__(self, index):
        return self._data[index]


class Pla2dRunner:
    def __init__(self, n=10):
        self._w = np.zeros([3])
        self._supervisor = TargetFunction()
        self._data = Data(n)
        self._step = 0
        self._fig = None
        self._sub_plot = None
        self._line = None

    def __init__(self, data, weights, supervisor):
        self._w = weights
        self._supervisor = supervisor
        self._data = data
        self._step = 0
        self._fig = None
        self._sub_plot = None
        self._line = None

    def _get_invalid_points(self):
        ret = []
        for point in self._data:
            if not self._validate(point):
                ret.append(point)
        return ret

    def _get_random_invalid_point(self):
        invalid_points = self._get_invalid_points()
        if len(invalid_points) == 0:
            return None
        else:
            random_iter = np.random.random_integers(0, len(invalid_points)-1)
            return invalid_points[random_iter]

    def _validate(self, point):
        enhanced_point = np.hstack(([1.], point))
        temp_val = self._w * enhanced_point
        class_value = np.sum(temp_val) > 0
        # class_value = np.dot(np.transpose(self._w), point) > 0
        return class_value == self._supervisor.classify(point)

    def _update(self, point):
        y = bool_to_int(self._supervisor.classify(point))
        enhanced_point = np.hstack(([1.], point))
        temp_val = y * enhanced_point
        self._w = np.add(self._w, temp_val)

    def run_iteration(self):
        self._step += 1
        invalid_point = self._get_random_invalid_point()
        if invalid_point is None:
            return True, self._step

        self._update(invalid_point)
        return False, self._step

    def get_prob(self, data_size=1000):
        prob_data = Data(data_size)
        miss_count = 0
        for point in prob_data:
            if not self._validate(point):
                miss_count += 1
        return miss_count / float(data_size)

    def plot(self):
        c_pos, c_neg = self._supervisor.group(self._data)
        c_pos = np.array(c_pos)
        c_neg = np.array(c_neg)
        if not hasattr(self, "_fig") or self._fig is None:
            plt.ion()
            self._fig = plt.figure()
            self._sub_plot = self._fig.add_subplot("111")
            self._line, = self._sub_plot.plot([0., 0.], [0., 0.], 'r-', lw=2)
            # plotting data points (once)
            self._sub_plot.scatter(c_pos[:, 0], c_pos[:, 1], c='green')
            self._sub_plot.scatter(c_neg[:, 0], c_neg[:, 1], c='red')

        # computing line parameters from weights
        a = -1. * self._w[1]/self._w[2]
        b = -1. * self._w[0]/self._w[2]
        start_x = -1.5
        stop_x = 1.5
        # plotting line
        self._line.set_xdata([start_x, stop_x])
        self._line.set_ydata([start_x*a+b, stop_x*a+b])
        self._fig.canvas.draw()

if __name__ == "__main__":
    step_refresher = 1
    alg = Pla2dRunner(500)
    for i in xrange(1000):
        done, step = alg.run_iteration()
        if step % step_refresher == 0 or done:
            print 'step: ', step, " prob: ", alg.get_prob(1000)*100, "%"
            alg.plot()

        if done:
            print 'done in ', step, ' steps'
            raw_input("press any key to quit")
            break
