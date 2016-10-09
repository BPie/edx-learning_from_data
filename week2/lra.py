import numpy as np
import matplotlib.pyplot as plt


def get_random_point():
    return np.random.rand(2)*2-1


def get_perp(p1, p2):
    return np.array([
        p2[1] - p1[1],
        p1[0] - p2[0]])


class TargetFunction:
    def __init__(self):
        self.p1 = get_random_point()
        self.p2 = get_random_point()
        self.p_noise = 0

    def classify_bool(self, points):
        perp = get_perp(self.p1, self.p2)
        point_mat = points-self.p1
        ret = np.tensordot(perp, point_mat, axes=([0], [1])) > 0
        if self.p_noise > 0:
            noisy_count = int(self.p_noise*ret.size)
            ret[0:noisy_count] = np.logical_not(ret[0:noisy_count])
        return ret

    def classify(self, points):
        bool_result = self.classify_bool(points)

        int_result_pos = bool_result * (1.)
        int_result_neg = np.logical_not(bool_result) * (-1.)
        int_result = int_result_pos + int_result_neg
        return np.expand_dims(int_result, axis=1)

    def group(self, points):
        classify_mat = self.classify(points)
        class_positive = points[np.where(classify_mat > 0)[0], :]
        class_negative = points[np.where(classify_mat < 0)[0], :]

        return class_positive, class_negative


class TargetVectorizedFunction(TargetFunction):
    def __init__(self, vfunc2d, p_noise=0.0):
        self._vf = vfunc2d
        self.p_noise = p_noise

    def classify_bool(self, points):
        ret = self._vf(points[:, 0], points[:, 1]) > 0
        if self.p_noise > 0:
            noisy_count = int(self.p_noise * ret.size)
            ret[0:noisy_count] = np.logical_not(ret[0:noisy_count])
        return ret


class Data:
    def __init__(self, n=10):
        self._n = n
        # faster than generating in a for loop (like in pla.py)
        self._data = np.random.rand(n, 2)*2-1

    def get_numpy_data(self):
        return self._data

    def get_numpy_data_enhanced(self):
        numpy_data = self.get_numpy_data()
        prefix = np.ones([numpy_data.shape[0], 1])
        return np.hstack((prefix, numpy_data))

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


class LinearRegressionAlgorithm:
    def __init__(self, n=10, supervisor=None, feature_transformator=None):
        self._w = np.zeros([3])
        if supervisor:
            self._supervisor = supervisor
        else:
            self._supervisor = TargetFunction()
        self._ft = feature_transformator
        self._data = Data(n)
        self._n = n
        self._fig = None
        self._sub_plot = None
        self._line = None

    def print_data(self):
        print(self._data.get_numpy_data())

    def compute_weights(self):
        # w = (X.T * X)^(-1)*X.T
        x = self._data.get_numpy_data()
        if self._ft:
            x_enh = self._ft(x)
        else:
            x_enh = self._data.get_numpy_data_enhanced()
        y = self._supervisor.classify(x)
        pseudo_inv = np.linalg.pinv(x_enh)
        self._w = np.dot(pseudo_inv, y)

    def get_prob(self, new_data=False):
        self.compute_weights()

        if new_data:
            self._data = Data(self._n)

        x = self._data.get_numpy_data()
        if self._ft:
            x_enh = self._ft(x)
        else:
            x_enh = self._data.get_numpy_data_enhanced()

        classify_mat = np.tensordot(self._w, x_enh, axes=([0], [1])) > 0
        # classify_mat = np.dot(self._w.T, x_enh) > 0
        true_classify_mat = self._supervisor.classify_bool(x)

        proper_class_mat = classify_mat == true_classify_mat
        return np.sum(proper_class_mat)/float(proper_class_mat.size)

    def plot(self):
        c_pos, c_neg = self._supervisor.group(self._data.get_numpy_data())
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

        # # computing line for data
        # p1 = self._supervisor.p1
        # p2 = self._supervisor.p2
        # self._line.set_xdata([p1[0], p2[0]])
        # self._line.set_ydata([p1[1], p2[1]])

        self._fig.canvas.draw()

if __name__ == "__main__":
    alg = LinearRegressionAlgorithm(500)
    alg.compute_weights()
    alg.plot()

    raw_input("press any key to quit")
