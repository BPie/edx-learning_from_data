import numpy as np

def f(x,y):
    temp = x*0.6-y*0.2+0.3>0 
    return temp

class pla2d:
    def __init__(self, x1=0., x2=0., y1=0., y2=0.):
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        
    def __init__(self, p1, p2):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)

    def set_points(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def _get_perp(self):
        return np.array([ 
            self.p2[1] - self.p1[1], 
            self.p1[0] - self.p2[0] ])

    def validate(self, point):
        return np.dot(self._get_perp(), point-self.p1) > 0



if __name__ == "__main__":
    temp = pla2d([0.1, 0.1],
                 [0.4, 0.4])

    p1 = np.array([0., 1.])
    p2 = np.array([1., 0.])
    p3 = np.array([0., 0.])
    p4 = np.array([1., 1.])
    p5 = np.array([1., -1.])

    print temp.validate(p1)
    print temp.validate(p2)
    print temp.validate(p3)
    print temp.validate(p4)
    print temp.validate(p5)


