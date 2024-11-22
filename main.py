import numpy as np
from forward import ForwardIntersection
if __name__ == "__main__":
    GCP1 = np.matrix([[5083.205,5852.099,527.925],
                        [5780.02,5906.365,571.549],
                        [5210.879,4258.446,461.81],
                        [5909.264,4314.283,455.484]])
    GCP2=GCP1
    planePoint1 = np.matrix([[16.012, 79.963],
                            [88.56, 81.134],
                            [13.362, -79.37],
                            [82.24, -80.027]])
    planePoint2 = np.matrix([[-73.93, 78.706],
                            [-5.252, 78.184],
                            [-79.122, -78.879],
                            [-9.887, -80.089]])
    unknownPlanePoint1=np.matrix([[16.012, 79.963],
                            [88.56, 81.134],
                            [13.362, -79.37],
                            [82.24, -80.027],
                            [51.758,80.555],
                            [14.618,-0.231],
                            [49.88,-0.782],
                            [86.14,-1.346],
                            [48.035,-79.962]])
    unknownPlanePoint2=np.matrix([[-73.93, 78.706],
                            [-5.252, 78.184],
                            [-79.122, -78.879],
                            [-9.887, -80.089],
                            [-39.953,78.463],
                            [-76.006,0.036],
                            [-42.201,-1.022],
                            [-7.706,-2.112],
                            [-44.438,-79.736]])
    forward = ForwardIntersection(GCP1, GCP2, planePoint1, planePoint2,unknownPlanePoint1,unknownPlanePoint2)
    intersectionPoints2=forward.compute_intersection()
    print("Intersection Points:",intersectionPoints2)
