import numpy as np
def initialize_parameters(GCP,planePoint):
    fk = 150.0
    m = 10000.0
    centerx0y0 = np.matrix([0.0, 0.0])
    PhotoCenter = np.sum(GCP, axis=0) / 4
    PhotoCenter[0, 2] = fk * m / 1000
    H = PhotoCenter[0, 2]
    phi = omega = ka = 0.0
    return fk, m, centerx0y0, GCP, planePoint, PhotoCenter, H, phi, omega, ka