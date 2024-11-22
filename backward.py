import numpy as np
from parameter import initialize_parameters
class BackwardItersection:
    def __init__(self,GCP,planePoint):
        self.fk, self.m, self.center, self.matri1, self.matri2, self.PhotoCenter, self.H, self.phi, self.omega, self.ka = initialize_parameters(GCP,planePoint)

    def compute_rotation_matrix(self, phi, omega, ka):
        R = np.matrix([[np.cos(ka) * np.cos(phi) - np.sin(ka) * np.sin(omega) * np.sin(phi),
                        np.sin(ka) * np.cos(omega),
                        np.sin(phi) * np.cos(ka) + np.cos(phi) * np.sin(omega) * np.sin(ka)],
                       [-np.cos(phi) * np.sin(ka) - np.sin(phi) * np.sin(omega) * np.cos(ka),
                        np.cos(omega) * np.cos(ka),
                        -np.sin(phi) * np.sin(ka) + np.cos(phi) * np.sin(omega) * np.cos(ka)],
                       [-np.sin(phi) * np.cos(omega), -np.sin(omega), np.cos(omega) * np.cos(phi)]])
        return R

    def compute_matrices(self, fk, matri1, matri2, PhotoCenter, R, omega, ka):
        a1, b1, c1 = R[0, 0], R[0, 1], R[0, 2]
        a2, b2, c2 = R[1, 0], R[1, 1], R[1, 2]
        a3, b3, c3 = R[2, 0], R[2, 1], R[2, 2]
        lxy = np.zeros_like(matri2)
        Zb = np.zeros((matri1.shape[0], 1))
        A = np.zeros((2 * matri1.shape[0], 6))

        for i in range(matri2.shape[0]):
            lx = matri2[i, 0] + fk * (
                    a1 * (matri1[i, 0] - PhotoCenter[0, 0]) + b1 * (matri1[i, 1] - PhotoCenter[0, 1]) + c1 * (
                    matri1[i, 2] - PhotoCenter[0, 2])) / (a3 * (matri1[i, 0] - PhotoCenter[0, 0]) + b3 * (
                    matri1[i, 1] - PhotoCenter[0, 1]) + c3 * (matri1[i, 2] - PhotoCenter[0, 2]))
            ly = matri2[i, 1] + fk * (
                    a2 * (matri1[i, 0] - PhotoCenter[0, 0]) + b2 * (matri1[i, 1] - PhotoCenter[0, 1]) + c2 * (
                    matri1[i, 2] - PhotoCenter[0, 2])) / (a3 * (matri1[i, 0] - PhotoCenter[0, 0]) + b3 * (
                    matri1[i, 1] - PhotoCenter[0, 1]) + c3 * (matri1[i, 2] - PhotoCenter[0, 2]))
            lxy[i, 0], lxy[i, 1] = lx, ly
            x, y = matri2[i, 0], matri2[i, 1]

            Zb[i, 0] = a3 * (matri1[i, 0] - PhotoCenter[0, 0]) + b3 * (matri1[i, 1] - PhotoCenter[0, 1]) + c3 * (
                    matri1[i, 2] - PhotoCenter[0, 2])
            A[2 * i, 0] = (a1 * fk + a3 * x) / Zb[i, 0]
            A[2 * i, 1] = (b1 * fk + b3 * x) / Zb[i, 0]
            A[2 * i, 2] = (c1 * fk + c3 * x) / Zb[i, 0]
            A[2 * i, 3] = y * np.sin(omega) - (
                    x * (x * np.cos(ka) - y * np.sin(ka)) / fk + fk * np.cos(ka)) * np.cos(omega)
            A[2 * i, 4] = -fk * np.sin(ka) - x * (x * np.sin(ka) + y * np.cos(ka)) / fk
            A[2 * i, 5] = y
            A[2 * i + 1, 0] = (a2 * fk + a3 * y) / Zb[i, 0]
            A[2 * i + 1, 1] = (b2 * fk + b3 * y) / Zb[i, 0]
            A[2 * i + 1, 2] = (c2 * fk + c3 * y) / Zb[i, 0]
            A[2 * i + 1, 3] = -x * np.sin(omega) - (
                    y * (x * np.cos(ka) - y * np.sin(ka)) / fk - fk * np.sin(ka)) * np.cos(omega)
            A[2 * i + 1, 4] = -fk * np.cos(ka) - y * (x * np.sin(ka) + y * np.cos(ka)) / fk
            A[2 * i + 1, 5] = -x

        l = np.matrix(lxy).reshape(8, 1)
        return Zb, A, l

    def update_parameters(self, A, l, PhotoCenter, phi, omega, ka):
        A = np.matrix(A)
        mat1 = ((A.T * A).I * A.T * l)
        delta = mat1
        dphi, domega, dka = delta[3, 0], delta[4, 0], delta[5, 0]
        phi += dphi
        omega += domega
        ka += dka
        PhotoCenter[0, 0] += delta[0, 0]
        PhotoCenter[0, 1] += delta[1, 0]
        PhotoCenter[0, 2] += delta[2, 0]
        return PhotoCenter, phi, omega, ka, delta

    def backward(self):
        times = 0
        V=None
        while True:
            times += 1
            # print('第', times, '次迭代')
            R = self.compute_rotation_matrix(self.phi, self.omega, self.ka)
            Zb, A, l = self.compute_matrices(self.fk, self.matri1, self.matri2, self.PhotoCenter, R, self.omega,
                                             self.ka)
            self.PhotoCenter, self.phi, self.omega, self.ka, delta = self.update_parameters(A, l, self.PhotoCenter,
                                                                                            self.phi, self.omega,
                                                                                            self.ka)
            # print('PhotoCenter:', self.PhotoCenter)
            # print('phi:', self.phi)
            # print('omega:', self.omega)
            # print('ka:', self.ka)
            if np.abs(delta).max() < 3e-5:
                # print('delta:', delta)
                # print('dphi', delta[3, 0])
                # print('domega', delta[4, 0])
                # print('dka', delta[5, 0])
                V = A @ delta - l
                break
        print('迭代次数:', times)
        print('phi:', self.phi)
        print('omega:', self.omega)
        print('ka:', self.ka)
        print('PhotoCenter:', self.PhotoCenter)

        #estimate unit weight error
        VV=np.sum(np.square(V))
        m0=np.sqrt(VV/(2*len(self.matri1)-6))
        print('Unit Weight Error:',m0)
        Xs=self.PhotoCenter[0,0]
        Ys=self.PhotoCenter[0,1]
        Zs=self.PhotoCenter[0,2]
        return np.hstack((np.array([Xs]),np.array([Ys]),np.array([Zs]), np.array([self.phi]), np.array([self.omega]),np.array( [self.ka])))
