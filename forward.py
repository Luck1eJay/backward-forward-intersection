import numpy as np
from backward import BackwardItersection

class ForwardIntersection:
    def __init__(self, GCP1, GCP2, planePoint1, planePoint2,unknownPlanePoint1,unknownPlanePoint2):
        self.GCP1 = GCP1
        self.GCP2 = GCP2
        self.planePoint1 = planePoint1
        self.planePoint2 = planePoint2
        self.unknownPlanePoint1=unknownPlanePoint1
        self.unknownPlanePoint2=unknownPlanePoint2
        self.exterior_orientation1 = None
        self.exterior_orientation2 = None

    def compute_baseline_components(self):
        exterior_orientation1 = np.array(self.exterior_orientation1)
        exterior_orientation2 = np.array(self.exterior_orientation2)
        baseline = self.exterior_orientation2[:3] - self.exterior_orientation1[:3]
        print('exterior_orientation1:',exterior_orientation1)
        print('exterior_orientation2:',exterior_orientation2)
        print("Baseline:", baseline)
        return baseline

    def compute_auxiliary_coordinates(self, exterior_orientation, planePoint,f):
        phi, omega, kappa = exterior_orientation[3:]
        R = self.compute_rotation_matrix(phi, omega, kappa)
        planePointMatrix= np.hstack([planePoint, -f * np.ones((planePoint.shape[0], 1))])
        auxiliary_coords = R @ planePointMatrix.T
        print("Auxiliary Coordinates:", auxiliary_coords)
        return auxiliary_coords

    def compute_rotation_matrix(self, phi, omega, ka):
        R = np.matrix([[np.cos(ka) * np.cos(phi) - np.sin(ka) * np.sin(omega) * np.sin(phi), np.sin(ka) * np.cos(omega),
                        np.sin(phi) * np.cos(ka) + np.cos(phi) * np.sin(omega) * np.sin(ka)],
                       [-np.cos(phi) * np.sin(ka) - np.sin(phi) * np.sin(omega) * np.cos(ka),
                        np.cos(omega) * np.cos(ka),
                        -np.sin(phi) * np.sin(ka) + np.cos(phi) * np.sin(omega) * np.cos(ka)],
                       [-np.sin(phi) * np.cos(omega), -np.sin(omega), np.cos(omega) * np.cos(phi)]]).T
        print("Rotation Matrix:", R)
        return R

    def compute_projection_coefficients(self, baseline, auxiliary_coords1, auxiliary_coords2):
        projection_coefficients = np.zeros((auxiliary_coords1.shape[1], 2))
        for i in range(auxiliary_coords1.shape[1]):
            b = baseline
            a1 = auxiliary_coords1[:, i]
            a2 = auxiliary_coords2[:, i]
            N1 = (b[0]*a2[2]-b[2]*a2[0])/(a1[0]*a2[2]-a1[2]*a2[0])
            N2 = (b[0]*a1[2]-b[2]*a1[0])/(a2[0]*a1[2]-a2[2]*a1[0])
            projection_coefficients[i,0]= N1.item()
            projection_coefficients[i,1]= N2.item()
        print("Projection Coefficients:", projection_coefficients)
        return projection_coefficients

    def compute_U1V1W1_coordinates(self, projection_coefficients, auxiliary_coords1):
        num_coords = projection_coefficients.shape[0]
        UVW_coordinates = np.zeros((num_coords, 3))
        for i in range(num_coords):
            N1 = projection_coefficients[i, 0]
            UVW_coordinates[i, 0] = N1 * auxiliary_coords1[0, i]
            UVW_coordinates[i, 1] = N1 * auxiliary_coords1[1, i]
            UVW_coordinates[i, 2] = N1 * auxiliary_coords1[2, i]
        print("UVW_coordinates:", UVW_coordinates)
        return UVW_coordinates
    # def compute_U2V2W2_coordinates(self, projection_coefficients, auxiliary_coords2):
    #     num_coords = projection_coefficients.shape[0]
    #     UVW_coordinates = np.zeros((num_coords, 3))
    #     for i in range(num_coords):
    #         N2 = projection_coefficients[i, 1]
    #         UVW_coordinates[i, 0] = N2 * auxiliary_coords2[0, i]
    #         UVW_coordinates[i, 1] = N2 * auxiliary_coords2[1, i]
    #         UVW_coordinates[i, 2] = N2 * auxiliary_coords2[2, i]
    #     return UVW_coordinates

    def compute_ground_coordinates(self, projection_coefficients, auxiliary_coords1, auxiliary_coords2,U1V1W1_coordinates,exterior_orientation1,exterior_orientation2):

        num_coords = projection_coefficients.shape[0]
        ground_coordinates= np.zeros((num_coords, 3))
        ground_Y=np.zeros((num_coords, 1))
        for i in range(num_coords):
            N1 = projection_coefficients[i, 0]
            N2 = projection_coefficients[i, 1]
            ground_coordinates[i, 0] = exterior_orientation1[0] + U1V1W1_coordinates[i, 0]
            ground_coordinates[i, 1] = exterior_orientation1[1] + U1V1W1_coordinates[i, 1]
            ground_coordinates[i, 2] = exterior_orientation1[2] + U1V1W1_coordinates[i, 2]
            ground_Y[i,0]=0.5*(exterior_orientation1[1] + N1*auxiliary_coords1[1,i]+ exterior_orientation2[1]+ N2*auxiliary_coords2[1,i])
        print("Ground Coordinates:", ground_coordinates)
        print("Mean Ground Y:", ground_Y)
        return ground_coordinates

    def compute_intersection(self):
        # 调用后方交会
        backward1 = BackwardItersection(self.GCP1, self.planePoint1)
        backward2 = BackwardItersection(self.GCP2, self.planePoint2)
        self.exterior_orientation1 = backward1.backward()
        self.exterior_orientation2 = backward2.backward()

        baseline = self.compute_baseline_components()
        fk = backward1.fk
        auxiliary_coords1 = self.compute_auxiliary_coordinates(self.exterior_orientation1, self.unknownPlanePoint1,fk)
        auxiliary_coords2 = self.compute_auxiliary_coordinates(self.exterior_orientation2, self.unknownPlanePoint2,fk)
        projection_coefficients = self.compute_projection_coefficients(baseline, auxiliary_coords1, auxiliary_coords2)
        U1V1W1_coordinates = self.compute_U1V1W1_coordinates(projection_coefficients, auxiliary_coords1)
        ground_coordinates = self.compute_ground_coordinates(projection_coefficients, auxiliary_coords1, auxiliary_coords2,U1V1W1_coordinates,self.exterior_orientation1,self.exterior_orientation2)
        return ground_coordinates

