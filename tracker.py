import numpy as np
import scipy.linalg

class KalmanFilter(object):

    def __init__(self):
        ndim, dt = 2, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1. / 30
        self._std_weight_velocity = 1. / 250
        self._std_weight_mea_detect = 0.05
        self._std_weight_mea_LK = 0.1

    def initiate(self, measurement, w, h):

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [2 * self._std_weight_position * w, 2 * self._std_weight_position * h, 10 * self._std_weight_velocity * w, 10 * self._std_weight_velocity * h]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, w, h):

        std_pos = [self._std_weight_position * w, self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * w, self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, w, h, mea_type='detect'):

        std_weight_mea = self._std_weight_mea_detect if mea_type == 'detect' else self._std_weight_mea_LK

        std = [std_weight_mea * w, std_weight_mea * h]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, w, h, mea_type):

        projected_mean, projected_cov = self.project(mean, covariance, w, h, mea_type)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance