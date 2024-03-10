import numpy as np
import torch
from scipy.special import erf, erfc


def log_prob(x, y):
    if x > y:
        x, y = y, x
    x, y = x/np.sqrt(2), y/np.sqrt(2)
    if abs(x) <= 1/np.sqrt(2) and abs(y) <= 1/np.sqrt(2):
        return np.log((erf(y) - erf(x))/2)
    elif x >= 0 and y >= 0:
        return np.log((erfc(x) - erfc(y))/2)
    elif x <= 0 and y <= 0:
        return np.log((erfc(-y) - erfc(-x))/2)
    else:
        return np.log((erf(y) - erf(x))/2)


def diagonalize_symmetric_matrix(A):
    """
    Diagonalizes a symmetric matrix A.
    Returns P, D such that A = P @ D @ P.T,
    where D is a diagonal matrix and P is an orthogonal matrix.

    Parameters:
    - A: numpy array representing a symmetric matrix.

    Returns:
    - P: Orthogonal matrix where columns are eigenvectors of A.
    - D: Diagonal matrix containing eigenvalues of A.
    """
    # Ensure A is a numpy array
    A = np.array(A)

    # Compute the eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # D is a diagonal matrix of the eigenvalues
    D = np.diag(eigenvalues)

    # P is the matrix of eigenvectors
    P = eigenvectors

    return P, D


def correlated_truncated_normal_expectation(mean, covariance, lower_bound, upper_bound):
    P, D = diagonalize_symmetric_matrix(covariance)
    P_T = P.T
    new_lower = P_T * HR.lower
    new_upper = P_T * HR.upper
    stds = np.diag(D)
    mean = np.zeros(D.shape[0])
    return P * truncated_normal_expectation(mean, stds, new_lower, new_upper)


def correlated_HR_probability(HR, cov):
    P, D = diagonalize_symmetric_matrix(cov)
    P_T = P.T
    new_lower = P_T * HR.lower
    new_upper = P_T * HR.upper
    new_HR = HyperRectangle(new_lower, new_upper)
    stds = np.diag(D)
    h_ids = [i for i in range(D.shape[0])]
    return HR_probability(new_HR, h_ids, stds)


def truncated_normal_expectation(mean, std, lower_bound, upper_bound):
    # https://en.wikipedia.org/wiki/Truncated_normal_distribution
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std

    phi_a, psi_a = 1 / (2 * np.pi) ** 0.5 * np.exp(-a ** 2 / 2), 0.5 * (1 + erf(a / 2 ** 0.5))
    phi_b, psi_b = 1 / (2 * np.pi) ** 0.5 * np.exp(-b ** 2 / 2), 0.5 * (1 + erf(b / 2 ** 0.5))

    expectation = mean - std * (phi_b - phi_a)/(psi_b - psi_a)
    # expectation = mean - std * (phi_b - phi_a) / (np.exp(log_prob(b, a)))
    return expectation


def weighted_noise_prob(HR, h_ids, stds):
    HR_prob = HR_probability(HR, h_ids, stds)
    t = truncated_normal_expectation(0, torch.tensor(stds), HR.lower[:, h_ids], HR.upper[:, h_ids])
    return HR_prob * t.squeeze()


def HR_probability(HR, h_ids, stds):
    lower_list = []
    upper_list = []
    len_vector = len(h_ids)
    for i in range(len_vector):
        lower_list += [HR.lower[:, h_ids[i]]/stds[i]]
        upper_list += [HR.upper[:, h_ids[i]]/stds[i]]
    prob = 0
    for j in range(len_vector):
        prob += (log_prob(upper_list[j], lower_list[j]))
    return np.exp(prob)
