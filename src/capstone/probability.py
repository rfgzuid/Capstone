import numpy as np
import torch
from scipy.special import erf, erfc


def log_prob(x, y):
    if x > y:
        x, y = y, x
    x, y = x/np.sqrt(2), y/np.sqrt(2)
    if abs(x) <= 1/np.sqrt(2) and abs(y) <= 1/np.sqrt(2):
        return (erf(y) - erf(x))/2
    elif x >= 0 and y >= 0:
        return (erfc(x) - erfc(y))/2
    elif x <= 0 and y <= 0:
        return (erfc(-y) - erfc(-x))/2
    else:
        return (erf(y) - erf(x))/2


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
    new_lower = np.matmul(P_T, lower_bound)
    new_upper = np.matmul(P_T, upper_bound)
    stds = np.diag(D)
    mean = np.zeros(D.shape[0])
    return np.matmul(P, truncated_normal_expectation(mean, stds, new_lower, new_upper))


def correlated_HR_probability(HR, cov):
    P, D = diagonalize_symmetric_matrix(cov)
    P_T = P.T
    new_lower = np.matmul(P_T, HR.lower)
    new_upper = np.matmul(P_T, HR.upper)
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

def truncated_normal_expectation_batched(mean, std, lower_bound, upper_bound):
    # mean, std should be tensors of the shape [1, 1] or scalar values.
    # lower_bound, upper_bound should be tensors of shape [num_noises, len(h_ids)].
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
    
    phi_a = 1 / (2 * torch.pi) ** 0.5 * torch.exp(-a ** 2 / 2)
    phi_b = 1 / (2 * torch.pi) ** 0.5 * torch.exp(-b ** 2 / 2)
    psi_a = 0.5 * (1 + torch.erf(a / 2 ** 0.5))
    psi_b = 0.5 * (1 + torch.erf(b / 2 ** 0.5))

    expectation = mean - std * (phi_b - phi_a) / (psi_b - psi_a)
    return expectation

def weighted_noise_prob_batched(HR, h_ids, stds):
    # First, compute the batched probability for the noise partitions.
    HR_prob = HR_probability_batched(HR, h_ids, stds)
    # Now, compute the expectation for the noise partitions.
    # Note: mean is assumed to be 0 for all partitions here, can be adjusted if different.
    mean = torch.zeros(1, device=HR.lower.device)
    std_tensor = torch.tensor(stds, device=HR.lower.device, dtype=HR.lower.dtype)
    # Ensure lower_bound and upper_bound are correctly selected for h_ids and reshaped.
    lower_bound = HR.lower[..., h_ids]
    upper_bound = HR.upper[..., h_ids]
    expectation = truncated_normal_expectation_batched(mean, std_tensor, lower_bound, upper_bound)
    # Multiplying the probabilities with the expectations, element-wise.
    weighted_noise_proba = HR_prob.view(-1, 1) * expectation
    return weighted_noise_proba


def weighted_noise_prob(HR, h_ids, stds):
    HR_prob = HR_probability(HR, h_ids, stds)
    t = truncated_normal_expectation(0, torch.tensor(stds), HR.lower[:, h_ids], HR.upper[:, h_ids])
    print("t", t.shape)
    print("final", (HR_prob * t.squeeze()).shape)
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
        prob *= (log_prob(upper_list[j], lower_list[j]))
    return prob

def HR_probability_batched(HR, h_ids, stds):
    if not isinstance(stds, torch.Tensor):
        # No need to unsqueeze - broadcast (since the relevant indexing is in the last dimension) will take care of it
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        stds = torch.tensor(stds, device=HR.lower.device, dtype=HR.lower.dtype)
    # Correcting for the provided shape of HR.lower and HR.upper
    lower_normalized = HR.lower[..., h_ids] / stds
    upper_normalized = HR.upper[..., h_ids] / stds
    # Compute log probabilities in a batched manner
    log_probs = log_prob_batched(upper_normalized, lower_normalized)
    # Sum log probabilities across the h_ids dimensions
    log_prob_sum = torch.sum(log_probs, dim=-1)  # Summing across the last dimension
    # Convert log probabilities back to probabilities
    prob = torch.exp(log_prob_sum)
    return prob

def log_prob_batched(x, y):
    # Ensure inputs are torch tensors
    x, y = torch.as_tensor(x), torch.as_tensor(y)

    # Normalize inputs
    x, y = x / torch.sqrt(torch.tensor(2.0)), y / torch.sqrt(torch.tensor(2.0))

    # Find where x > y and swap
    swap_mask = x > y
    x_temp = x.clone()
    x[swap_mask], y[swap_mask] = y[swap_mask], x_temp[swap_mask]
    
    # Initialize result tensor
    result = torch.zeros_like(x)
    
    # Compute probabilities using error functions and their complements
    mask1 = (torch.abs(x) <= 1/torch.sqrt(torch.tensor(2.0))) & (torch.abs(y) <= 1/torch.sqrt(torch.tensor(2.0)))
    result[mask1] = torch.log((erf(y[mask1]) - erf(x[mask1])) / 2)

    mask2 = (x >= 0) & (y >= 0)
    result[mask2] = torch.log((erfc(x[mask2]) - erfc(y[mask2])) / 2)

    mask3 = (x <= 0) & (y <= 0)
    result[mask3] = torch.log((erfc(-y[mask3]) - erfc(-x[mask3])) / 2)

    mask4 = ~(mask1 | mask2 | mask3)
    result[mask4] = torch.log((erf(y[mask4]) - erf(x[mask4])) / 2)
    
    return result