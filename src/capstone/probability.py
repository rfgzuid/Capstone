import numpy as np
import torch
from scipy.stats import truncnorm
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


def truncated_normal_expectation(mean, std_dev, lower_bound, upper_bound):
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
    res = truncnorm.expect(args=(a, b), loc=mean, scale=std_dev)
    return res


def test(mean, std, lower_bound, upper_bound):
    # https://en.wikipedia.org/wiki/Truncated_normal_distribution
    a, b = (lower_bound - mean) / std, (upper_bound - mean) / std

    phi_a, psi_a = 1 / (2 * np.pi) ** 0.5 * np.exp(-a ** 2 / 2), 0.5 * (1 + erf(a / 2 ** 0.5))
    phi_b, psi_b = 1 / (2 * np.pi) ** 0.5 * np.exp(-b ** 2 / 2), 0.5 * (1 + erf(b / 2 ** 0.5))

    expectation = mean - std * (phi_b - phi_a)/(psi_b - psi_a)
    # expectation = mean - std * (phi_b - phi_a) / (np.exp(log_prob(b, a)))
    return expectation



def weighted_noise_prob(HR, h_ids, stds):
    res = torch.zeros(len(h_ids))
    HR_prob = HR_probability(HR, h_ids, stds)
    for i in range(len(h_ids)):
        res[i] = (HR_prob * truncated_normal_expectation(0, stds[i], HR.lower[:, h_ids[i]], HR.upper[:, h_ids[i]]))
    t = test(0, torch.tensor(stds), HR.lower[:, h_ids], HR.upper[:, h_ids])
    print(res, HR_prob*t.squeeze())
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
