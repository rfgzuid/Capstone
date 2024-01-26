import numpy as np
import torch
from scipy.stats import truncnorm
from scipy.special import erf, erfc


def log_prob(x, y):
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
    return mean + std_dev * (truncnorm.expect(args=(a, b), loc=mean, scale=std_dev))

def weighted_noise_prob(HR, h_ids, stds):
    res = torch.tensor(len(h_ids))
    HR_prob = HR_probability(HR)
    for i in range(len(h_ids)):
        res[i] = HR_prob * truncated_normal_expectation(0, stds[i], HR.lower[i], HR.upper[i])
    return res

def HR_probability(HR, h_ids, stds):
    lower_list = []
    upper_list = []
    len_vector = len(h_ids)
    for i in range(len_vector):
        lower_list += [HR.lower[i]/stds[i]]
        upper_list += [HR.upper[i]/stds[i]]
    prob = 0
    for j in range(len_vector):
        prob += (log_prob(upper_list[j], lower_list[j]))
    return np.exp(prob)