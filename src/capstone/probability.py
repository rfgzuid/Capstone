import numpy as np
from scipy.stats import truncnorm
from scipy.special import erf, erfc


# TODO: Maybe move this function?
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
    
# TODO: Maybe move this function?
def truncated_normal_expectation(mean, std_dev, lower_bound, upper_bound):
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
    return mean + std_dev * (truncnorm.expect(args=(a, b), loc=mean, scale=std_dev))