import numpy as np
import scipy.optimize as opt
import scipy.signal as sig
from scipy.stats import pearsonr

def fun(k, ar1, ar2):
    return np.abs(np.sum(arr1 - k*arr2))

    # return np.corrcoef(ar1,k*ar2)


arr1 = np.asarray(list(range(3,8))+list(range(6,1))+list(range(3,11))+list(range(9,4)))
arr2 = arr1/3

# x0 = np.arange(0.1,5,0.05)
x0 = 2
bestK = opt.minimize(fun, x0, args=(arr1, arr2))

print(bestK)
