import numpy as np 

def chi2(data, theory, error):
    return np.sum((data-theory)**2/error**2)

# Maximized not minimized
def likelihood(data, theory, error):
    return np.exp(-0.5*chi2(data, theory, error))
