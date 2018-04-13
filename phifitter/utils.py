import numpy as np 

def create_replica(y, y_err):
    y_rep = [np.random.normal(yp,np.fabs(yp_err)) for yp,yp_err in zip(y,y_err)]
    return np.array(y_rep)

def random_search(function, bounds, n_samples):
    '''
    Generate randomly within the bounds and find the minumum of 
    function.
    '''

    if bounds is None:
        print('Cannot start random search without bounds')
        exit() 

    dims = len(bounds)
    samples = np.zeros(shape=(n_samples,dims))

    for i in range(dims):
        samples[:,i] = np.random.uniform(bounds[i][0], 
                                         bounds[i][1],
                                         n_samples)

    value = 1e24
    x0 = None
    for i in range(n_samples):
        current_value = function(samples[i,:])
        
        if current_value < value:
            value = current_value 
            x0 = samples[i,:]

    if x0 is None:
        print('Trouble with random search, got x0 = None after %d trials' % n_samples)

    return x0
