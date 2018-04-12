import numpy as np

import physics_model 
import utils 

def generate(model=physics_model.BeamSpinAsymmetryModel(),
             parameters=None,
             n_values=12, error=0.05):

    # basic axis 
    phi = np.linspace(-180.0, 180.0, n_values+1)

    # shift to center of bin
    phi_center = [phi[i] + 0.5*(phi[i+1]-phi[i]) for i in range(n_values)]
    phi_center = np.array(phi_center, dtype=np.float32)

    if parameters is None:
        model.initialize_parameters() 
    else:
        model.pars = parameters
        
    value = np.random.normal(0.0, error, n_values) + \
            model.evaluate(phi_center)

    err = np.repeat(error, n_values)

    return phi_center, value, err
