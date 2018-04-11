import numpy as np 

class Model(object):
    '''
    Model is the base clas that defines 
    the behaviour of the predictive functions
    that our fitters will expect.

    '''

    def __init__(self):
        self.n_pars = 1 
        self.pars = np.random.uniform(-1, 1, 1)

    def evaluate(self, x):
        '''
        x: The data sent in we expect to be 
        phi. 
        '''
        return 1.0 

    def initialize_parameters(self):
        '''
        Setup random parameters. 
        '''
        self.pars = np.random.uniform(-1, 1, self.n_pars)

    def update(self, p):
        self.pars = p

    def update_and_evaluate(self, x, p):
        self.update(p)
        return self.evaluate(x)

class BeamSpinAsymmetryModel(Model):
    '''
    BeamSpinAsymmetryModel defines the phi 
    dependence based on the cross section. 

    It has 3 parameters in this case, which 
    are ratios of structure functions. 

    '''

    def __init__(self):
        self.n_pars = 3 
        self.initialize_parameters() 

    def evaluate(self, x):
        return (self.pars[0] * np.sin(x*np.pi/180.0))/(1 + self.pars[1] * np.cos(x*np.pi/180.0) + \
                       self.pars[2] * np.cos(x*np.pi/90.0))

class UnpolarizedSIDISModel(Model):
    '''
    UnpolarizedSIDISModel defines the phi dependence of the 
    SIDIS cross section expected from theory. 
    '''

    def __init__(self):
        self.n_pars = 3 
        self.initialize_parameters() 

    def evaluate(self, x):
        return self.pars[0] + self.pars[1] * np.cos(x*np.pi/180.0) + \
            self.pars[2] * np.cos(x*np.pi/90.0)

