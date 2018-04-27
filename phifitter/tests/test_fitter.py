#!/usr/bin/env python 

import multiprocessing 
import numpy as np 
import unittest

from phifitter import fitter 
from phifitter import loss 
from phifitter import physics_model 

class TestSingleFit(unittest.TestCase):
    def test(self):
        estimator = fitter.SingleFitter(loss_function=loss.chi2, 
                                        model=physics_model.BeamSpinAsymmetryModel())                          
        
        # generate a test set 
        phi   = np.linspace(-175.0, 175.0, 17)
        value = estimator.model.evaluate(phi) + np.random.normal(0, 0.01, len(phi)) 
        error = np.repeat(0.05, len(phi))

        # do fitting 
        true_pars = estimator.model.pars 
        result = estimator.fit(phi, value, error)

        self.assertTrue(len(result.fit_parameters) is estimator.model.n_pars)

class TestSingleRegularizedFit(unittest.TestCase):
    def test(self):
        estimator = fitter.SingleRegularizedFitter(loss_function=loss.chi2, 
                                                   model=physics_model.BeamSpinAsymmetryModel(),
                                                   penalty=1e-2)                          

        # generate a test set 
        phi   = np.linspace(-175.0, 175.0, 17)
        value = estimator.model.evaluate(phi) + np.random.normal(0, 0.05, len(phi)) 
        error = np.repeat(0.05, len(phi))
        
        # do fitting 
        true_pars = estimator.model.pars 
        result = estimator.fit(phi, value, error)

        self.assertTrue(len(result.fit_parameters) is estimator.model.n_pars)

class TestReplicaFit(unittest.TestCase):
    def test(self):

#        cores = multiprocessing.cpu_count() 
#        if not cores:

        # just test single core for now
        cores = 1
        estimator = fitter.ReplicaFitter(loss_function=loss.chi2, 
                                         model=physics_model.BeamSpinAsymmetryModel(), 
                                         n_cores=cores, n_replicas=20)                          
    
        # generate a test set 
        phi   = np.linspace(-175.0, 175.0, 17)
        value = estimator.model.evaluate(phi) + np.random.normal(0, 0.05, len(phi)) 
        error = np.repeat(0.05, len(phi))

        # do fitting 
        true_pars = estimator.model.pars 
        result = estimator.fit(phi, value, error)
    
        self.assertTrue(len(result.fit_parameters) is estimator.model.n_pars)

class TestBayesianVegasFit(unittest.TestCase):
    def test(self):

        def prior(p):
            return 1.0


        bounds = [[-1, 1], 
                  [-1, 1], 
                  [-1, 1]]
        estimator = fitter.BayesianVegasFitter(likelihood=loss.likelihood, prior=prior, 
                                               bounds=bounds, model=physics_model.BeamSpinAsymmetryModel(),
                                               n_iterations=10, n_evaluations=15000)
        
        # generate a test set
        phi   = np.linspace(-175.0, 175.0, 17)
        value = estimator.model.evaluate(phi) + np.random.normal(0, 0.05, len(phi))
        error = np.repeat(0.05, len(phi))
        
        # do fitting
        true_pars = estimator.model.pars
        result = estimator.fit(phi, value, error)

        self.assertTrue(len(result.fit_parameters) is estimator.model.n_pars)
        
class TestBayesianMCMCFit(unittest.TestCase):

    def test(self):
        def prior(p):
            return 1.0

        bounds = [[-1, 1],
                  [-1, 1],
                  [-1, 1]]

        estimator = fitter.BayesianMCMCFitter(likelihood=loss.likelihood, prior=prior,
                                               bounds=bounds, model=physics_model.BeamSpinAsymmetryModel())

        x, y = estimator.samples.shape 
        self.assertTrue(x == estimator.n_iterations)
        self.assertTrue(y == estimator.model.n_pars)


if __name__ == "__main__":
    unittest.main() 
