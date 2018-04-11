#!/usr/bin/env python 

import numpy as np 
import sys, os
sys.path.insert(0, os.path.abspath('../'))

import fitter 
import loss 
import physics_model 

def test_single_fit():
    estimator = fitter.SingleFitter(loss_function=loss.chi2, 
                                    model=physics_model.BeamSpinAsymmetryModel())                          

    # generate a test set 
    phi   = np.linspace(-175.0, 175.0, 17)
    value = estimator.model.evaluate(phi) + np.random.normal(0, 0.05, len(phi)) 
    error = np.repeat(0.05, len(phi))

    # do fitting 
    true_pars = estimator.model.pars 
    estimator.fit(phi, value, error)

    print('Parameters: ', true_pars)
    print('Parameter Est.: ', estimator.fit_parameters)
    print('Error on Pars: ', estimator.fit_errors)
    print('Loss: ', estimator.loss)
    print('Quality: ', estimator.quality)

def test_single_regularized_fit(coef=0.1):
    estimator = fitter.SingleRegularizedFitter(loss_function=loss.chi2, 
                                               model=physics_model.BeamSpinAsymmetryModel(),
                                               penalty=coef)                          

    # generate a test set 
    phi   = np.linspace(-175.0, 175.0, 17)
    value = estimator.model.evaluate(phi) + np.random.normal(0, 0.05, len(phi)) 
    error = np.repeat(0.05, len(phi))

    # do fitting 
    true_pars = estimator.model.pars 
    estimator.fit(phi, value, error)

    print('Parameters: ', true_pars)
    print('Parameter Est.: ', estimator.fit_parameters)
    print('Error on Pars: ', estimator.fit_errors)
    print('Loss: ', estimator.loss)
    print('Quality: ', estimator.quality)

def test_replica_fit(cores=1):
    estimator = fitter.ReplicaFitter(loss_function=loss.chi2, 
                                     model=physics_model.BeamSpinAsymmetryModel(), 
                                     n_cores=cores, n_replicas=20)                          
    
    # generate a test set 
    phi   = np.linspace(-175.0, 175.0, 17)
    value = estimator.model.evaluate(phi) + np.random.normal(0, 0.05, len(phi)) 
    error = np.repeat(0.05, len(phi))

    # do fitting 
    true_pars = estimator.model.pars 
    estimator.fit(phi, value, error)

    print('Parameters: ', true_pars)
    print('Parameter Est.: ', estimator.fit_parameters)
    print('Error on Pars: ', estimator.fit_errors)
    print('Loss: ', estimator.loss)
    print('Quality: ', estimator.quality)
    
def test_bayesian_vegas_fit():

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
    estimator.fit(phi, value, error)

    print('Parameters: ', true_pars)
    print('Parameter Est.: ', estimator.fit_parameters)
    print('Error on Pars: ', estimator.fit_errors)
    print('Loss: ', estimator.loss)
    print('Quality: ', estimator.quality)
    print('Evidence: ', estimator.z)

if __name__ == "__main__":
#    test_single_fit()
#    test_single_regularized_fit(5.0)
#    test_replica_fit()
#    test_replica_fit(4)
    test_bayesian_vegas_fit()
