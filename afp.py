#!/usr/bin/env python 

import argparse
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import tqdm

from multiprocessing import Process, Queue
from scipy.optimize import minimize

TO_RADIANS = np.pi/180.0
TO_DEGREES = 1/TO_RADIANS

def bsa_model(phi, pars):
    return pars[0]*np.sin(phi*TO_RADIANS)/(1+pars[1]*np.cos(phi*TO_RADIANS)+pars[2]*np.cos(2*phi*TO_RADIANS))

def generate(phi, pars, err):
    return bsa_model(phi, pars) + np.random.normal(0, err, len(phi))


def create_replica(y, y_err):
    y_rep = [np.random.normal(yp,np.fabs(yp_err)) for yp,yp_err in zip(y,y_err)]
    return np.array(y_rep)

def bootstrap_worker(q, loss_function, physics_model, bounds,
                     phi, data, error, n_replicas=20):

    np.random.seed(os.getpid())

    results = []
    for irep in tqdm.tqdm(range(n_replicas)):
        rep = create_replica(data, error)
        pars,errs = perform_single(loss_function, physics_model, bounds, phi, rep, error)
        results.append(pars)

    q.put(results)

def bootstrap(loss_function, physics_model, bounds,
              phi, data, error, n_replicas=20):

    results = []
    for irep in tqdm.tqdm(range(n_replicas)):
        rep = create_replica(data, error)
        pars,errs = perform_single(loss_function, physics_model, bounds, phi, rep, error)
        results.append(pars)

    return results

def perform_bootstrap(loss_function, physics_model, bounds,
                      phi, data, error, n_replicas=20, n_cores=4):

    if n_cores is 1:
        results = bootstrap(loss_function, physics_model, bounds,
                            phi, data, error, n_replicas)
        results = np.array(results, dtype=np.float32)

    elif n_cores > 1:
        q = Queue()
        workers = []

        # spawn processes
        reps_per_core = int(n_replicas/n_cores)
        for job in range(n_cores):
            workers.append(Process(target=bootstrap_worker, args=(q, loss_function, physics_Model, bounds, phi, data, error, reps_per_core)))
            workers[job].start()

        # get results
        result_pool = []
        for worker in workers:
            rv = q.get()
            result_pool.append(rv)

        # end
        for worker in workers:
            worker.join()

        # aggregate results so they look normal
        results = [item for sublist in result_pool for item in sublist]
        results = np.array(results, dtype=np.float32)

    else:
        print('What are you trying to do asking for %d cores?' % n_cores)

    pars = []
    errs = []
    for ipar in range(3):
        pars.append(np.average(results[:,ipar]))
        errs.append(np.std(results[:,ipar]))

    return pars, errs

def loss_function(data, theory, error):
    return np.sum(((data-theory)/error)**2)

def perform_single(loss_function, model, bounds,
                   phi, data, error):

    func = lambda p: loss_function(data, model(phi, p), error)

    bad_fit = True
    while bad_fit:
        result = minimize(func, x0=np.random.uniform(-1,1,3), bounds=bounds)
        identity = np.identity(3)
        err = np.sqrt(np.array(np.matrix(result.hess_inv * identity).diagonal()))
        bad_fit = not result.success

    return result.x, err[0]

def perform_mcmc():
    basic_model = pm.Model()
    
    with basic_model:
        alpha = pm.Bound(pm.Normal, lower=-1, upper=1)('alpha', mu=0, sd=1)
        beta  = pm.Bound(pm.Normal, lower=-1, upper=1)( 'beta', mu=0, sd=0.05)
        gamma = pm.Bound(pm.Normal, lower=-1, upper=1)('gamma', mu=0, sd=0.05)
        
        mu = alpha * np.sin( (np.pi/180) * test_data.phi) / (1 + \
                                                                 beta * np.cos( (np.pi/180) * test_data.phi) + \
                                                                 gamma * np.cos( 2*(np.pi/180) * test_data.phi)
                                                             )
        
        y = pm.Normal('y', mu=mu, sd=data_errs, observed=d.value)
        trace = pm.sample(1000, tune=500)
        
        pars = []
        errs = []
        pars.append(np.average(trace['alpha']))
        pars.append(np.average(trace['beta']))
        pars.append(np.average(trace['gamma']))
        errs.append(np.std(trace['alpha']))
        errs.append(np.std(trace['beta']))
        errs.append(np.std(trace['gamma']))
        
        params['value'].append(np.array([trace['alpha'],
                                         trace['beta'],
                                         trace['gamma']]))


if __name__ == "__main__":

    phi = np.linspace(-180, 180, 25)
    
    centers = lambda p: np.array([p[i] + 0.5*(p[i+1]-p[i]) for i in range(len(p)-1)], dtype=np.float32)
    phi_center = centers(phi)

    err = 0.015
    gen = generate(phi_center, [0.035, 0.0, 0.0], err)

    df_dict = {}
    df_dict['phi'] = phi_center
    df_dict['value'] = gen
    df_dict['error'] = np.ones(len(phi_center))*err
    df_dict['bin'] = np.repeat(1, len(phi_center))

    df = pd.DataFrame(df_dict)
    df.to_csv('generated.csv', index=False)
    df.to_json('generated.json')


    plt.errorbar(phi_center, gen, np.ones(len(phi_center)) * err)
    plt.show()
