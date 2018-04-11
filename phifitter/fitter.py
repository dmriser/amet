import numpy as np 
import os 
import pandas as pd 
import tqdm 
import vegas

from multiprocessing import Process, Queue 
from scipy.optimize import minimize 
from scipy.stats import chi2 as chi2pdf

# from this project 
import loss 
import physics_model
import utils 

class BaseFitter(object):
    '''
    BaseFitter: 
    -----------
    This class defines the basic behaviour of the 
    fitting methods. 
    '''

    def __init__(self):
        self.bounds         = None 
        self.model          = None 
        self.loss_function  = None 
        self.fit_parameters = None 
        self.fit_errors     = None 
        self.loss           = None 
        self.quality        = None 

    def fit(self, phi, value, error):
        pass 


class SingleRegularizedFitter(BaseFitter):
    '''
    SingleRegularizedFitter is the basic fit 
    where the scipy.optimize.minimize 
    method is used to find the minimum. 
    
    In this case we add a penalty of 
    penalty * np.sum( pars**2 ).  This forces 
    the parameters to be small. 

    This is called L2 regularization 
    in machine learning.
    '''

    def __init__(self, model, loss_function, bounds=None, penalty=1.0):
        self.model = model
        self.loss_function = loss_function 
        self.bounds = bounds 
        self.penalty = penalty 

    def fit(self, phi, value, error):
        
        func = lambda p: self.loss_function(value, self.model.update_and_evaluate(phi, p), error) + self.penalty * np.sum(p**2)

        bad_fit = True
        while bad_fit:
            if self.bounds is not None:
                result = minimize(func, x0=self.model.pars, bounds=self.bounds)
                identity = np.identity(self.model.n_pars)
                err = np.sqrt(np.array(np.matrix(result.hess_inv * identity).diagonal()))
                err = err[0]
            else:
                result = minimize(func, x0=self.model.pars)
                err = np.sqrt(result.hess_inv.diagonal())

            bad_fit = not result.success

        self.fit_parameters = result.x 
        self.fit_errors = err
        self.loss = self.loss_function(value, 
                                       self.model.update_and_evaluate(phi, self.fit_parameters), 
                                       error)

        pred = self.model.update_and_evaluate(phi, self.fit_parameters)
        ndf = len(phi)
        self.quality = 1-chi2pdf.cdf(loss.chi2(value, pred, error), ndf) 

class SingleFitter(BaseFitter):
    '''
    SingleFitter is the basic fit 
    where the scipy.optimize.minimize 
    method is used to find the minimum. 
    '''

    def __init__(self, model, loss_function, bounds=None):
        self.model = model
        self.loss_function = loss_function 
        self.bounds = bounds 

    def fit(self, phi, value, error):
        
        func = lambda p: self.loss_function(value, self.model.update_and_evaluate(phi, p), error)

        bad_fit = True
        while bad_fit:
            if self.bounds is not None:
                result = minimize(func, x0=self.model.pars, bounds=self.bounds)
                identity = np.identity(self.model.n_pars)
                err = np.sqrt(np.array(np.matrix(result.hess_inv * identity).diagonal()))
                err = err[0]
            else:
                result = minimize(func, x0=self.model.pars)
                err = np.sqrt(result.hess_inv.diagonal())

            bad_fit = not result.success

        self.fit_parameters = result.x 
        self.fit_errors = err
        self.loss = self.loss_function(value, 
                                       self.model.update_and_evaluate(phi, self.fit_parameters), 
                                       error)

        pred = self.model.update_and_evaluate(phi, self.fit_parameters)
        ndf = len(phi)
        self.quality = 1-chi2pdf.cdf(loss.chi2(value, pred, error), ndf) 
        
class ReplicaFitter(BaseFitter):
    '''
    ReplicaFitter runs n_replicas single fits on 
    data that is randomly generated based on the 
    data provided and the errors provided. 
    '''

    def __init__(self, model, loss_function, bounds=None, 
                 n_replicas=100, n_cores=1):
        self.model = model 
        self.loss_function = loss_function 
        self.bounds = bounds 
        self.n_replicas = n_replicas 
        self.n_cores = n_cores 
        self.fit_container = []

        self.fit_parameters = np.zeros(self.model.n_pars)
        self.fit_errors = np.zeros(self.model.n_pars)

        # the single fitter is used for every fit 
        self.single_fitter = SingleFitter(self.model, self.loss_function, self.bounds)

    def fit(self, phi, value, error):
        
        # start with a fresh empty container 
        self.fit_container = [] 

        # run single fit 
        if self.n_cores is 1:
            results = self._run_single(phi, value, error)
            results = np.array(results, dtype=np.float32)

        # do multi core fitting 
        elif self.n_cores > 1:
            q = Queue()
            workers = []

            # spawn processes
            reps_per_core = int(self.n_replicas/self.n_cores)
            reps_to_give = np.repeat(reps_per_core, self.n_cores)
            reps_to_give[-1] += self.n_replicas - np.sum(reps_to_give)

            for job in range(self.n_cores):
                workers.append(Process(target=self._mp_worker, args=(q, phi, value, error, reps_to_give[job])))
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
            self.fit_container = results 

        else:
            print('What are you trying to do asking for %d cores?' % n_cores)


        # setup results 
        self._set_results(phi, value, error) 

    def _set_results(self, phi, value, error):

        self.fit_container = np.array(self.fit_container, dtype=np.float32)
        
        for ipar in range(self.model.n_pars):
            self.fit_parameters[ipar] = np.average(self.fit_container[:,ipar])
            self.fit_errors[ipar] = np.std(self.fit_container[:,ipar])

        self.loss = self.loss_function(value,
                                       self.model.update_and_evaluate(phi, self.fit_parameters),
                                       error)

        pred = self.model.update_and_evaluate(phi, self.fit_parameters)
        ndf = len(phi)
        self.quality = 1-chi2pdf.cdf(loss.chi2(value, pred, error), ndf)

    def _mp_worker(self, q, phi, value, error, reps):
        np.random.seed(os.getpid())

        results = []
        for irep in tqdm.tqdm(range(reps)):
            rep = utils.create_replica(value, error)
            self.single_fitter.fit(phi, rep, error)
            results.append(self.single_fitter.fit_parameters)
        
        q.put(results)

    def _run_single(self, phi, value, error):
        
        for irep in tqdm.tqdm(range(self.n_replicas)):
            rep = utils.create_replica(value, error)
            self.single_fitter.fit(phi, rep, error)
            self.fit_container.append(self.single_fitter.fit_parameters)
            
class BayesianVegasFitter(BaseFitter):
    '''
    BayesianVegasFitter uses Vegas iterative MC 
    integration to perform the integral over the 
    likelihood function.
    '''

    def __init__(self, model, likelihood, prior, bounds, 
                 n_iterations=12, n_evaluations=2000):
        self.model = model
        self.likelihood = likelihood
        self.prior = prior 
        self.bounds = bounds 
        self.z = 0.0
        self.n_iterations = n_iterations 
        self.n_evaluations = n_evaluations 

    def fit(self, phi, value, error):
        vegas_integrator = vegas.Integrator(self.bounds)

        # This call starts shaping the g(x) function which is used to 
        # generate samples for the integral.  The result is a better 
        # starting shape and a more accurate representation of the 
        # integral quality.
        vegas_integrator(lambda p: self._integrand(phi, value, error, p),
                         nitn=4,
                         neval=1000)

        result = vegas_integrator(lambda p: self._integrand(phi, value, error, p),
                                  nitn=self.n_iterations,
                                  neval=self.n_evaluations)

        self._set_result(result, phi, value, error)


    def _set_result(self, result, phi, value, error):
        self.z = result[0].mean 
        self.quality = result.Q 
        self.fit_parameters = np.array([result[1].mean/self.z, 
                                       result[2].mean/self.z, 
                                       result[3].mean/self.z])

        self.fit_errors = np.array([result[4].mean/self.z - self.fit_parameters[0]**2, 
                                   result[5].mean/self.z - self.fit_parameters[1]**2,
                                   result[6].mean/self.z - self.fit_parameters[2]**2])

        self.fit_errors = np.sqrt(self.fit_errors)
        self.loss = loss.chi2(value, 
                              self.model.update_and_evaluate(phi, self.fit_parameters),
                              error)

    def _integrand(self, phi, value, error, p):
        theory = self.model.update_and_evaluate(phi, p)
        f = self.likelihood(value, theory, error)*self.prior(p)
        return [f, f*p[0], f*p[1], f*p[2],
                f*p[0]**2, f*p[1]**2, f*p[2]**2]


