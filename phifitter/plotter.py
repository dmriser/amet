import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

def plot_phi(phi, value, error):
    '''
    Plot phi distribution
    with values and errorbars.
    '''
    plt.figure( figsize=(4,3) )
    plt.errorbar(x=phi, y=value, yerr=error, 
                 color='black', linestyle='', marker='.')
    plt.axhline(0.0, color='black', linestyle='--', 
                alpha=0.8, linewidth=1)

    plt.xlabel('$\phi$')

    vmin = np.min(value) - np.average(error)
    vmax = np.max(value) + np.average(error)

    if vmin < 0:
        vmin += 0.2*vmin 
    else:
        vmin -= 0.2*vmin 

    if vmax > 0:
        vmax += 0.4*vmax 
    else:
        vmax -= 0.4*vmax 

    plt.ylim([vmin,vmax])

def plot_single_fit(phi, value, error, model, parameters=None):

    if parameters is not None:
        model.pars = parameters

    plot_phi(phi, value, error)

    phi_axis = np.linspace(-180, 180, 100)
    plt.plot(phi_axis, model.evaluate(phi_axis), 
             marker='', linestyle='-', alpha=0.8, color='red')

    
def plot_replicas(phi, value, error, model, replicas):
    '''
    Use the replica values to draw lines of the fitter results.
    '''
    
    plot_phi(phi, value, error)

    phi_axis = np.linspace(-180, 180, 100)
    for rep in replicas:
        plt.plot(phi_axis, model.update_and_evaluate(phi_axis, rep), 
                 marker='', linestyle='-', alpha=0.02, color='red')


def plot_parameter_histogram(parameters, bins=40):

    '''
    parameters should be a numpy array 
    with shape (n, n_parameters) where n
    is the number of trials/values
    '''

    n_values, n_pars = parameters.shape

    n_col = 3
    n_row = np.ceil(n_pars/n_col) + 1

    plt.figure(figsize=(n_col*4, n_row*3))

    for i in range(n_pars):
        plt.subplot(n_row, n_col, i+1)
        plt.hist(parameters[:,i], bins,
                 histtype='stepfilled', color='red',
                 edgecolor='black', alpha=0.8);

        plt.xlabel('Parameter %d' % i)

    plt.tight_layout() 

def plot_kde(par1, par2,
             label1='Parameter 1',
             label2='Parameter 2'):
    
    sns.kdeplot(par1, par2, shade=True,
                shade_lowest=False, cmap='Reds')
    plt.xlabel(label1)
    plt.ylabel(label2)

def plot_kde_grid(parameters):

    n_values, n_pars = parameters.shape

    n_col = n_pars
    n_row = n_col

    plt.figure(figsize=(n_col*4, n_row*4))

    for i in range(n_pars):
        for j in range(i+1, n_pars):
            plt.subplot(n_row, n_col, j+n_col*i + 1)
            plot_kde(parameters[:,i],
                     parameters[:,j],
                     'Parameter %d' % i,
                     'Parameter %d' % j)

    plt.tight_layout() 



    
