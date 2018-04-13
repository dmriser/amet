import matplotlib.pyplot as plt 
import numpy as np 

def plot_phi(phi, value, error):
    '''

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

def plot_replicas(phi, value, error, model, replicas):
    '''
    Use the replica values to draw lines of the fitter results.
    '''
    
    plot_phi(phi, value, error)

    phi_axis = np.linspace(-180, 180, 100)
    for rep in replicas:
        plt.plot(phi_axis, model.update_and_evaluate(phi_axis, rep), 
                 marker='', linestyle='-', alpha=0.02, color='red')

