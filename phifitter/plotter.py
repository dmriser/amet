import matplotlib.pyplot as plt 
import numpy as np 

def plot_phi(phi, value, error):
    '''
    Create a simple plot of phi
    with values and errorbars.
    '''
    plt.figure( figsize=(16,9) )
    plt.errorbar(x=phi, y=value, yerr=error, 
                 color='black', linestyle='', marker='.')

