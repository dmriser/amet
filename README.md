# Phi-Fitter 
This package contains various different parameter estimation
methods. 

### Setup 
This section will explain how to setup this package.

### Currently Implemented Methods 
- Single fit 
- Regularized (L2) single fit 
- Replica fit
- Bayesian Analysis (direct integral using VEGAS)
- Bayesian Analysis (Markov-Chain MC, Metropolis Hastings).  This method needs autocorrelation check function.

### Inputing Data
This API can be used with any data you can get into a numpy array.  If you want to use the built-in reader your data should be placed into a csv file with the following columns.
- `bin_index` - The integer index of the current bin, if you're fitting just one bin use any number (0) that is the same for all entries.
- `value` - Observed value of asymmetry or counts.
- `error` - The error of the asymmetry or counts, this can come from statistical uncertainty or can include systematic errors as well.
- `phi` - The numerical (float) value of $\phi$ in degrees.


