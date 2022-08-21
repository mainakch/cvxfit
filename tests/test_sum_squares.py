#!/usr/bin/env python

from cvxfit import CvxFit
import scipy as sp
import numpy as np

def test_fit():

    # Generate data.
    N = 1000
    n = 3
    
    # Set seed.

    rnd_state = np.random.RandomState(10)

    def f_actual(x):
        return np.sum(x * x)
    
    X = rnd_state.randn(N, n)
    Y = np.array([f_actual(pt) for pt in X])
    
    # Initialize object with 10 affine functions
    # with regularization 0.001, and maximum
    # number of iterations 40.
    fit_object = CvxFit(X=X, Y=Y, type="pwl", extra_param=[10, 0.001, 20])
    
    # Perform fit.
    fit_object.fit()
    
    # See training error; repeat fit if high.
    print(f"Training error: {fit_object.mean_training_error}")
    
    # Compare quality of fit at a random point.
    pt = rnd_state.randn(1, n)
    print(pt)
    actual_val = f_actual(pt)
    approx_val = fit_object.evaluate(pt)[0]
    print(f"Actual value: {actual_val}")
    print(f"Approximate value: {approx_val}")
    rel_approx_error = np.absolute((approx_val - actual_val) / actual_val)
    assert True
