#!/usr/bin/env python

from cvxfit import CvxFit
import scipy as sp
import numpy as np

def test_fit():

    # Generate data.
    N = 1000
    n = 3
    
    def f_actual(x):
        return sp.sum(x * x)
    
    X = sp.randn(N, n)
    Y = np.array([f_actual(pt) for pt in X])
    
    # Initialize object with 10 affine functions
    # with regularization 0.001, and maximum
    # number of iterations 20.
    fit_object = CvxFit(X=X, Y=Y, type="pwl", extra_param=[10, 0.001, 20])
    
    # Perform fit.
    fit_object.fit()
    
    # See training error; repeat fit if high.
    print(f"Training error: {fit_object.mean_training_error}")
    
    # Compare quality of fit at a random point.
    pt = sp.randn(1, n)
    print(f"Actual value: {f_actual(pt)}")
    print(f"Approximate value: {fit_object.evaluate(pt)[0]}")
