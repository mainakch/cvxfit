# CvxFit

CvxFit is a package which provides classes for fitting convex functions to given data.

## Usage

Typical usage looks like this:

```
#!/usr/bin/env python

from cvxfit import CvxFit
import scipy as sp

# Generate data.
N = 1000
n = 3

def f_actual(x):
    return sp.sum(x*x)

X = sp.randn(N, n)
Y = sp.array([f_actual(pt) for pt in X])

# Initialize object with 10 affine functions
# with regularization 0.001, and maximum
# number of iterations 20.
fit_object = CvxFit(X=X, Y=Y, type='pwl', extra_param=[10, 0.001, 20])

# Perform fit.
fit_object.fit()

# See training error; repeat fit if high.
print 'Training error: ' + str(fit_object.mean_training_error)

# Compare quality of fit at a random point.
pt = sp.randn(1, n)
print 'Actual value: ' + str(f_actual(pt))
print 'Approximate value: ' + str(fit_object.evaluate(pt)[0])
```

## Authors

This package was originally written and tested by Mainak Chowdhury, Alon Kipnis and Milind Rao.

## Acknowledgements

This package came out of a course project for EE364b at Stanford University, Spring 2014, taught by Prof. Stephen Boyd. We would like to thank all members of the awesome teaching staff for their feedback and suggestions.
