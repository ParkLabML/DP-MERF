"""Module containing kernel related classes"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import numpy as np
# import scipy.signal as sig

def mmd2_biased(X, Y, k):
    """
    Implement the quadratic-time biased estimator of MMD^2.
    
    * X: a numpy array such that X[i, ..] is one point. X.shape[0] == m
    * Y: a numpy tensor such that Y[i, ..] is one point. Y.shape[0] == n
    * k: an instance of abcdp.kernel.Kernel on the input points

    Return a scalar representing an estimate of quadratic-time, biased,
    **squared** MMD.
    """
    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)
    mmd2 = np.mean(Kxx) + np.mean(Kyy) - 2.0*np.mean(Kxy)
    return mmd2


class Kernel(object):
    """Abstract class for kernels"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        pass

    def __call__(self, X1, X2):
        return self.eval(X1, X2)

class KLinear(Kernel):
    def eval(self, X1, X2):
        return X1.dot(X2.T)

    def pair_eval(self, X, Y):
        return np.sum(X*Y, 1)

    def __str__(self):
        return "KLinear()"

# end class KLinear

class KGauss(Kernel):

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        D2 = np.sum(X1**2, 1)[:, np.newaxis] - 2.0*X1.dot(X2.T) + np.sum(X2**2, 1)
        K = np.exp(-D2/(2.0*self.sigma2))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(-D2/(2.0*self.sigma2))
        return Kvec

    def __str__(self):
        return "KGauss(%.3f)"%self.sigma2

# end KGauss


