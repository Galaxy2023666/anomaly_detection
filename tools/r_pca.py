"""Robust Principal Component Analysis"""

# Code Stable Principal Component Pursuit
# @DLegorreta

import numpy as np
import scipy as sc
import pandas as pd
from numba import double, jit

from statsmodels import robust
from typing import Union
from sklearn.utils.validation import check_array
from statsmodels.tsa.stattools import adfuller

def _check_representation(frequency: int, X: np.ndarray):
    """
    Check if the frequency is integer and the X array is 1D

    Parameters
    ----------

    frequency : int
    X         : numpy array,np.ndarray

    Returns

    X         : numpy array with float values

    """

    X = check_array(X, ensure_2d=False)

    if type(frequency) == float:
        raise TypeError("The frequency must be int, not float")

    if X.ndim > 1:
        raise TypeError('A 1D array was expected, instead a 2D array'
                        ' was obtained')
    return X


def _autodiff(X: np.ndarray):
    """
    Apply Test Dickey-Fuller Test in Time Serie

    Parameters
    ----------
    X  :	numpy array

    Returns
    -------
    X    : 	numpy array
    flag :  Bool

    """
    # Dickey-Fuller Test
    adf = adfuller(X)

    if (adf[1] > 0.05):
        # No Stationary
        flag = True
        X = np.nan_to_num(np.diff(X, prepend=0), nan=0)
        return X, flag
    else:
        # Stationary
        flag = False
        return X, flag


def _scale(scale: bool, X: Union[pd.Series, np.ndarray]):
    """
    Normalization of the Time Serie

    Parameters
    ----------
    scale 	: bool
    X 		: pd.Serie or Numpy Array

    Returns
    -------
    global_mean : float, mean of the Time Serie
    global_sdt  : float, sdt of the Time Serie
    X           : numpy array, 1D Array
    """
    if isinstance(X, pd.Series):
        X = X.values.copy()

    if (scale):
        global_mean = np.mean(X)
        global_sdt = np.std(X)
        X = (X - global_mean) / global_sdt
        return global_mean, global_sdt, X
    else:
        return 0.0, 1.1, X


def _reshape_pad(X: np.ndarray, frequency: int):
    """
    Transform 1d array to 2d array. Check if the frequency is
    divisor of the length of the vector, if not, then fill  in
    the vector  on the left

    Parameters
    ----------

    X 	: 1d numpy array
    frequency : int

    Returns
    -------
    X_out: 2D numpy array

    """
    l = len(X)
    r = l % frequency
    if r == 0:
        return X.reshape((frequency, -1), order='F')
    else:
        diff = frequency - r
        X_out = np.pad(array=X, pad_width=(diff, 0), mode='constant', constant_values=0).reshape((frequency, -1),
                                                                                                 order='F')
        return X_out


def _decision_mad(x, mu, ma):
    """
     Outliers with mad
    """
    if x != 0 and ((abs(x - ma) / mu) > 1.4826):
        return 1
    else:
        return 0


def _mad_outlier(X: np.ndarray):
    """
    Outliers with mad
    """

    L = np.where(X == 0, np.NAN, X).copy()
    L = L[~np.isnan(L)]
    L = np.abs(L)
    mad = robust.mad(L)
    median = np.nanmedian(L)
    Output = np.apply_along_axis(_decision_mad, axis=0, arr=X.reshape(1, -1), mu=mad, ma=median)
    return Output

@jit('f8[:](f8[:],f4)')
def Dsoft(M, penalty):
    """ Inverts the singular values
    takes advantage of the fact that singular values are never negative

    Parameters
    ----------
    M : numpy array
    penalty: float number
             penalty scalar to penalize singular values

    Returns
    -------
    M matrix with elements penalized when them viole this condition
    """
    penalty = float(penalty)
    Out = np.maximum((M - penalty), 0.0)
    return Out


@jit
def SVT(M, penalty):
    """
    Singular Value Thresholding on a numeric matrix

    Parameters
    ----------
    M : numpy array
    penalty: float number
             penalty scalar to penalize singular values

    Returns
    -------
    S: numpy array
       The singular value thresholded matrix  its
    Ds: numpy array
       Thresholded singular values
    """
    penalty = float(penalty)
    U, s, V = np.linalg.svd(M, full_matrices=False)
    Ds = Dsoft(s, penalty)
    # L=np.dot(U,np.diag(Ds))
    S = np.dot(np.dot(U, np.diag(Ds)), V)
    return S, Ds


@jit(forceobj=True)
def SoftThresholdMatrix(x, penalty):
    """sign(x) * pmax(abs(x) - penalty,0)
    """
    x = np.sign(x) * np.maximum(np.abs(x) - penalty, 0)
    return x


@jit(forceobj=True)
def getDynamicMu(X):
    # X=np.array(X).astype('float64')
    m, n = X.shape
    E_sd = np.std(X)
    mu = E_sd * np.sqrt(2 * np.maximum(m, n))
    return np.maximum(mu, 0.001)


@jit(forceobj=True)
def getL(X, S, mu, L_penalty):
    mu = float(mu)
    L_penalty = float(L_penalty)
    L_penalty2 = L_penalty * mu
    C = np.subtract(X, S, out=None)  # cambio
    L0, L1 = SVT(C, L_penalty)
    L_nuclearnorm = np.sum(L1)
    return L0, L_penalty2 * L_nuclearnorm


@jit(forceobj=True)
def getS(X, L, mu, s_penalty):
    mu = float(mu)
    s_penalty = float(s_penalty)
    s_penalty2 = s_penalty * mu
    C = np.subtract(X, L, out=None)  # Cambio
    S = SoftThresholdMatrix(C, s_penalty2)
    S_l1norm = np.sum(np.abs(S))
    return S, s_penalty2 * S_l1norm


@jit(forceobj=True)
def getE(X, L, S):
    R = X.copy()
    np.subtract(X, L, out=R)
    E = np.subtract(R, S, out=None)
    return E, np.linalg.norm(E, 'fro')


@jit(forceobj=True)
def objective(L, S, E):
    return (0.5 * E) + L + S


def RPCA(X, lpenalty=-1, spenalty=-1, verbose=True):
    """
    Robust Principal Component Pursuit.

    Parameters
    ----------
    X : numpy array
    lpenalty: float, default -1
    spenalty: float, default -1
              Scalar to penalize remainder matrix to find Anomalous Values or Noise Values
    verbose:  bool, optional (default=False)
              Controls the verbosity of the matrix building process.

    Returns:
    --------
    X : numpy array original (DEPRECATED)
    L_matrix : numpy array, L_matrix is low rank
    S_matrix : numpy array, S_matrix is sparse
    E_matrix : numpy array, E_matrix is the remainder matrix of noise

    Reference:
    ----------
    Stable Principal Component Pursuit
    Zihan Zhou, Xiaodong Li, John Wright, Emmanuel Candes, Yi Ma
    https://arxiv.org/pdf/1001.2363.pdf
    """

    X = np.array(X).astype('float64').copy()

    m, n = X.shape

    # Penality parameters
    if lpenalty == -1:
        lpenalty = (1.4) / np.sqrt(min(n, m))

    if spenalty == -1:
        spenalty = (1.4) / np.sqrt(max(n, m))

    # Convergence condition
    itere = 1
    maxIter = 2000
    converged = False
    obj_prev = 0.5 * np.linalg.norm(X, 'fro')
    tol = (1e-8) * obj_prev
    diff = 2 * tol
    mu = (X.size) / (4 * np.linalg.norm(X, 1))

    # Initialization
    L_matrix = np.zeros_like(X, dtype='float')
    S_matrix = np.zeros_like(X, dtype='float')
    E_matrix = np.zeros_like(X, dtype='float')

    # Optimization
    while itere < maxIter and diff > tol:

        S_matrix, S_1 = getS(X, L_matrix, mu, spenalty)

        L_matrix, L_1 = getL(X, S_matrix, mu, lpenalty)

        E_matrix, E_1 = getE(X, L_matrix, S_matrix)

        obj = objective(L_1, S_1, E_1)

        if verbose:
            print("Objective function: %4.8f  on previous iteration %d " % (obj_prev, itere - 1))
            print("Objective function: %4.8f  on iteration %d " % (obj, itere))

            diff = np.abs(obj_prev - obj)
            obj_prev = obj
            mu = getDynamicMu(E_matrix)

            itere += 1
            # print( "Diff Value:%2.10f and tol value %2.10f"%(diff,tol))
            if (diff < tol):
                converged = True
                break
        else:
            diff = np.abs(obj_prev - obj)
            obj_prev = obj
            mu = getDynamicMu(E_matrix)
            itere += 1
            if diff < tol:
                converged = True
                break

    if converged:
        # print("Converged within %d iterations"% itere)
        return L_matrix, S_matrix, E_matrix
    else:
        raise ValueError('Matrix Values do not converge,failed to converge within'
                         '{0} maxIter iterations'.format(maxIter))