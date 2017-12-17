import sys
import matplotlib.pyplot as plt
import numpy as np

from wavelet_inpaint_denoise.core import WaveletCoeffs, dwt2
from wavelet_inpaint_denoise.masks import bsplinelinear


# Proximal operator for the 1-norm.

def prox(wcoeffs, thresh): 
    if not isinstance(wcoeffs, WaveletCoeffs):
        raise TypeError('First argument must be of type waveletCoeffs')
    temp = WaveletCoeffs(wcoeffs.masks, wcoeffs.levels, wcoeffs.sizes[0])
    for lev in range(1, wcoeffs.levels + 1):
        thresh_mask = (np.abs(wcoeffs[lev]) >
                       thresh[lev - 1]).astype(np.float64)
        temp[lev] = (wcoeffs[lev] - thresh_mask *
                     np.sign(wcoeffs[lev]) * thresh[lev - 1]) * thresh_mask
    return temp


def apgd(f, A, thresh=[0.1, 0.07, 0.04, 0.01], masks=bsplinelinear, levels=4, iters=20, verbose=True, showiters=False):
    # APGD algorithm
    if verbose:
        sys.stdout.write("Running the APGD algorithm:\n")
    x_k = dwt2(f, masks, levels)  # x_k = x_0
    x_km1 = x_k  # x_{k-1}=x_{-1} = x_0
    t_k = 1
    t_km1 = 0
    n = levels
    h = np.array(masks)
    # in general, this has to be the maximum singular value of A. For
    # projections (inpainting) this is 1
    L = 1.0
    thresh = np.array(thresh)
    for k in range(1, iters + 1):
        if verbose:
            sys.stdout.write('\rIteration ' + str(k) + ' of ' + str(iters))
            sys.stdout.flush()
        y_k = x_k - ((x_k - x_km1) * (np.ones(n + 1) * (t_km1 - 1) / t_k))
        g_k = y_k - (dwt2(A * (A * y_k.invdwt2() - f), h, n)
                     * np.ones(n + 1) / L)
        x_kp1 = prox(g_k, thresh / L)
        t_kp1 = (1 + np.sqrt(1 + 4 * (t_k**2))) / 2

        x_km1 = x_k
        x_k = x_kp1
        t_km1 = t_k
        t_k = t_kp1
        if showiters:
            plt.clf()
            disp = x_k.invdwt2()
            disp[disp < 0] *= 0
            plt.imshow(disp, cmap=plt.cm.gray)
    if verbose:
        sys.stdout.write('\nDone\n')
    return x_k.invdwt2()
