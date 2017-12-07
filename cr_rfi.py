#!/usr/bin/env python
"""
Apply CR method to real HERA data.
"""
import numpy as np
import pylab as P
import uvtools
from cr import *

np.random.seed(10)

sigma_n0 = 1e-3 # Noise level
sigma_s0 = 1e-3 # Signal amplitude


#-------------------------------------------------------------------------------
# Load data from datafile and rescale/change units as necessary
#-------------------------------------------------------------------------------

# Load data from Aaron's file
dat = np.load("philbull_data.npz")
_d = dat['d']
_w = dat['w']
nu = np.linspace(100., 200., _d.shape[1]) # FIXME: Not the real frequencies!
x = np.linspace(0., 1., nu.size) # Normalised frequency
#times = dat['times']
#nu = dat['fqs'] * 1e3 # Frequency channels, in MHz

"""
# Plot data, incl. RFI mask
P.subplot(111)
uvtools.plot.waterfall(_d*_w, mx=0, drng=4, mode='log')
P.colorbar()
P.show()
exit()
"""

# Choose which 1D slice we want to work with
ID = 2200
d = _d[ID].real
w = _w[ID]
d *= w # Apply mask to raw data vector

"""
# Plot data in 1D slice
P.subplot(111)
P.plot(nu, np.abs(d*w), lw=1.8)
for i in range(w.size):
    if w[i] < 0.9: P.axvline(nu[i], color='k', alpha=0.07, lw=3.)
P.show()
exit()
"""

#-------------------------------------------------------------------------------
# Package data and covariances together into a dictionary
#-------------------------------------------------------------------------------

# Construct noise covariance matrix
sigma_n = sigma_n0 / (1.+x) # FIXME: I just made this up for now
Ninv = np.eye(d.size) / sigma_n**2.

# Construct initial model for signal covariance
pk = lambda k: sigma_s0**2. * np.ones(k.size)
k = np.fft.fftfreq(nu.size, d=(nu[1] - nu[0]))
Sk = np.eye(d.size) * pk(k)
Skinv = np.eye(d.size) / pk(k)
Sk[0,0] = Skinv[0,0] = 0.

# Package data into dictionary
dataset = {
    'nu':       nu,
    'd':        d,
    'w':        w,
    'Ninv':     Ninv,
    'Skinv':    Skinv,
}

#-------------------------------------------------------------------------------
# Solve linear system using CG search
#-------------------------------------------------------------------------------

# Solve linear system once (with one random realisation)
x1 = solve_system(sample=True, dataset=dataset, Nc=35) # 25

# List for collecting CR results
x_results = [x1,]

#-------------------------------------------------------------------------------
# Reconstruct CR solution and plot
#-------------------------------------------------------------------------------

# Reconstruct Gaussian and continuum signals from solution vector, x1
ys1 = np.dot(x1[:nu.size], T_signal(nu.size))
yc1 = np.dot(x1[nu.size:], T_continuum(x1.size - nu.size, nu))

# Plot to compare solution with input data vector
P.subplot(121)
P.plot(nu, d, 'k-', lw=1.8, alpha=1.)
P.plot(nu, yc1 + np.mean(ys1), 'g-', lw=1.8)
P.plot(nu, yc1 + ys1, 'r-', lw=1.8, alpha=1.)

# Plot masked channels
for i in range(w.size):
    if w[i] < 0.9: P.axvline(nu[i], color='k', alpha=0.07, lw=3.)

P.ylim((-0.2, 0.05))
P.xlabel(r"$\nu$ $[\rm MHz]$", fontsize=18)
P.ylabel(r"${\rm Re}[V(\nu, \tau)]$", fontsize=18)

#-------------------------------------------------------------------------------
# Construct PSDs with and without CR solution and compare
#-------------------------------------------------------------------------------
# Compare delay spectrum with and without constrained realisation
# The main effect of the RFI flagging window function can be seen in the delay 
# spectrum (c.f. Parsons & Backer 2009, Fig. 2). Using the constrained 
# realisation code should hopefully bring it much closer to the true spectrum.

# Define Fourier wavenumbers in delay space
ktau = np.fft.fftfreq(nu.size, d=nu[1]-nu[0])
idxs = np.argsort(ktau)

# Get delay spectra (FTs of frequency spectra)
#ds_true = np.fft.fft(d)
ds_rfi = np.fft.fft(d*w)
ds_cr = np.fft.fft(yc1 + ys1)
ds_cr_gauss = np.fft.fft(ys1)

# Calculate PSDs
#psd_true = ds_true * ds_true.conj()
psd_rfi = ds_rfi * ds_rfi.conj()
psd_cr = ds_cr * ds_cr.conj()
psd_cr_gauss = ds_cr_gauss * ds_cr_gauss.conj()

# Plot power spectra
P.subplot(122)
P.plot(ktau[idxs], psd_rfi[idxs], 'k-', lw=1.8, label="Masked data")
P.plot(ktau[idxs], psd_cr[idxs], 'r-', lw=1.8, label="CR (continuum + gaussian)", alpha=0.6)
P.plot(ktau[idxs], psd_cr_gauss[idxs], 'b-', lw=1.8, label="CR (gaussian)", alpha=0.6)

P.legend(loc='upper right', frameon=True)
P.yscale('log')
P.xscale('log')

P.xlabel(r"$k_\nu$ $[\rm MHz]^{-1}$", fontsize=18)
P.ylabel(r"$P[{\rm Re}(V)]$", fontsize=18)

P.tight_layout()
P.show()
