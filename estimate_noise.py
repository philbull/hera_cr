#!/usr/bin/env python
"""
Estimate frequency-frequency noise covariance matrix.
"""
import numpy as np
import pylab as P
import uvtools

#-------------------------------------------------------------------------------
# Load data from datafile and rescale/change units as necessary
#-------------------------------------------------------------------------------

# Load data from Aaron's file
dat = np.load("philbull_data.npz")
d = dat['d']
w = dat['w']
#nu = np.linspace(100., 200., _d.shape[1]) # FIXME: Not the real frequencies!
#x = np.linspace(0., 1., nu.size) # Normalised frequency
#times = dat['times']
#nu = dat['fqs'] * 1e3 # Frequency channels, in MHz


"""
for i in range(10):
    try:
        h, edges = np.histogram(d_masked[:,170+i*40], bins=500)
        tc = 0.5 * (edges[:-1] + edges[1:])
        cdf = np.cumsum(h.astype(float))
        cdf /= cdf[-1]

        #P.hist(d_masked[:,175], bins=400, alpha=0.3, range=(-4., 3.), normed=True)
        #P.hist(d_masked[:,275], bins=400, alpha=0.3, range=(-4., 3.), normed=True)
        P.plot(tc, cdf)
        #P.plot(tc[1:], np.diff(cdf))
    except:
        pass
#P.yscale('log')
P.show()
exit()
"""


def est_noise_covariance(d, w, mask_thres=0.98, outlier_thres=95.):
    """
    Estimate frequency-frequency noise covariance by performing finite 
    differences on the data in the time direction and then averaging.
    
    Trims outliers to obtain more reliable results.
    """
    
    # Build frequency mask by dropping channels that have more than a handful 
    # of masked values in the time direction
    w_freq = np.mean(w, axis=0)
    w_freq /= np.max(w_freq)
    w_freq[w_freq < mask_thres] = 0.
    w_freq[w_freq >= mask_thres] = 1.

    # Create masked array
    mask = np.logical_not( w.astype(bool) )
    d_masked = np.ma.array(d.real, mask=mask)
    
    # Difference masked array in the time direction. Assuming that the signal 
    # is almost constant between neighbouring time samples, these values should 
    # essentially be the difference of the noise in neighbouring time samples
    diff_d = np.ma.diff(d_masked, axis=0)
    
    # Trim outliers in each frequency channel to allow better variance estimate
    # (Noise field should be mean-zero, but isn't in practise.)
    percentile = np.nanpercentile(np.abs(diff_d), outlier_thres, axis=0)
    msk = np.zeros(diff_d.shape)
    for i in range(percentile.size):
        idx = np.where(np.abs(diff_d[i]) > percentile[i])
        msk[i][idx] = 1. # FIXME: True or False?
    diff_d_mask = np.ma.array(diff_d, mask=msk.astype(bool))
    std = np.ma.std(diff_d_mask, axis=0)
    return std, percentile


std, pct = est_noise_covariance(d, w, mask_thres=0.98, outlier_thres=99.99)

# Create masked array
mask = np.logical_not( w.astype(bool) )
d_masked = np.ma.array(d, mask=mask)
stdx = np.std(d*w, axis=0) #np.ma.std(d_masked.real, axis=0)
print stdx.size

j = 475
P.hist(d_masked[:,j], bins=1000, alpha=0.3, range=(-4., 3.), normed=True)
P.axvline(pct[j], color='r', lw=1.8)
P.axvline(-pct[j], color='r', lw=1.8)
x = np.linspace(-4., 3., 5000)
P.plot(x, np.exp(-0.5*(x/std[j])**2.)/np.sqrt(2.*np.pi*std[j]**2.), 'b-', lw=1.8)
P.plot(x, np.exp(-0.5*(x/stdx[j])**2.)/np.sqrt(2.*np.pi*stdx[j]**2.), 'g-', lw=1.8)
P.xlim(-2.5, 1.5)
P.ylim(-0.5, 6.)
P.show()
exit()

# Plot data, incl. RFI mask
P.subplot(121)
uvtools.plot.waterfall(d*w, mx=0, drng=4, mode='log')
P.colorbar()

w_freq = np.mean(w, axis=0)
w_freq /= np.max(w_freq)
print np.min(w_freq), np.max(w_freq)

thres = 0.98
w_freq[w_freq < thres] = np.nan
w_freq[w_freq >= thres] = 1.

# Create masked array
mask = np.logical_not( w.astype(bool) )
d_masked = np.ma.array(d, mask=mask)
diff_d = np.ma.diff(d_masked, axis=0)

P.subplot(122)
P.plot(np.ma.mean(diff_d, axis=0).real, 'b-')
P.plot(np.ma.std(diff_d, axis=0).real, 'k-', lw=1.8, alpha=0.2)
P.plot(np.ma.std(diff_d, axis=0).real*w_freq, 'r-', lw=1.8)

P.plot(std, 'y-', lw=1.8)

#P.plot(w_freq)
P.show()
exit()

#P.hist(diff_d.flatten())
#P.show()
#exit()

print diff_d.shape, d_masked.shape
P.subplot(111)
uvtools.plot.waterfall(diff_d, mx=0, drng=4, mode='log')
P.colorbar()
P.show()
