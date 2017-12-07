#!/usr/bin/env python
"""
Extract only specific data from HERA miriad files, and store at numpy arrays.
"""
from pyuvdata import UVCal, UVData
import numpy as np, pylab as P
import aipy, uvtools, hera_cal
import glob

# Pattern to match when loading files
PATTERN = "../hera_data/zen.*.xx.HH.uvOR"
BASELINE = (0,1)

def extract_data(fname, baseline=(0,1), all_redundant=False):
    """
    Extract data for only a certain set of redundant baselines, and save to a 
    numpy file.
    """
    # See which baselines are included in the dataset
    uv = UVData()
    uv.read_miriad(fname) # This is slow

    # Get info about antenna pairs contained in the dataset
    aa = hera_cal.utils.get_aa_from_uv(uv)
    info = hera_cal.omni.aa_to_info(aa, pols='x', tol=1)
    
    # Load data and mask (for whole redundant set, or just the specified baseline)
    if all_redundant:
        # Find set of redundant baselines that contains 'baseline'
        redundant_sets = info.get_reds()
        rset0 = []
        baseset_keys = []
        for rset in redundant_sets:
            if baseline in rset: rset0 = rset
        
        # Loop over baselines in this redundant set, loading their data
        d = []; w = []
        for r in rset0:
            _d = uv.get_data(r[0], r[1], 'xx')
            _w = np.logical_not( uv.get_flags(r[0], r[1], 'xx') )
            d.append(_d); w.append(_w)
    else:
        # Load data for a single baseline
        d = uv.get_data(baseline[0], baseline[1], 'xx')
        w = uv.get_flags(baseline[0], baseline[1], 'xx')
        w = np.logical_not(w)
    return d, w


if __name__ == '__main__':
    # Get list of available files
    files = glob.glob(PATTERN)
    
    # Load data and save each file one by one
    for f in files:
        try:
            d, w = extract_data(f, baseline=BASELINE)
            fname = "%s_%d_%d" % (f, BASELINE[0], BASELINE[1])
            np.savez(fname, d=d, w=w)
            print("Saved to %s.npz" % fname)
        except:
            print("Failed to extract %s" % f)

exit()

def est_noise_covariance(d, w, mask_thres=0.98, outlier_thres=95.):
    """
    Estimate frequency-frequency noise covariance by performing finite 
    differences on the data in the time direction and then averaging.
    
    Trims outliers to obtain more reliable results.
    """
    
    # Build frequency mask by dropping channels that have more than a handful 
    # of masked values in the time direction
    #w_freq = np.mean(w, axis=0)
    #w_freq /= np.max(w_freq)
    #w_freq[w_freq < mask_thres] = 0.
    #w_freq[w_freq >= mask_thres] = 1.

    # Create masked array
    #mask = np.logical_not( w.astype(bool) )
    mask = np.logical_not( w.astype(bool) )
    d_masked = np.ma.array(d, mask=mask)
    
    # Difference masked array in the time direction. Assuming that the signal 
    # is almost constant between neighbouring time samples, these values should 
    # essentially be the difference of the noise in neighbouring time samples
    diff_d = np.ma.diff(d_masked, axis=0)
    #sigma = np.std(diff_d, axis=0)
    idxs = np.where(np.abs(diff_d) < 0.1)
    sigma = np.std(diff_d[idxs].real)
    
    idxs = np.where(np.abs(diff_d.imag) < 0.1)
    sigma_i = np.std(diff_d[idxs].imag)
    print diff_d
    print sigma, sigma_i
    
    P.hist(diff_d.flatten(), bins=100, range=(-0.2, 0.2), normed=True)
    xx = np.linspace(-0.1, 0.1, 500)
    P.plot(xx, np.exp(-0.5*(xx/sigma)**2.)/np.sqrt(2.*np.pi)/sigma, 'r-', lw=1.8)
    P.show()
    
    return sigma, np.std(diff_d.real, axis=0), np.std(diff_d.imag, axis=0)
    
    ###########################
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

"""
# Fit a polynomial to the moving average
x = np.linspace(0., 1., sqrtNr.size)
avg_r = moving_average(sqrtNr, n=50)
avg_r2 = sqrtNr

# Clean up zeros and NaN values, use those to construct weights
ww = np.ones(x.size)
ww[np.where(np.logical_or(avg_r == 0., np.isnan(avg_r)))] = 0.
avg_r[np.where(np.isnan(avg_r))] = 0.

ww2 = np.ones(x.size)
ww2[np.where(np.logical_or(avg_r2 == 0., np.isnan(avg_r2)))] = 0.
avg_r2[np.where(np.isnan(avg_r2))] = 0.

# Fit polynomial
p = np.polyfit(x, avg_r, deg=18, w=ww)
p2 = np.polyfit(x, avg_r2, deg=18, w=ww2)
ch = np.polynomial.chebyshev.chebfit(x, avg_r2, deg=18, w=ww2)
print p
"""
