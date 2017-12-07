#!/usr/bin/python
"""
Plot HERA data.
"""
#from pyuvdata import UVCal, UVData
import numpy as np
import pylab as P
#import aipy, sys, os, glob
import uvtools
#import hera_qm, hera_cal

# Load data from Aaron's file
dat = np.load("philbull_data.npz")
d = dat['d']
w = dat['w']
times = dat['times']
fqs = dat['fqs']

# FIXME: Should do delay filter first! Need 'sdf' array to do that, though.
#d_mdl, _, info = uvtools.dspec.delay_filter(d, w, 200., sdf, tol=1e-9, window='blackman-harris', skip_wgt=.1)


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def estimate_corrmat(d):
    """
    Estimate correlation matrix using FFT.
    """
    # Absolute value corrmat
    y = np.abs(d)
    y -= np.mean(y)
    Fy = np.fft.fft2(y)
    corr_abs = np.fft.ifft2(Fy * Fy.conj())
    
    # Phase correlation matrix
    y = np.angle(d)
    y -= np.mean(y)
    Fy = np.fft.fft2(y)
    corr_phs = np.fft.ifft2(Fy * Fy.conj())
    
    return corr_abs, corr_phs


def fill_linear_interp(d, w, smooth=10):
    """
    Fill missing data using linear interpolation along the frequency direction.
    """
    d_new = d * w #np.abs(d * w)
    #m_new = []
    
    # Loop over times
    for i in range(d.shape[0]):
        
        # Calculate moving average
        #m_avg = moving_average(d_new[i], smooth)
        #m_new.append(m_avg)
        
        # Find continguous masked pixels
        idxs = np.where(w[i] == 0.)[0]
        start = [idxs[0],]; stop = []
        for j in range(1, len(idxs)):
            if (idxs[j] - idxs[j-1]) > 1:
                stop.append(idxs[j-1])
                start.append(idxs[j])
        if len(start) > len(stop):
            stop.append(idxs[-1])
        
        # Linear interpolation across contiguous mask
        for a, b in zip(start, stop):
            if a < 1: continue
            y1 = d_new[i,a-1]
            y2 = y1 if b >= w[i].size-1 else d_new[i,b+1]
            dx = np.arange(b - a + 1) + 1.
            grad = (y2 - y1) / (dx[-1] + 1.)
            d_new[i,a:b+1] = y1 + grad * dx
    return d_new

# Fill the mask with a simple linear interpolation
print "Interpolating..."
d_new = fill_linear_interp(d, w, smooth=10)

# Calculate correlation functions
print "Estimating corrmat..."
corr_abs, corr_phs = estimate_corrmat(d_new)
corr_abs1, corr_phs1 = estimate_corrmat(d*w)

#P.imshow(np.log10(np.fft.fft2(corr_abs.real).real), aspect=0.25)

P.imshow(np.abs(np.fft.fftshift( np.fft.fft2(corr_abs)) ), aspect=0.25, interpolation='none')
P.colorbar()
P.show()
exit()

for i in [0, 10, 20, 30]:
    s1, s2 = corr_abs.shape
    P.plot(corr_abs.real[i,:s2/2], 'k-', lw=2.)
    P.plot(corr_abs1.real[i,:s2/2], 'r--', lw=2.)
    
    P.plot(corr_abs.real[:s1/2,i], 'k-', lw=2.)
    P.plot(corr_abs1.real[:s1/2,i], 'b--', lw=2.)

#P.plot(corr_abs.real[:,0])
#P.plot(corr_abs.real[:,10])
#P.plot(corr_abs.real[:,20])

P.show()
exit()


#for i in [100, 400, 800, 1400, 1800, 2500]:
#    P.plot(np.abs(d*w)[i], 'k.', lw=1.8)
#    P.plot(np.abs(d_new)[i], 'rx', lw=1.8)
#P.tight_layout()
#P.show()
#exit()


# Plot waterfalls
P.subplot(211)
#uvtools.plot.waterfall(d*w, mx=0, drng=4, mode='log')
uvtools.plot.waterfall(corr_abs, mx=0, drng=4, mode='log')
P.colorbar()

P.subplot(212)
uvtools.plot.waterfall(corr_phs, mx=0, drng=4, mode='log')
#uvtools.plot.waterfall(d_new, mx=0, drng=4, mode='log')
P.colorbar()
P.show()


"""
# Choose a frequency direction
j = 3600
s = (d*w)[j]

# FIXME: Estimate correlation function in time and frequency
y = (d*w)[j]
y -= np.mean(y)
Fy = np.fft.fft(y)
corr = np.fft.ifft(Fy * Fy.conj())

y = np.abs((d*w)[j])
y -= np.mean(y)
Fy = np.fft.fft(y)
corr2 = np.fft.ifft(Fy * Fy.conj())


P.plot(corr[:corr.size/2], 'b-', lw=1.8)
P.plot(corr2[:corr2.size/2], 'y--', lw=1.8)
P.show()
"""
exit()

y = np.abs(d*w)

Fy = np.fft.fft2(y)
corr = np.fft.ifft2(Fy*Fy.conj()).real

print corr

P.matshow(np.log10(corr))
P.colorbar()
P.show()

print corr.shape
print d.shape
exit()


# Plot 1D
P.subplot(111)

P.plot(np.abs(s), lw=1.8)
P.plot(np.abs((d*w)[j-2]), lw=1.8)

P.yscale('log')
P.tight_layout()
P.show()




exit()
# Plot waterfalls
P.subplot(221)
uvtools.plot.waterfall(d*w, mx=0, drng=4, mode='log')
P.colorbar()

P.subplot(222)
uvtools.plot.waterfall(d*w, mx=0, drng=4, mode='phs')
P.colorbar()

P.subplot(223)
uvtools.plot.waterfall(np.real(d*w), mx=0, drng=4, mode='log')
P.colorbar()

P.subplot(224)
uvtools.plot.waterfall(np.imag(d*w), mx=0, drng=4, mode='log')
P.colorbar()

P.show()
