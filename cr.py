"""
Perform constrained realisations of 1D data.
"""
import numpy as np

def T_signal(Ns):
    """
    Project signal amplitudes to data vector space.
    """
    # Trivial; amplitude of signal in each frequency channel
    return np.eye(Ns)


def T_continuum(Nc, nu):
    """
    Project continuum amplitudes to data vector space.
    """
    # Normalised frequency vector
    x_nu = (nu - nu[0]) / (nu[-1] - nu[0])
    
    # Continuum amplitudes are coefficients of power series
    T = np.zeros((Nc, nu.size))
    for i in range(Nc):
        T[i] = x_nu**float(i)
    return T


def apply_mat(x, nu, Ninv, Skinv):
    """
    Apply matrix operator to solution vector, A.x.
    """
    y = np.zeros(x.size)
    
    # Projection operators
    Nc = x.size - nu.size
    Ts = T_signal(nu.size)
    Tc = T_continuum(Nc, nu)
    
    # Separate signal and continuum solution blocks
    x_s = x[:nu.size]
    x_c = x[nu.size:]
    ##x_c = x
    
    # Apply prior matrix to signal part (in Fourier space)
    y[:nu.size] += np.fft.ifft( np.dot(Skinv, np.fft.fft(x_s)) ).real
    
    # Apply inverse noise weight to projected solution blocks, N^-1 T x
    y_s = np.dot(Ninv, np.dot(Ts.T, x_s))
    y_c = np.dot(Ninv, np.dot(Tc.T, x_c))
    
    # Add results of matrix application to output vector
    y[:nu.size] += np.dot(Ts, y_s)
    y[:nu.size] += np.dot(Ts, y_c)
    y[nu.size:] += np.dot(Tc, y_s)
    y[nu.size:] += np.dot(Tc, y_c)
    ##y += np.dot(Tc, y_c)
    return y

    
def rhs(d, nu, Nc, Ninv, Skinv, sample=False):
    """
    Calculate the RHS of the linear system, b.
    """
    # Projection operators
    Ns = d.size
    Ts = T_signal(d.size)
    Tc = T_continuum(Nc, nu)
    
    # Construct RHS vector
    b0 = np.dot(Ninv, d)
    b_signal = np.dot(Ts, b0)
    b_continuum = np.dot(Tc, b0)
    
    # Construct RHS vector
    b = np.concatenate((b_signal, b_continuum))
    
    # Add random terms (for sampling) if requested
    if sample:
        # Add random term for signal prior, S^-1/2 . omega
        omega_s = np.random.randn(nu.size)
        #b_s = np.dot(np.linalg.cholesky(Skinv), np.fft.fft(omega_s))
        b_s = np.dot(np.sqrt(Skinv), np.fft.fft(omega_s))
        b_s = np.fft.ifft(b_s).real
        # FIXME: Should guarantee real result by construction

        # Add random term for noise, (U^T N^-1 U)^1/2 . omega
        # FIXME: Performing sqrt on N^-1 directly, to avoid issues with mask
        omega_n = np.random.randn(Ns+Nc)
        yn_c = np.dot(np.sqrt(Ninv), Tc.T)
        yn_s = np.dot(np.sqrt(Ninv), Ts.T)

        # Build U^T N^-1 U operator and take sqrt
        mat = np.zeros((Ns+Nc, Ns+Nc))
        mat[:Ns,:Ns] = np.dot(Ts, yn_s)
        mat[Ns:,Ns:] = np.dot(Tc, yn_c)
        mat[:Ns,Ns:] = np.dot(Ts, yn_c)
        mat[Ns:,:Ns] = np.dot(Tc, yn_s)
        b_n = np.dot(mat, omega_n)
    
        # Add random terms 
        b[:Ns] += b_s
        b[:] += b_n
        
    return b


def conj_grad(x_in, nu, b, Ninv, Skinv, tol=1e-6, Niter=200):
    """
    Apply conjugate gradient method to solve linear system.
    """
    # Calculate initial residual and its norm
    x = x_in.copy()
    r = b - apply_mat(x, nu, Ninv, Skinv)
    res = np.dot(r.T, r)
    p = r.copy()
    
    # Iterate CG solver
    i = 0
    while res > tol and i < Niter:
        if i % 100 == 0: print("%3d : %3.3e / %3.3e" % (i, res, tol))
        
        # CG: Calculate correction amplitude, alpha_k
        A_dot_p = apply_mat(p, nu, Ninv, Skinv)
        alpha = res / np.dot(p.T, A_dot_p)
        
        # Calculate next iteration
        x = x + alpha * p
        r = r - alpha * A_dot_p
        
        # Calculate new residual vector
        res_new = np.dot(r.T, r)
        
        # Update conjugate vector and residual
        beta = res_new / res
        res = res_new
        p = r + beta*p
        
        # Increment iteration counter
        i += 1
    print "\tIters: %d" % i
    return x, r


def solve_system(x=None, sample=False, dataset=None, Nc=25):
    """
    
    Parameters
    ----------
    dataset : dict
        Dictionary with keys that contain the following data:
          'nu':     Frequency in each channel.
          'd':      Data vector.
          'w':      Mask vector.
          'Ninv':   Inverse noise covariance matrix.
          'Skinv':  Inverse signal covariance matrix.
    
    Nc : int, optional
        No. of continuum modes to fit. Default: 25.
    """
    
    # Unpack data vectors and covariance matrices
    nu = dataset['nu']
    d = dataset['d']
    w = dataset['w']
    Ninv = dataset['Ninv']
    Skinv = dataset['Skinv']
    
    # Apply mask to Ninv
    wNinv = Ninv.copy()
    wNinv[np.diag_indices(Ninv.shape[0])] = \
                                      wNinv[np.diag_indices(Ninv.shape[0])] * w

    # Calculate RHS
    b = rhs(d*w, nu, Nc, wNinv, Skinv, sample=sample)

    # Initial guess of solution (zero for signal, O(1) for continuum coeffs)
    if x is None:
        x = np.zeros(b.size)
        """
        x[nu.size:] = [
            1.158e+01, -1.009e+01, 1.758e+01, 5.663e+00, -4.031e+00, -7.466e+00, -7.209e+00, -5.396e+00, 
            -3.179e+00, -1.098e+00, 6.267e-01, 1.932e+00, 2.827e+00, 3.359e+00, 3.585e+00, 4.219e-01, 
            4.156e-01, 4.091e-01, 4.025e-01, 3.959e-01, 3.893e-01, 3.827e-01, 3.762e-01, 3.698e-01, 3.635e-01 ]
        """

    # Solve system usign CG search
    x_bf, r = conj_grad(x, nu, b, wNinv, Skinv, Niter=5000)
    print "\tFinal res = %4.4e" % np.dot(r.T, r)
    return x_bf

