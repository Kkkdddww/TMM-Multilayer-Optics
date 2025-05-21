import numpy as np

def transfer_matrix(n, d, wavelength) : 
    k0 = 2 * np.pi / wavelength
    delta = k0 * n * d
    Y = n
    M = np.array([ [np.cos(delta), 1j * np.sin(delta)/Y], [1j * Y * np.sin(delta) , np.cos(delta)] ])
    return M


def total_transfer_matrix(layers, wavelength) :
    M_total = np.identity(2, dtype = complex)
    for layer in layers[1:-1] :
        M = transfer_matrix(layer['n'], layer['d'], wavelength)
        M_total = M_total @ M
    return M_total


def compute_rt(M, n0, ns) :
    Y0, Ys = n0, ns
    m11 , m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
    r_num = Y0*m11 + Y0*Ys*m12 - m21 - Ys*m22
    r_den = Y0*m11 + Y0*Ys*m12 + m21 + Ys*m22
    r = r_num / r_den
    t = 2 * Y0 / r_den
    R = np.abs(r)**2
    T = np.abs(t)**2 * np.real(Ys/Y0)
    return R, T
