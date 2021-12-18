import numpy, numpy.linalg


einsum = numpy.einsum


def improve_b_none(b, **kwargs): return b


# Zero-force constrain
def improve_b_zf(b, dm0, Sg, integrals_3c1e_ip_ovlp, force_v0=0.):
    """
    Simplified version of Yair Kurzweil and Martin Head-Gordon, 
    PRA, 80, 012509 (2009) to apply zero-force constrain with 
    respect to target (CCSD) density.

    Args:
        b (ndarray): vbg coefficients at current iteration.
        dm0 (ndarray): target density matrix in AO.
        Sg (ndarray): integral \int g_u(r) g_t(r) dr.
        integrals_3c1e_ip_ovlp (ndarray): 
            integral \int nabla xi_i(r) xi_j(r) g_u(r) dr
        force_v0 (ndarray): force due to v0 and rho0.

    Returns:
        Revised vbg coefficients that meet the zero-force constrains.
    """
    
    nbas_orbital = dm0.shape[0]
    nbas_potential = len(b)
    n = nbas_potential + 3
    
    dm0_I = einsum('ij,aijt->at', dm0, integrals_3c1e_ip_ovlp) 
    dm0_I += einsum('ij,ajit->at', dm0, integrals_3c1e_ip_ovlp) 

    A = numpy.zeros([n, n])
    B = numpy.zeros(n)

    A[:nbas_potential, :nbas_potential] = Sg
    
    A[:nbas_potential, nbas_potential:] = dm0_I.T
    A[nbas_potential:, :nbas_potential] = dm0_I
    # A[nbas_potential:, nbas_potential:] = 0.

    B[:nbas_potential] = einsum('ij,j->i', Sg, b)
    B[nbas_potential:] = -force_v0

    x = numpy.linalg.solve(A, B)
    return x[:nbas_potential]

