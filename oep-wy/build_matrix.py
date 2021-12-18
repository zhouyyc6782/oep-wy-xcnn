import numpy 
from utility import clean_array


einsum = numpy.einsum


def build_vbg(b, integrals_3c1e):
    """
    vbg_{ij} = \sum_t b_t \int xi_i(r) g_t(r) xi_j(r) dr
    vbg_{ij} = \sum_t b_t I_{ijt}

    Args:
        b (ndarray): potential coefficients.
        integrals_3c1e (ndarray): integral table.

    Returns:
        ndarray: vbg in matrix representation under orbital basis set.
    """
    return einsum('ijt,t->ij', integrals_3c1e, b)        


def build_vbg1(b, integrals_3c1e_ip_ovlp):
    return -einsum('aijt,t->aij', integrals_3c1e_ip_ovlp, b) 

def build_grad(dm, dm_in, integrals_3c1e):
    """
    grad_t = \int (rho(r) - rho_in(r)) g_t(r) dr
    drho(r) = \sum_{ij} dm_{ij} xi_i(r) xi_j(r)
    grad_t = \sum_{ij} dm_{ij} \int xi_i(r) xi_j(r) g_t(r) dr
           = \sum_{ij} dm_{ij} I_{ijt}
    Args:
        dm (ndarray): current density matrix.
        dm_in (ndarray): target density matrix.
        integrals_3c1e (ndarray): integral table.

    Returns:
        ndarray: gradient of Ws.
    """
    g = einsum('ij,ijt->t', dm-dm_in, integrals_3c1e)
    clean_array(g)
    return g


def build_Hess(nbas_orbital, nbas_potential, n_occ, c, e, integrals_3c1e):
    """
    """
    cTs = einsum('ij,jkt->ikt', c.T, integrals_3c1e)
    cTsc = einsum('ijt,jk->ikt', cTs, c)
    
    n = nbas_potential
    H = numpy.zeros([n, n], float)
    for u in range(n):
        for t in range(u+1):
            for i in range(n_occ):
                for a in range(n_occ, nbas_orbital):
                    H[u][t] += cTsc[i][a][u] * cTsc[a][i][t] / (e[i] - e[a])

            if t != u: H[t][u] = H[u][t]

    H *= 4.
    clean_array(H)
    return H


def build_reg_grad(b, Tg, Lambda):
    """
    Lambda regulation term appended to gradient
    """
    reg_grad = einsum('j,ij->i', b, Tg)
    reg_grad *= 4 * Lambda
    return reg_grad


def build_reg_Hess(Tg, Lambda):
    """
    Lambda regulation term appended to Hessian
    """
    reg_Hess = Tg * Lambda * 4
    return reg_Hess

    
# interface
def build_grad_general(wy, idx=None):
    if idx is None:
        return build_grad(wy.dm, wy.dm_in, wy.integrals_3c1e_ovlp)
    else:
        return build_grad(wy.dm, wy.dm_in[idx], wy.integrals_3c1e_ovlp)


def build_grad_with_reg(wy, idx=None):
    if idx is None:
        g = build_grad(wy.dm, wy.dm_in, wy.integrals_3c1e_ovlp)
        reg = build_reg_grad(wy.b_potential, wy.Tg, wy.Lambda)
    else:
        g = build_grad(wy.dm, wy.dm_in[idx], wy.integrals_3c1e_ovlp)
        reg = build_reg_grad(wy.b_potential[idx], wy.Tg, wy.Lambda)
    return g - reg


def build_Hess_general(wy, idx=None):
    if idx is None:
        return build_Hess(
                wy.nbas_orbital, 
                wy.nbas_potential,
                wy.n_occ, 
                wy.c, wy.e,
                wy.integrals_3c1e_ovlp
                )
    else:
        return build_Hess(
                wy.nbas_orbital, 
                wy.nbas_potential,
                wy.n_occ, 
                wy.c[idx], wy.e[idx],
                wy.integrals_3c1e_ovlp
                )


def build_Hess_with_reg(wy, idx=None):
    H = build_Hess_general(wy, idx=idx)
    reg = build_reg_Hess(wy.Tg, wy.Lambda)
    return H - reg

