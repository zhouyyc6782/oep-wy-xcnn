import numpy
import scipy.linalg 

from build_matrix import build_vbg


def solve_KS(b, integrals_3c1e_ovlp, h, s, mo_occ):
    """
    Solve the KS equation.

    Args:
        b (ndarray): vbg coefficients.
        integrals_3c1e_ovlp (ndarray): integral table used to build vbg matrix.
        h (ndarray): fixed matrix T + vn + v0.
        s (ndarray): overlap matrix.
        mo_occ (ndarray): occupation number of MOs.

    Returns:
        MO energies;
        MO coefficients;
        density matrix in AO;
        vbg matrix in AO.
    """

    vbg = build_vbg(b, integrals_3c1e_ovlp)

    f = h + vbg
    e, c = scipy.linalg.eigh(f, s)
    
    mocc = c[:, mo_occ>0]
    dm = numpy.dot(mocc * mo_occ[mo_occ>0], mocc.T.conj())

    return e, c, dm, vbg

