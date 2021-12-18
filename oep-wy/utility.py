import numpy
from pyscf import scf, cc


einsum = numpy.einsum


def rhf(mol):
    """
    Use PySCF to do RHF calculation.
    Args:
        mol (Mole): PySCF Mole class.

    Returns:
        PySCF RHF object with SCF converged.
    """
    mf = scf.RHF(mol)
    mf.run()
    return mf


def ccsd(mol, mf=None):
    """
    Use PySCF to do CCSD calculation.
    Args:
        mol (Mole): PySCF Mole class.
        mf (RHF, optional): PySCF RHF class.  

    Returns:
        CCSD rdm1 in AO,
        RHF dm in AO.
    """

    if mf is None:
        mf = scf.RHF(mol)
        mf.run()

    c = mf.mo_coeff
    dm = mf.make_rdm1()

    mcc = cc.CCSD(mf)
    ecc, t1, t2 = mcc.kernel()
    rdm1 = mcc.make_rdm1()
    rdm1_ao = einsum('pi,ij,qj->pq', c, rdm1, c.conj())

    return rdm1, dm
     

def rks(mol, xc=None, grid_level=3):
    """
    Use PySCF to do RKS calculation.
    Args:
        mol (Mole): PySCF Mole class.
        xc (str, optional): xc functional.
        grid_level (int, optional): numerical grids level.
    Returns:
        PySCF RKS object with SCF converged.
    """
    mr = scf.RKS(mol)
    if xc is not None:
        mr.xc = xc
    mr.grids.level = grid_level
    mr.kernel()
    return mr


def clean_array(A, crit=1e-16): 
    A[abs(A) < crit] = 0


def dump_matrix(mat, file_path):
    assert isinstance(mat, numpy.ndarray)
    numpy.save(file_path, mat)


def print_mat(mat, name=None):
    if name is None:
        print(mat.shape)
        print(mat)
    else:
        print(name, mat.shape)
        print(mat)

