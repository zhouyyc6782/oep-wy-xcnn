import numpy
import pyscf

from density import _density_on_grids

einsum = numpy.einsum


def dft(mol, mr=None, xc=None, level=3):
    if mr is None:
        mr = pyscf.scf.RKS(mol)
    if xc is not None:
        mr.xc = xc
    mr.grids.level = level
    mr.run()
    return mr.make_rdm1()


def rhf(mol, mf=None):
    if mf is None:
        mf = pyscf.scf.RHF(mol)
    mf.kernel()
    return mf.make_rdm1()


def ccsd(mol, mcc=None, mf=None):
    if mf is None:
        mf = pyscf.scf.RHF(mol)
        mf.kernel()
    c = mf.mo_coeff
    if mcc is None:
        mcc = pyscf.cc.CCSD(mf)
    ecc, t1, t2 = mcc.kernel()
    rdm1 = mcc.make_rdm1()
    rdm1_ao = einsum('pi,ij,qj->pq', c, rdm1, c.conj())
    return rdm1_ao


def print_mat(mat, name=None):
    if name is None:
        print(mat.shape)
        print(mat)
    else:
        print(name, mat.shape)
        print(mat)


def I(mol, dm1, dm2, coords, weights):
    rho1 = _density_on_grids(mol, coords, dm1)
    rho2 = _density_on_grids(mol, coords, dm2)
    drho = rho1 - rho2

    a = numpy.sum(drho * drho * weights) 
    b = numpy.sum(rho1 * rho1 * weights) + numpy.sum(rho2 * rho2 * weights)
    return a / b

