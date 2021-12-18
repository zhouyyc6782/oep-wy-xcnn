from __future__ import print_function
import time
import numpy 

from xcnn.utility import numint


## def head_gordon_zf(mol, dm, coords, weights, wxc, f=0.):
##     print("Construct vxc based on ZF constrain...\t", end="")
##     tt = time.time()
## 
##     # electron density on grid points
##     rho = numint.eval_rho_on_grids(mol, dm, coords=coords, deriv=1, xctype="GGA")
## 
##     N = len(coords)
## 
##     AU = rho[3, :]
##     AL = rho[3, :] * weights
## 
##     B = wxc
## 
##     a = -1. / numpy.dot(AL, AU)
##     LB = numpy.dot(AL, B)
## 
##     vxc = B + a * LB * AU + f * a * AU
## 
##     print("time elapses: %.8lf s" % (time.time() - tt))
## 
##     return vxc[:N]

def head_gordon_zf(mol, dm, coords, weights, wxc, f=numpy.zeros([3])):
    print("Construct vxc based on ZF constrain...\t", end="")
    tt = time.time()

    # electron density on grid points
    rho = numint.eval_rho_on_grids(mol, dm, coords=coords, deriv=1, xctype="GGA")

    N = len(coords)
    AU = numpy.zeros([N, 3])
    AL = numpy.zeros([3, N])

    for k in range(3):
        AU[:, k] = rho[k+1, :]
        AL[k, :] = rho[k+1, :] * weights

    B = wxc

    ALAU = numpy.einsum('ij,jk->ik', AL, AU)
    assert(ALAU.shape == (3, 3))
    a = -numpy.linalg.inv(ALAU)

    LB = numpy.einsum('ij,j->i', AL, B)
    aLB = numpy.einsum('ij,j->i', a, LB)
    UaLB = numpy.einsum('ij,j->i', AU, aLB)

    # TODO 
    # check sign of f
    af = numpy.einsum('ij,j->i', a, f)
    Uaf = numpy.einsum('ij,j->i', AU, af)

    vxc = B + UaLB + Uaf

    print("time elapses: %.8lf s" % (time.time() - tt))

    return vxc[:N]

