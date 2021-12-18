from __future__ import division
import numpy 
from pyscf import dft


BLK_SIZE = 200
einsum = numpy.einsum
XCTYPE = {
        0: 'lda',
        1: 'gga',
        }


def _density_on_grids(mol, coords, dm):
    ao_values = mol.eval_gto("GTOval_sph", coords)
    rho = einsum('ij,ai,aj->a', dm, ao_values, ao_values)
    return rho
    # END of _density_on_grids()


def _density_on_grids2(mol, coords, dm, deriv=0):
    ao_values = dft.numint.eval_ao(mol, coords, deriv=deriv)
    rho = dft.numint.eval_rho(mol, ao_values, dm, xctype=XCTYPE[deriv])
    return rho
    # END of _density_on_grids2()


def calc_real_space_density(mol, coords, dm):
    total_size = len(coords)
    n_blk = total_size // BLK_SIZE
    res = total_size - n_blk * BLK_SIZE

    d = numpy.zeros(total_size, dtype=float)
    for i in range(n_blk):
        index = slice(i*BLK_SIZE, (i+1)*BLK_SIZE) 
        d[index] = _density_on_grids(mol, coords[index], dm)
    if res > 0:
        d[-res:] = _density_on_grids(mol, coords[-res:], dm)
    
    return d
    # END of calc_real_space_density()


def calc_density_cube(wy, outfile='./rho_oep.cube', mode='diff', dm=None):
    from pyscf.tools import cubegen
    if dm is not None:
        cubegen.density(wy.mol, outfile, dm)
    else:
        if mode.lower() == 'diff': 
            cubegen.density(wy.mol, outfile, wy.dm_out-wy.dm_in_ccsd)
        elif mode.lower() == 'ccsd':
            cubegen.density(wy.mol, outfile, wy.dm_in_ccsd)
        elif mode.lower() == 'oep':
            cubegen.density(wy.mol, outfile, wy.dm_out)
        elif mode.lower() == 'rks':
            cubegen.density(wy.mol, outfile, wy.mr.make_rdm1())
        else:
            pass
     

def output_real_space_density(coords, d, path="./density.dat"):
    with open(path, "w") as fp:
        for i, coord in enumerate(coords):
            fp.write("%+16.8e\t%+16.8e\t%+16.8e\t%+16.8e\n" % 
                    (coord[0], coord[1], coord[2], d[i]))


def calc_real_space_density_error(rho0, rho1, weights):
    drho = rho1 - rho0
    nelec0 = numpy.dot(rho0, weights)
    nelec1 = numpy.dot(rho1, weights)
    ndiff = numpy.dot(abs(drho), weights)
    return nelec0, nelec1, ndiff

    # END of calc_real_space_density_error

def I(rho0, rho1, weights):
    drho = rho1 - rho0
    a = numpy.dot(drho*drho, weights)
    b = numpy.dot(rho0*rho0, weights) + numpy.dot(rho1*rho1, weights)
    return a / b
    # END of I()

