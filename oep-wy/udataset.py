from __future__ import print_function, division
import errno
import warnings
import numpy
from pyscf import dft
try:
    from tqdm import tqdm
except ImportError:
    warning.warn('tqdm not found. Progress monitor is disabled.', ImportWarning)
    tqdm = None

from density import _density_on_grids2
from potential import _vbg_on_grids


BLK_SIZE = 200


def _gen_data(wy, mesh, coords):
    rho = _density_on_grids2(wy.mol, coords, wy.dm_oep, deriv=1) 
    pbg = _vbg_on_grids(wy, mesh)
    return rho, pbg
    # END of _gen_data()


def _pack_data(rho, pbg, n, na3):
    rho, rhox, rhoy, rhoz = rho

    rho = rho.reshape(n, na3)
    rhox = rhox.reshape(n, na3)
    rhoy = rhoy.reshape(n, na3)
    rhoz = rhoz.reshape(n, na3)
    pbg = pbg.reshape(n, -1)
    return numpy.concatenate((rho, rhox, rhoy, rhoz, pbg), axis=1)
    # END of _pack_data


def ugen_data(wy, mesh, coords, cube_point):
    na = cube_point
    na3 = na * na * na
    data = numpy.empty([len(mesh), 4 * na3 + 1], dtype=numpy.float32)

    total_size = len(mesh)
    n_blk = total_size // BLK_SIZE
    res = total_size - n_blk * BLK_SIZE
    
    pbar = None
    if tqdm is not None:
        pbar = tqdm(range(total_size))

    for ib in range(n_blk):
        index_m = slice(ib * BLK_SIZE, (ib + 1) * BLK_SIZE)
        index_c = slice(ib * BLK_SIZE * na3, (ib + 1) * BLK_SIZE * na3) 
        rho, pbg = _gen_data(wy, mesh[index_m], coords[index_c])
        data_slice = _pack_data(rho, pbg, BLK_SIZE, na3)
        data[index_m] = data_slice.astype(numpy.float32)
        
        if pbar is not None: pbar.update(BLK_SIZE)

    if res > 0:
        rho, pbg = _gen_data(wy, mesh[-res:], coords[-res * na3:])
        data_slice = _pack_data(rho, pbg, res, na3)
        data[-res:] = data_slice.astype(numpy.float32)

        if pbar is not None: pbar.update(res)

    if pbar is not None: pbar.close()
    
    return data
        





