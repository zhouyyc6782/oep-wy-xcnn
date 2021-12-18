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


def _mesh_xz(mesh):
    print('Transform grids onto half xz')
    coords = numpy.zeros([len(mesh), 3])
    coords[:, 0] = numpy.sqrt(
            numpy.sum(
                mesh[:, :2] * mesh[:, :2],
                axis=1
                ))
    coords[:, 2] = mesh[:, 2]
    coords_tuple = [tuple(c) for c in coords]
    new_mesh = numpy.asarray(list(set(coords_tuple)))
    print('Refined mesh size: %d' % (len(new_mesh)))
    return new_mesh


def _mesh_xz_half(mesh, sgn=1):
    print('Transform grids onto half xz-plane')
    coords = numpy.zeros([len(mesh), 3])
    coords[:, 0] = numpy.sqrt(
            numpy.sum(
                mesh[:, :2] * mesh[:, :2],
                axis=1
                ))
    coords[:, 2] = numpy.abs(mesh[:, 2]) * sgn
    coords_tuple = [tuple(c) for c in coords]
    new_mesh = numpy.asarray(list(set(coords_tuple)))
    print('Refined mesh size: %d' % (len(new_mesh)))
    return new_mesh


def _rot(rad, d):
    if d == 0:
        return numpy.array([
            [1., 0., 0.],
            [0., numpy.cos(rad), -numpy.sin(rad)],
            [0., numpy.sin(rad), numpy.cos(rad)],
            ])
    if d == 1:
        return numpy.array([
            [numpy.cos(rad), 0., numpy.sin(rad)],
            [0., 1., 0.], 
            [-numpy.sin(rad), 0., numpy.cos(rad)],
            ])
    if d == 2:
        return numpy.array([
            [numpy.cos(rad), -numpy.sin(rad), 0],
            [numpy.sin(rad), numpy.cos(rad), 0],
            [0., 0., 1.],
            ])


def gen_mesh(mol, opts):
    if True: # grid type == ??
        grids = dft.gen_grid.Grids(mol)
        grids.prune = None
        grids.level = opts['MeshLevel']
        grids.build()
        mesh = grids.coords
        print('Origin mesh size: %d' % (len(mesh)))
        sym = opts['Symmetric'].lower() if opts['Symmetric'] is not None else None
        if opts['RandomTransform']: sym = 'none'
        if sym is None or sym == 'none':
            pass
        elif sym == 'xz':
            mesh = _mesh_xz(mesh)
        elif sym == 'xz+':
            mesh = _mesh_xz_half(mesh, 1)
        elif sym == 'xz-':
            mesh = _mesh_xz_half(mesh, -1)
        else:
            warnings.warn('Unknown Symmetric option. Fallback to ``None"', RuntimeWarning)

        rot_mat = None
        if opts['RandomTransform']:
            rad = numpy.random.rand(3) * 2 * numpy.pi
            print('Rotation angle')
            print('\tx: %.12lf' % (rad[0]))
            print('\ty: %.12lf' % (rad[1]))
            print('\tz: %.12lf' % (rad[2]))
            rot_mat = numpy.zeros([4, 3, 3])
            for i in range(3):
                rot_mat[i] = _rot(rad[i], i)
            rot_mat[3] = numpy.einsum('ij,jk,kl->il', rot_mat[0], rot_mat[1], rot_mat[2])
            origin_mesh = mesh.copy()
            mesh = numpy.einsum('ij,jk->ik', rot_mat[3], origin_mesh.T).T
            assert(mesh.shape == origin_mesh.shape)

    if rot_mat is not None:
        return mesh, origin_mesh, rot_mat
    else:
        return mesh


def _gen_offset(cube_length, cube_point):
    a = cube_length
    na = cube_point
    da = a / float(na - 1)
    a0 = -a / 2.

    offset = numpy.zeros([na*na*na, 3], dtype=float)
    p = 0
    for i in range(na):
        for j in range(na):
            for k in range(na):
                offset[p][0] = a0 + da * i
                offset[p][1] = a0 + da * j
                offset[p][2] = a0 + da * k
                p += 1

    return offset
    # END of _gen_offset()


def gen_grids(mesh, cube_length, cube_point):
    offset = _gen_offset(cube_length, cube_point)
    coords = numpy.zeros([len(mesh)*len(offset), 3], dtype=float)
    na3 = len(offset)
    assert(na3 == cube_point**3)

    for i, m in enumerate(mesh):
        coords[i * na3 : (i + 1) * na3] = m + offset 

    return coords 
    # END of gen_grids()


def _gen_data(wy, mesh, coords, dm_oep, b):
    rho = _density_on_grids2(wy.mol, coords, dm_oep, deriv=1) 
    pbg = _vbg_on_grids(wy, mesh, b)
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


def gen_data(wy, mesh, coords, cube_point):
    if wy.b_potential.ndim == 2:
        return ugen_data(wy, mesh, coords, cube_point)

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
        rho, pbg = _gen_data(wy, mesh[index_m], coords[index_c], wy.dm_oep, wy.b_potential)
        data_slice = _pack_data(rho, pbg, BLK_SIZE, na3)
        data[index_m] = data_slice.astype(numpy.float32)
        
        if pbar is not None: pbar.update(BLK_SIZE)

    if res > 0:
        rho, pbg = _gen_data(wy, mesh[-res:], coords[-res * na3:], wy.dm_oep, wy.b_potential)
        data_slice = _pack_data(rho, pbg, res, na3)
        data[-res:] = data_slice.astype(numpy.float32)

        if pbar is not None: pbar.update(res)

    if pbar is not None: pbar.close()
    
    return data
        

def ugen_data(wy, mesh, coords, cube_point):
    na = cube_point
    na3 = na * na * na
    data = numpy.empty([2, len(mesh), 4 * na3 + 1], dtype=numpy.float32)

    total_size = len(mesh)
    n_blk = total_size // BLK_SIZE
    res = total_size - n_blk * BLK_SIZE
    
    for ispin in [0, 1]:
        pbar = None
        if tqdm is not None:
            pbar = tqdm(range(total_size))
        for ib in range(n_blk):
            index_m = slice(ib * BLK_SIZE, (ib + 1) * BLK_SIZE)
            index_c = slice(ib * BLK_SIZE * na3, (ib + 1) * BLK_SIZE * na3) 
            rho, pbg = _gen_data(wy, mesh[index_m], coords[index_c], wy.dm_oep[ispin], wy.b_potential[ispin])
            data_slice = _pack_data(rho, pbg, BLK_SIZE, na3)
            data[ispin][index_m] = data_slice.astype(numpy.float32)
            
            if pbar is not None: pbar.update(BLK_SIZE)

        if res > 0:
            rho, pbg = _gen_data(wy, mesh[-res:], coords[-res * na3:], wy.dm_oep[ispin], wy.b_potential[ispin])
            data_slice = _pack_data(rho, pbg, res, na3)
            data[ispin][-res:] = data_slice.astype(numpy.float32)

            if pbar is not None: pbar.update(res)

        if pbar is not None: pbar.close()
    
    return data


