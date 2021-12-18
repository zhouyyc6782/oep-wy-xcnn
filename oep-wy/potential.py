from __future__ import division
import numpy 


BLK_SIZE = 200
einsum = numpy.einsum


def _vbg_on_grids(wy, coords, b=None):  
    if b is None: b = wy.b_potential
    aux = wy.mol.copy()
    aux.basis = wy.concatenate_basis
    aux.build()

    concatenate_ao_values = aux.eval_gto("GTOval_sph", coords)
    
    potential_ao_values = numpy.empty([len(coords), wy.nbas_potential])
    ptr_orb = 0
    ptr_pot = 0
    for i in range(wy.mol.natm):
        ptr_orb_inc = wy.nbas_orbital_atom[i]
        ptr_pot_inc = wy.nbas_potential_atom[i]
        potential_ao_values[:, ptr_pot:ptr_pot+ptr_pot_inc] = \
                concatenate_ao_values[:, ptr_orb+ptr_orb_inc:ptr_orb+ptr_orb_inc+ptr_pot_inc]
        ptr_orb += ptr_orb_inc + ptr_pot_inc
        ptr_pot += ptr_pot_inc

    # sum_{t} b_{t} g_{t}(r)
    # potential = einsum('ij,j->i', potential_ao_values, wy.b_potential)
    potential = einsum('ij,j->i', potential_ao_values, b)

    return potential
    # END of _vbg_on_grids()


def _vj_on_grids(wy, coords, dm):
    print('Compute potential of electron density. Wait patiently... ', end='', flush=True)
    aux = wy.mol.copy()

    vj = numpy.zeros(len(coords))
    for i, p in enumerate(coords):
        aux.set_rinv_origin(p)
        vj[i] = numpy.einsum('ij,ji', aux.intor('int1e_rinv'), dm)
    print('Done.', flush=True)
    return vj
    # END of _vj_on_grids()


def calc_real_space_vbg(wy, coords):
    total_size = len(coords)
    n_blk = total_size // BLK_SIZE
    res = total_size - n_blk * BLK_SIZE

    if wy.b_potential.ndim == 1:
        p = numpy.zeros(total_size, dtype=float)
        for i in range(n_blk):
            index = slice(i*BLK_SIZE, (i+1)*BLK_SIZE)
            p[index] = _vbg_on_grids(wy, coords[index])

        if res > 0:
            p[-res:] = _vbg_on_grids(wy, coords[-res:])
    else:
        p = numpy.zeros([2, total_size], dtype=float)
        for k in range(2):
            for i in range(n_blk):
                index = slice(i*BLK_SIZE, (i+1)*BLK_SIZE)
                p[k][index] = _vbg_on_grids(wy, coords[index], b=wy.b_potential[k])

            if res > 0:
                p[k][-res:] = _vbg_on_grids(wy, coords[-res:], b=wy.b_potential[k])

    return p
    # END of calc_real_space_vbg()


def calc_real_space_vj(wy, coords, dm):
    return _vj_on_grids(wy, coords, dm)
    # END of calc_real_space_vj()


def calc_vbg_cube(wy, outfile='./vbg.cube'):
    aux = wy.mol.copy()
    aux.basis = wy.concatenate_basis
    aux.build()

    coef = numpy.zeros(wy.nbas_orbital + wy.nbas_potential)

    ptr_orb = 0
    ptr_pot = 0
    pb = 0
    for i in range(wy.mol.natm):
        ptr_orb_inc = wy.nbas_orbital_atom[i]
        ptr_pot_inc = wy.nbas_potential_atom[i]
        coef[ptr_orb+ptr_orb_inc:ptr_orb+ptr_orb_inc+ptr_pot_inc] = wy.b_potential[ptr_pot:ptr_pot+ptr_pot_inc]
        ptr_orb += ptr_orb_inc + ptr_pot_inc
        ptr_pot += ptr_pot_inc

    from pyscf.tools import cubegen
    cubegen.orbital(aux, outfile, coef)
    # END of calc_vbg_cube()


def output_real_space_potential(coords, p, path="./potential.dat"):
    with open(path, "w") as fp:
        for i, coord in enumerate(coords):
            fp.write("%+16.8e\t%+16.8e\t%+16.8e\t%+16.8e\n" % 
                    (coord[0], coord[1], coord[2], p[i]))
    # END of output_real_space_potential()


