from __future__ import print_function
import errno
import os
import sys
import numpy


import configparser
from Config import get_options
from wy import WY
from uwy import UWY
from dataset import *
from density import *
from potential import *



def oep():
    oep_opts = get_options(sys.argv[1], 'OEP')
    
    for k, v in oep_opts.items():
        print(k, '\t', v, '\t', v.__class__)

    print('Check point files saved to %s' % (oep_opts['CheckPointPath']))
    try: 
        os.makedirs(oep_opts['CheckPointPath'])
    # except FileExistsError:
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('CHK folder already exists. Old files will be overwritten.')

    print('==== OEP ====')
    if oep_opts['SpinUnrestricted']: 
        wy = UWY(oep_opts)
        wy.OEP()
    else:
        wy = WY(oep_opts)
        wy.OEP()
    print()
    print('==== Force ====')
    force = wy.force(False)
    print(force)

    print('==== Analysis ====')
    if oep_opts['RealSpaceAnalysis']:
        coords = wy.mr.grids.coords
        weights = wy.mr.grids.weights
        if oep_opts['SpinUnrestricted']:
            rho_in = numpy.array([
                calc_real_space_density(wy.mol, coords, wy.dm_in[0]), 
                calc_real_space_density(wy.mol, coords, wy.dm_in[1]), 
                ])
            rho_oep = numpy.array([
                calc_real_space_density(wy.mol, coords, wy.dm_oep[0]), 
                calc_real_space_density(wy.mol, coords, wy.dm_oep[1]), 
                ])
            rho_diff = rho_oep - rho_in

            output_real_space_density(coords, rho_diff[0], '%s/rho_diff_a.dat' % (wy.chk_path))
            output_real_space_density(coords, rho_diff[1], '%s/rho_diff_b.dat' % (wy.chk_path))
            output_real_space_density(coords, rho_in[0], '%s/rho_in_a.dat' % (wy.chk_path))
            output_real_space_density(coords, rho_in[1], '%s/rho_in_b.dat' % (wy.chk_path))
            output_real_space_density(coords, rho_oep[0], '%s/rho_oep_a.dat' % (wy.chk_path))
            output_real_space_density(coords, rho_oep[1], '%s/rho_oep_b.dat' % (wy.chk_path))

            print('Density difference in real space: rho_in, rho_oep, abs_diff')
            print('%16.12e\t%16.12e\t%16.12e' % (calc_real_space_density_error(rho_in[0], rho_oep[0], weights)))
            print('%16.12e\t%16.12e\t%16.12e' % (calc_real_space_density_error(rho_in[1], rho_oep[1], weights)))

            pbg = calc_real_space_vbg(wy, coords)
            output_real_space_potential(coords, pbg[0], '%s/pbg_a.dat' % (wy.chk_path))
            output_real_space_potential(coords, pbg[1], '%s/pbg_b.dat' % (wy.chk_path))

        else:
            rho_in = calc_real_space_density(wy.mol, coords, wy.dm_in)
            output_real_space_density(coords, rho_in, '%s/rho_in.dat' % (wy.chk_path))
            rho_oep = calc_real_space_density(wy.mol, coords, wy.dm_oep)
            output_real_space_density(coords, rho_oep, '%s/rho_oep.dat' % (wy.chk_path))
            rho_diff = rho_oep - rho_in
            output_real_space_density(coords, rho_diff, '%s/rho_diff.dat' % (wy.chk_path))

            print('Density difference in real space: rho_in, rho_oep, abs_diff')
            print('%16.12e\t%16.12e\t%16.12e' % (calc_real_space_density_error(rho_in, rho_oep, weights)))

            pbg = calc_real_space_vbg(wy, coords)
            output_real_space_potential(coords, pbg, '%s/pbg.dat' % (wy.chk_path))

        # if (oep_opts['FullRealSpacePotential'] and oep_opts['ReferencePotential'][0].lower() == 'fermi-amaldi'):
        #     dm0 = numpy.load(oep_opts['ReferencePotential'][1])
        #     dm = (wy.mol.nelectron - 1) / wy.mol.nelectron * dm0 - wy.dm_oep
        #     pj = calc_real_space_vj(wy, coords, dm)
        #     output_real_space_potential(coords, pj, '%s/pxc.dat' % (wy.chk_path))

    print('Mulliken analysis')
    ao_slice = wy.mol.aoslice_by_atom()
    print('    OEP')
    q = numpy.einsum('ij,ji->i', wy.dm_oep, wy.mol.intor('int1e_ovlp'))
    for ia in range(wy.mol.natm):
        s0, s1, p0, p1 = ao_slice[ia]
        print('    ', wy.mol.atom_symbol(ia), q[p0:p1].sum())
    print('    CCSD')
    q = numpy.einsum('ij,ji->i', wy.dm_ccsd, wy.mol.intor('int1e_ovlp'))
    for ia in range(wy.mol.natm):
        s0, s1, p0, p1 = ao_slice[ia]
        print('    ', wy.mol.atom_symbol(ia), q[p0:p1].sum())

    print('==== END OF OEP ====')
    return wy
    # END of oep()


def dataset(wy):
    dataset_opts = get_options(sys.argv[1], 'DATASET')    
    for k, v in dataset_opts.items():
        print(k, '\t', v, '\t', v.__class__)

    try: 
        os.makedirs(dataset_opts['OutputPath'])
    # except FileExistsError:
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass

    spin_unrestricted = wy.b_potential.ndim == 2

    fn = '%s/%s' % (dataset_opts['OutputPath'], dataset_opts['OutputName'])
    if not spin_unrestricted:
        if os.path.isfile('%s.npy' % (fn)):
            print('Data file %s.npy exists and will be overwritten.' % (fn))
    else:
        if os.path.isfile('%s_a.npy' % (fn)):
            print('Data file %s_a.npy exists and will be overwritten.' % (fn))
        if os.path.isfile('%s_b.npy' % (fn)):
            print('Data file %s_b.npy exists and will be overwritten.' % (fn))

    print('==== DATASET ====')

    if dataset_opts['RandomTransform']:
        mesh, origin_mesh, rot_mat = gen_mesh(wy.mol, dataset_opts)
    else:
        mesh = gen_mesh(wy.mol, dataset_opts)
        rot_mat = None
        origin_mesh = None

    coords = gen_grids(mesh, dataset_opts['CubeLength'], dataset_opts['CubePoint'])
    origin_coords = None
    if rot_mat is not None:
        inv_rot_mat = numpy.linalg.inv(rot_mat[3])
        origin_coords = numpy.einsum('ij,jk->ik', inv_rot_mat, coords.T).T

    if origin_coords is not None:
        data = gen_data(wy, origin_mesh, origin_coords, dataset_opts['CubePoint'])
    else:
        data = gen_data(wy, mesh, coords, dataset_opts['CubePoint'])

    if not spin_unrestricted:
        numpy.save(fn, data)
    else:
        numpy.save('%s_a' % (fn), data[0])
        numpy.save('%s_b' % (fn), data[1])
    numpy.save('%s_coords' % (fn), mesh)
    if origin_mesh is not None:
        numpy.save('%s_coords_no_rot' % (fn), origin_mesh)

    print('Data size:', data.shape)
    print('==== END OF DATASET ====')

    # END of dataset()


def main():
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    sections = config.sections()
    if 'OEP' in sections: 
        wy = oep()

    if 'DATASET' in sections: 
        dataset(wy)


if __name__ == '__main__':
    main()
    print()



