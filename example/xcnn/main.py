from __future__ import print_function
import os, sys, errno
import numpy

import pyscf
ver = pyscf.__version__.split('.')
import configparser
from Config import get_options
from nnks import NNKS
from utility import dft, ccsd, I


def main():
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    sections = config.sections()

    xcnn_opts = get_options(sys.argv[1], 'XCNN')
    for k, v in xcnn_opts.items():
        print(k, '\t', v, '\t', v.__class__)

    print('Check point files saved to %s' % (xcnn_opts['CheckPointPath']))
    try: 
        os.makedirs(xcnn_opts['CheckPointPath'])
    #except FileExistsError:
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('CHK folder already exists. Old files will be overwritten.')

    nnks = NNKS(xcnn_opts)
    print('Prepare to perform post DFT SCF calculation.')
    nnks.scf()
    numpy.save('%s/dm_nn' % (xcnn_opts['CheckPointPath']), nnks.dm)
    dm_rks = dft(nnks.mol, xc='B3LYPG')
    dm_cc = ccsd(nnks.mol)
    numpy.save('%s/dm_rks_b3lypg' % (xcnn_opts['CheckPointPath']), dm_rks)
    numpy.save('%s/dm_ccsd' % (xcnn_opts['CheckPointPath']), dm_cc)

    print()
    print('---- Force ----')
    force = nnks.force(True)
    print('Force on atoms')
    for f in force:
        print('%+16.12e %+16.12e %+16.12e' % (f[0], f[1], f[2]))
    print()

    # mtorque = numpy.cross(force, nnks.mol.atom_coords())
    # print('mtorque')
    # for t in mtorque:
    #     print('%+16.12e %+16.12e %+16.12e' % (t[0], t[1], t[2]))
    # print()

    print('---- I ----')
    print('rks, nn vs ccsd: %16.12e\t%16.12e' % 
            (
                I(nnks.mol, dm_cc, dm_rks, nnks.grids._coords, nnks.grids.weights), 
                I(nnks.mol, dm_cc, nnks.dm, nnks.grids._coords, nnks.grids.weights), 
            ))


if __name__ == '__main__':
    main()

