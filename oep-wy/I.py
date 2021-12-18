from __future__ import division
import argparse
import numpy

from pyscf import gto, scf
from density import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('atom', type=str)
    parser.add_argument('basis', type=str)
    parser.add_argument('--charge', dest='charge', type=int, nargs='?', default=0)
    parser.add_argument('dm_fns', type=str, nargs='+')
    parser.add_argument('--grid_level', dest='grid_level', type=int, nargs='?', default=3)
    args = parser.parse_args()

    mol = gto.M(atom=args.atom, basis=args.basis, charge=args.charge)
    mr = scf.RKS(mol)
    mr.grids.level = args.grid_level
    mr.grids.build()

    coords = mr.grids.coords
    weights = mr.grids.weights
    
    dms = [numpy.load(fn) for fn in args.dm_fns]
    rhos = [calc_real_space_density(mol, coords, dm) for dm in dms] 
    Is = [I(rhos[0], rho, weights) for rho in rhos]
    for i in Is:
        print('%+16.8e' % (i))


if __name__ == '__main__':
    main()

