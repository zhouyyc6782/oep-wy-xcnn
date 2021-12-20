from __future__ import print_function
import numpy
import sys
from pyscf import gto, scf

from xcnn.io.config import get_options
from xcnn.io.read_structure import read_structure
from xcnn.post_dft.nnks import NNKS
from xcnn.post_dft.model import *
from xcnn.utility.ccsd import ccsd
from xcnn.tools.compare_rho import compare_rho

from compare_density import compare_density


def main():
    mol = gto.M(atom=sys.argv[1], basis="augccpvqz", charge=int(sys.argv[2]), spin=0)
    mr = scf.RKS(mol)
    mr.xc = "B3LYPG"
    mr.run()
    dm_rks = mr.make_rdm1()

    dm_cc, _ = ccsd(mol)
    dms = [numpy.load(fn) for fn in sys.argv[3:-1]]
    dms.insert(0, dm_cc)
    dms.insert(1, dm_rks)
    labels = sys.argv[3:-1]
    labels.insert(0, "cc")
    labels.insert(1, "b3lypg")

    if sys.argv[-1].find("I") != -1:
        print("[31m======== I ========[0m")
        for i, dm in enumerate(dms[1:]):
            numerator, denominator = compare_density(
                    mol, dm, dms[0], mr.grids, "I", 
                    full_output=True)
            print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))

    if sys.argv[-1].find("wI1") != -1:
        print("[32m======== wI1 ========[0m")
        for i, dm in enumerate(dms[1:]):
            numerator, denominator = compare_density(
                    mol, dm, dms[0], mr.grids, "wI1", 
                    full_output=True)
            print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))

    if sys.argv[-1].find("wI2") != -1:
        print("[33m======== wI2 ========[0m")
        for i, dm in enumerate(dms[1:]):
            numerator, denominator = compare_density(
                    mol, dm, dms[0], mr.grids, "wI2", 
                    full_output=True)
            print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))

    if sys.argv[-1].find("rms") != -1:
        print("[34m======== rms**2 ========[0m")
        for i, dm in enumerate(dms[1:]):
            numerator, denominator = compare_density(
                    mol, dm, dms[0], mr.grids, "rms2", 
                    full_output=True)
            print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))

    if sys.argv[-1].find("mad") != -1:
        print("[35m======== mad ========[0m")
        for i, dm in enumerate(dms[1:]):
            numerator, denominator = compare_density(
                    mol, dm, dms[0], mr.grids, "mad", 
                    full_output=True)
            print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))

    if sys.argv[-1].find("wmad") != -1:
        print("[36m======== wmad ========[0m")
        for i, dm in enumerate(dms[1:]):
            numerator, denominator = compare_density(
                    mol, dm, dms[0], mr.grids, "wmad", 
                    full_output=True)
            print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))

    if sys.argv[-1].find("wmad2") != -1:
        print("[37m======== wmad2 ========[0m")
        for i, dm in enumerate(dms[1:]):
            numerator, denominator = compare_density(
                    mol, dm, dms[0], mr.grids, "wmad2", 
                    full_output=True)
            print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))

    if sys.argv[-1].find("draw") != -1:
        print("[35m======== graphic ========[0m")
        compare_rho(mol, numpy.array(dms), mr.grids.coords, sys.argv[-1], label=labels)

if __name__ == "__main__":
    main()

