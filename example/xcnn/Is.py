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
    mr.grids.build()
    # mr.run()

    dms = [numpy.load(fn) for fn in sys.argv[3:]]

    dms = numpy.array(dms)
    print("[34m======== I ========[0m")
    for i in range(len(dms)):
        for j in range(len(dms)):
            numerator, denominator = compare_density(
                    mol, dms[i], dms[j], mr.grids, "I", 
                    full_output=True)
            # print("%6s - %6s: %16.8e / %16.8e = %16.8e  " % (sys.argv[i+3].split("/")[-2], sys.argv[j+3].split("/")[-2], numerator, denominator, numerator / denominator))
            print("%6s - %6s: %16.8e / %16.8e = %16.8e  " % (sys.argv[i+3], sys.argv[j+3], numerator, denominator, numerator / denominator))
    

if __name__ == "__main__":
    main()

