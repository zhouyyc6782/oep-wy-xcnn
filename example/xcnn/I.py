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
    dms = [numpy.load(fn) for fn in sys.argv[3:]]
    dms.insert(0, dm_cc)
    dms.insert(1, dm_rks)
    labels = sys.argv[3:]
    labels.insert(0, "cc")
    labels.insert(1, "b3lypg")
    
    mr = scf.RKS(mol)
    mr.xc = "B3LYPG"
    mr.grids.build()
    dms = numpy.array(dms)
    print("[34m======== I ========[0m")
    for i, dm in enumerate(dms[1:]):
        numerator, denominator = compare_density(
                mol, dm, dms[0], mr.grids, "I", 
                full_output=True)
        print("%20s\t%16.8e / %16.8e = %16.8e" % (labels[i+1], numerator, denominator, numerator / denominator))
    

if __name__ == "__main__":
    main()

