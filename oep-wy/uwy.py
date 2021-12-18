from __future__ import print_function, division
import numpy
import os, os.path
from pyscf import gto, scf, dft, cc

from basis import get_integrals_3c1e, get_integrals_3c1e_ip_ovlp, get_overlap_g, get_kinetic_g
from IO import parse_str
from uwy_force import calc_force
from usearch import search

einsum = numpy.einsum

class UWY:
    """
    Basis class to handle the WY method.
    """

    def __init__(self, opts):
        self.opts = opts
        self.chk_path = opts['CheckPointPath']

        self._generate_mol(opts['Structure'], opts['OrbitalBasis'])
        self._get_basis_info(opts['OrbitalBasis'], opts['PotentialBasis'])        
        self._get_fixed_matrix(opts['ReferencePotential'], opts['CheckPointPath'])

        self._get_input_dm(opts['InputDensity'])
        self._get_init_coefficient(opts['PotentialCoefficientInit'])


        # computation options
        self.oep_max_iter = opts['MaxIterator']
        self.oep_crit = opts['ConvergenceCriterion']
        self.svd_cutoff = opts['SVDCutoff']
        self.Lambda = opts['LambdaRegulation']
        self.force_v0 = 0.
        self.constrains = list()
        if self.Lambda > 0: self.constrains.append('LambdaRegulation')
        if opts['ZeroForceConstrain']: self.constrains.append('ZeroForce')
    

    def _generate_mol(self, fn, basis):
        atom_list, atom_str, charge, spin = parse_str(fn)    
        
        self.mol = gto.M(
                atom=atom_str, 
                basis=basis, 
                charge=charge, 
                spin=spin,
                max_memory=32768,
                )
        
        n_elec = self.mol.nelectron
        n_ao = self.mol.nao_nr()
        self.mo_occ = numpy.zeros([2, n_ao])
        self.mo_occ[0][:n_elec//2] = 1
        self.mo_occ[1][:n_elec//2] = 1
        self.n_occ = n_elec // 2

        self.mf = scf.UHF(self.mol)
        self.mr = scf.UKS(self.mol)

        print('Structure info:')
        for a in self.mol._atom:
            print('%2s\t%+12.8e\t%+12.8e\t%+12.8e  AA' % 
                    (a[0], a[1][0], a[1][1], a[1][2]))
        print('charge = %d\nspin = %d' % (self.mol.charge, self.mol.spin))
        # END of _generate_mol()


    def _get_basis_info(self, orb_basis, pot_basis):
        mol = self.mol
        atoms = [mol.atom_symbol(i) for i in range(mol.natm)]

        res = get_integrals_3c1e(
                mol, atoms, 
                orb_basis, pot_basis)
        self.orbital_basis = res[0]
        self.potential_basis = res[1]
        self.concatenate_basis = res[2]
        self.integrals_3c1e_ovlp = res[3]

        res = get_integrals_3c1e_ip_ovlp(
                mol, atoms, 
                orb_basis, pot_basis)
        self.integrals_3c1e_ip_ovlp = res[3]

        self.nbas_orbital = 0 
        self.nbas_potential = 0
        self.nbas_orbital_atom = numpy.zeros(len(atoms), dtype=int)
        self.nbas_potential_atom = numpy.zeros(len(atoms), dtype=int)

        for i, atom in enumerate(atoms):
            for basis in self.orbital_basis[atom]: 
                self.nbas_orbital += basis[0] * 2 + 1 
                self.nbas_orbital_atom[i] += basis[0] * 2 + 1
                if len(basis[1]) > 2:
                    self.nbas_orbital += basis[0] * 2 + 1
                    self.nbas_orbital_atom[i] += basis[0] * 2 + 1
            for basis in self.potential_basis[atom]: 
                self.nbas_potential += basis[0] * 2 + 1
                self.nbas_potential_atom[i] += basis[0] * 2 + 1
                if len(basis[1]) > 2:
                    self.nbas_potential += basis[0] * 2 + 1
                    self.nbas_potential_atom[i] += basis[0] * 2 + 1

        self.Sg = get_overlap_g(self.mol, 
                self.concatenate_basis, 
                self.nbas_orbital_atom, 
                self.nbas_potential_atom)
        self.Tg = get_kinetic_g(self.mol,
                self.concatenate_basis,
                self.nbas_orbital_atom,
                self.nbas_potential_atom)

        self.aux_mol = self.mol.copy()
        self.aux_mol.basis = self.concatenate_basis
        self.aux_mol.build()

        # mr = scf.RKS(self.mol)
        self.mr.grids.build()
        self.orb_ao_value = dft.numint.eval_ao(self.mol, self.mr.grids.coords, deriv=1)
        self.concatenate_ao_value = dft.numint.eval_ao(self.aux_mol, self.mr.grids.coords, deriv=1)
        self.pot_ao_value = numpy.empty([4, len(self.mr.grids.coords), self.nbas_potential])
        ptr_orb = 0
        ptr_pot = 0
        for i in range(self.mol.natm):
            ptr_orb_inc = self.nbas_orbital_atom[i]
            ptr_pot_inc = self.nbas_potential_atom[i]
            self.pot_ao_value[:, :, ptr_pot:ptr_pot+ptr_pot_inc] = \
                    self.concatenate_ao_value[:, :, ptr_orb+ptr_orb_inc:ptr_orb+ptr_orb_inc+ptr_pot_inc]
            ptr_orb += ptr_orb_inc + ptr_pot_inc
            ptr_pot += ptr_pot_inc
        # END of _get_basis_info()


    def _get_fixed_matrix(self, ref_pot, chk_path):
        mol = self.mol
        self.S = mol.intor('int1e_ovlp_sph')
        self.T = mol.intor('cint1e_kin_sph')
        self.vn = mol.intor('cint1e_nuc_sph')
        
        if ref_pot[0].lower() == 'hfx':
            self.reference_potential = 'hfx'
            print("[31mUse HFX reference potential[0m")
            self.mr.xc = 'hf,'
            # if os.path.isfile('%s/dm_hfx.npy' % (chk_path)):
            #     print("[31mLoad dm from %s/dm_hfx.npy[0m" % (chk_path))
            #     dm_hfx = numpy.load('%s/dm_hfx.npy' % (chk_path))
            #     self.mr.kernel(dm0=dm_hfx)
            # else:
            self.dm_hfx = numpy.load(ref_pot[1])
            self.mr.kernel() # dm0=numpy.load(ref_pot[1]))
            # dm_hfx = self.mr.make_rdm1()
            # print("[31mSave dm to %s/dm_hfx.npy[0m" % (chk_path))
            # numpy.save('%s/dm_hfx' % (chk_path), dm_hfx)
            # numpy.save('%s/dm_rks_%s' % (chk_path, self.mr.xc), dm_hfx)
            # self.v0 = self.mr.get_veff(self.mol, self.mr.make_rdm1())
            self.v0 = self.mr.get_veff(self.mol, self.dm_hfx)

        elif ref_pot[0].lower() == 'fermi-amaldi':
            self.reference_potential = 'fermi-amaldi'
            print("[31mUse the Fermi-Amaldi reference potential[0m")
            self.dm_fa = numpy.load(ref_pot[1])
            if self.dm_fa.ndim == 3: self.dm_fa = self.dm_fa[0] + self.dm_fa[1]
            from pyscf.scf import jk
            self.v0 = jk.get_jk(mol, self.dm_fa)[0] * (mol.nelectron - 1) / mol.nelectron
            self.v0 = numpy.array([self.v0, self.v0])

        elif ref_pot[0].lower() == 'hfx-fa':
            self.reference_potential = 'hfx-fa'
            print("[31mUse HFX reference potential for ALPHA density[0m")
            print("[31mUse the Fermi-Amaldi reference potential for BETA density[0m")
            self.dm_hfx = numpy.load(ref_pot[1])
            self.dm_fa = numpy.load(ref_pot[1])
            if self.dm_fa.ndim == 3: self.dm_fa = self.dm_fa[0] + self.dm_fa[1]
            self.mr.kernel(dm0=numpy.load(ref_pot[1]))
            self.v0a = self.mr.get_veff(self.mol, self.dm_hfx)[0]
            from pyscf.scf import jk
            self.v0b = jk.get_jk(mol, self.dm_fa)[0] * (mol.nelectron - 1) / mol.nelectron
            self.v0 = numpy.array([self.v0a, self.v0b])
        
        elif ref_pot[0].lower() == 'sfa':
            self.reference_potential = 'sfa'
            print("[31mUse the Scaled Fermi-Amaldi reference potential[0m")
            self.dm_fa = numpy.load(ref_pot[1])
            if len(ref_pot) >= 3:
                self.sfa_factor = float(ref_pot[2])
            else:
                self.sfa_factor = 1.0
            if self.dm_fa.ndim == 3: self.dm_fa = self.dm_fa[0] + self.dm_fa[1]
            from pyscf.scf import jk
            self.v0 = jk.get_jk(mol, self.dm_fa)[0] * self.sfa_factor
            self.v0 = numpy.array([self.v0, self.v0])

        elif ref_pot[0].lower() == 'shfx':
            self.reference_potential = 'shfx'
            if len(ref_pot) >= 3:
                self.shfx_factor = float(ref_pot[2])
            else:
                self.shfx_factor = 1.0
            print("[31mUse Scaled HFX reference potential[0m")
            self.mr.xc = 'hf,'
            self.dm_hfx = numpy.load(ref_pot[1])
            self.mr.kernel(dm0=numpy.load(ref_pot[1]))
            from pyscf.scf import jk
            vj, vk = self.mr.get_jk(mol, self.dm_hfx)
            # self.v0 = self.mr.get_veff(self.mol, self.dm_hfx) * self.shfx_factor
            self.v0 = vj[0] + vj[1] - vk * self.shfx_factor

        else:
            raise NotImplementedError
        
        self.H = self.T + self.vn + self.v0
        # END of _get_fixed_matrix()
         

    def _get_input_dm(self, input_dm):
        if input_dm[0].lower() == 'none':
            self.mf.kernel()
            self.dm_rhf = self.mf.make_rdm1()
            numpy.save('%s/dm_rhf' % (self.chk_path), self.dm_rhf)
            c = self.mf.mo_coeff

            self.mcc = cc.CCSD(self.mf)
            ecc, t1, t2 = self.mcc.kernel()
            rdm1 = self.mcc.make_rdm1()
            self.dm_ccsd = einsum('pi,ij,qj->pq', c, rdm1, c.conj())
            self.dm_in = self.dm_ccsd
            numpy.save('%s/dm_ccsd' % (self.chk_path), self.dm_ccsd)

        elif input_dm[0].lower() == 'load':
            self.dm_ccsd = numpy.load(input_dm[1])
            self.dm_in = self.dm_ccsd
        else:
            self.dm_in = None
            self.dm_ccsd = None
        # END of _get_input_dm()


    def _get_init_coefficient(self, b_init):
        if b_init[0].lower() == 'zeros':
            self.b_init = numpy.array([
                    numpy.zeros(self.nbas_potential),
                    numpy.zeros(self.nbas_potential)])
            print('Use zero initial values for b.')
        elif b_init[0].lower() == 'load':
            print('Load initial values for b from %s' % (b_init[1]))
            if b_init[1].endswith('.npy'):
                self.b_init = numpy.load(b_init[1])
            else:
                self.b_init = numpy.loadtxt(b_init[1])
        else:
            raise NotImplementedError

        # END of _get_init_coefficient()
    

    def OEP(self):
        self.dm_oep = []

        search(self, True)
        self.dm_oep.append(self.dm)
        
        if 'LambdaRegulation' in self.constrains:
            ws_reg = 2 * einsum(
                    'i,ij,j->', 
                    self.b_potential[0], self.Tg, self.b_potential[0],
                    )
            print('Lambda regulation: %.12e\t%.12e' % (self.Lambda, ws_reg))

        search(self, False)
        self.dm_oep.append(self.dm)
        
        if 'LambdaRegulation' in self.constrains:
            ws_reg = 2 * einsum(
                    'i,ij,j->', 
                    self.b_potential[1], self.Tg, self.b_potential[1],
                    )
            print('Lambda regulation: %.12e\t%.12e' % (self.Lambda, ws_reg))

        self.dm_oep = numpy.array(self.dm_oep)
        numpy.save('%s/b' % (self.chk_path), self.b_potential)
        numpy.save('%s/dm_oep' % (self.chk_path), self.dm_oep)
        numpy.save('%s/mo_energy_oep' % (self.chk_path), self.e)
        numpy.save('%s/mo_coeff_oep' % (self.chk_path), self.c)

    
    def force(self, verbose=False):
        return calc_force(self, self.opts, verbose)



