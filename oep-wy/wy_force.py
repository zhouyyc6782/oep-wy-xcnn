from __future__ import print_function
import numpy

from pyscf.grad import rks as rks_grad

from build_matrix import build_vbg1
from solve_KS import solve_KS
from utility import clean_array, print_mat


einsum = numpy.einsum


def _HF_elec(mol, dm):
    """
    Hellmann-Feynman force, electronic part.

    Args:
        mol (Mole): PySCF Mole class.
        dm (ndarray): density matrix in AO.

    Returns:
        HF force at three directions.
    """
    
    offsetdic = mol.offset_nr_by_atom()
    HF_elec = numpy.zeros([mol.natm, 3])

    for k, ia in enumerate(range(mol.natm)):
        shl0, shl1, p0, p1 = offsetdic[ia]
        mol.set_rinv_origin(mol.atom_coord(ia))
        vrinv = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        HF_elec[k] += einsum("xij,ij->x", vrinv, dm) * 2

    return HF_elec


def _HF_nuc(mol):
    """
    Hellmann-Feynman force, nuclear part.

    Args:
        mol (Mole): PySCF Mole class.

    Returns:
        HF force at three directions.
    """

    HF_nuc = numpy.zeros((mol.natm, 3))
    atom_coords = mol.atom_coords()
    Z = mol.atom_charges()
    for i in range(mol.natm):
        for j in range(mol.natm):
            if j == i: continue
            Z_pair = Z[i] * Z[j]

            dis_r = numpy.sum((atom_coords[i] - atom_coords[j]) * (atom_coords[i] - atom_coords[j]))
            dis_r = numpy.sqrt(dis_r)
            dis_r3 = dis_r * dis_r * dis_r
            for k in range(3):
                HF_nuc[i][k] += Z_pair * (atom_coords[j][k] - atom_coords[i][k]) / dis_r3

    return HF_nuc


def HF_force(mol, dm, debug=False):
    HF_elec = _HF_elec(mol, dm)
    HF_nuc = _HF_nuc(mol)

    if debug:
        print_mat(HF_elec, "HF_elec")
        print_mat(HF_nuc, "HF_nuc")
        print_mat(HF_elec+HF_nuc, "HF")

    HF_force = HF_elec + HF_nuc
    return HF_force


# Pulay part
def _make_rdm1e(wy, dm=None, b_potential=None):
    if dm is None: dm = wy.dm_oep
    if b_potential is None: b_potential = wy.b_potential

    # clean_array(b_potential)
    # e, c, dm_rks, vbg = solve_KS(
    #         wy.b_potential, wy.integrals_3c1e_ovlp, 
    #         wy.H, wy.S, wy.mo_occ)

    mo0 = wy.c[:, wy.mo_occ>0]
    mo0e = mo0 * (wy.e[wy.mo_occ>0] * wy.mo_occ[wy.mo_occ>0])
    dme_rks = numpy.dot(mo0e, mo0.T.conj())

    return dm, dme_rks


def get_veff1(wy, g, dm, opts):
    """
    Get gradients of the effective potential.
    nabla veff = nabla v_0 + nabla v_bg.

    Args:
        wy (WY) : wy object.
        g (obj): PySCF gradient object.
        dm (ndarray): density matrix in AO.
        opts (dict): OEP options.

    Returns:
        Gradients of components of the effective potential.
    """
    n = wy.mol.nelectron
    
    if opts['ReferencePotential'][0].lower() == 'hfx':
        vj1, vk1 = g.get_jk(dm=wy.mr.make_rdm1())
        v01 = vj1 - vk1 * 0.5

    elif opts['ReferencePotential'][0].lower() == 'fermi-amaldi':
        vj1, vk1 = g.get_jk(dm=wy.dm_fa)
        v01 = vj1 * (wy.mol.nelectron - 1) / wy.mol.nelectron
    else:
        raise NotImplementedError
    
    vbg1 = build_vbg1(wy.b_potential, wy.integrals_3c1e_ip_ovlp)
    
    return v01, vbg1
    

def Pulay_force(wy, g, opts, dm, dme, debug=False):
    h1 = g.get_hcore()
    s1 = g.get_ovlp()
    
    v01, vbg1 = get_veff1(wy, g, dm, opts)
    veff1 = v01 + vbg1
    f1 = h1 + veff1

    offsetdic = wy.mol.offset_nr_by_atom()

    Pulay_tf = numpy.zeros([wy.mol.natm, 3])
    Pulay_ts = numpy.zeros([wy.mol.natm, 3])
    if debug:
        th = numpy.zeros([wy.mol.natm, 3])
        t0 = numpy.zeros([wy.mol.natm, 3])
        tbg = numpy.zeros([wy.mol.natm, 3])
    
    for ia in range(wy.mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        Pulay_tf[ia] += einsum('xij,ij->x', f1[:, p0:p1], dm[p0:p1]) * 2
        Pulay_ts[ia] -= einsum('xij,ij->x', s1[:, p0:p1], dme[p0:p1]) * 2
        
        if debug:
            th[ia] += einsum('xij,ij->x', h1[:, p0:p1], dm[p0:p1]) * 2 
            t0[ia] += einsum('xij,ij->x', v01[:, p0:p1], dm[p0:p1]) * 2 
            tbg[ia] += einsum('xij,ij->x', vbg1[:, p0:p1], dm[p0:p1]) * 2 
    
    if debug:
        print_mat(th, 'Pulay_th')
        print_mat(t0, 'Pulay_t0')
        print_mat(tbg, 'Pulay_tbg')
        print_mat(Pulay_tf, 'Pulay_tf')
        print_mat(Pulay_ts, 'Pulay_ts')
        print_mat(Pulay_tf+Pulay_ts, 'Pulay')
    
    return Pulay_tf + Pulay_ts


def calc_force(wy, opts, debug=False):
    debug = False
    g = rks_grad.Gradients(wy.mr)

    if opts['ReferencePotential'][0].lower() in ['hfx', 'fermi-amaldi']:
        g = rks_grad.Gradients(wy.mr)

    else:
        raise NotImplementedError

    HF = HF_force(wy.mol, wy.dm_oep, debug)

    dm_rks, dme_rks = _make_rdm1e(wy) 
    Pulay = Pulay_force(wy, g, opts, dm_rks, dme_rks, debug)
    
    return HF + Pulay


def Pulay_term(wy, v1, dm):
    t = numpy.zeros([wy.mol.natm, 3])
    offsetdic = wy.mol.offset_nr_by_atom()
    for ia in range(wy.mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        t[ia] += einsum('xij,ij->x', v1[:, p0:p1], dm[p0:p1]) * 2 
    return t



