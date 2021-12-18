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
        # mol.set_rinv_origin(mol.atom_coord(ia))
        with mol.with_rinv_as_nucleus(ia):
            vrinv = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
            vrinv += vrinv.transpose(0, 2, 1)
            HF_elec[k] += einsum('xij,ij->x', vrinv, dm) # * 2

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
def _make_rdm1e(uwy, dm=None, b_potential=None):
    if dm is None: dm = uwy.dm_oep
    if b_potential is None: b_potential = uwy.b_potential

    mo0 = uwy.c[0][:, uwy.mo_occ[0]>0]
    mo0e = mo0 * (uwy.e[0][uwy.mo_occ[0]>0] * uwy.mo_occ[0][uwy.mo_occ[0]>0])
    dme_rks_a = numpy.dot(mo0e, mo0.T.conj())

    mo0 = uwy.c[1][:, uwy.mo_occ[1]>0]
    mo0e = mo0 * (uwy.e[1][uwy.mo_occ[1]>0] * uwy.mo_occ[0][uwy.mo_occ[1]>0])
    dme_rks_b = numpy.dot(mo0e, mo0.T.conj())

    dme_rks = numpy.array([dme_rks_a, dme_rks_b]) 

    return dm, dme_rks


def get_veff1(uwy, g, dm, opts):
    """
    Get gradients of the effective potential.
    nabla veff = nabla v_0 + nabla v_bg.

    Args:
        uwy (WY) : uwy object.
        g (obj): PySCF gradient object.
        dm (ndarray): density matrix in AO.
        opts (dict): OEP options.

    Returns:
        Gradients of components of the effective potential.
    """
    n = uwy.mol.nelectron
    
    if opts['ReferencePotential'][0].lower() == 'hfx':
        # vj1, vk1 = g.get_jk(dm=uwy.mr.make_rdm1())
        vj1, vk1 = g.get_jk(dm=uwy.dm_hfx)
        v01 = vj1[0] + vj1[1] - vk1 

    elif opts['ReferencePotential'][0].lower() == 'fermi-amaldi':
        vj1, vk1 = g.get_jk(dm=uwy.dm_fa)
        v01 = vj1 * (uwy.mol.nelectron - 1) / uwy.mol.nelectron
        v01 = numpy.array([v01, v01])

    elif opts['ReferencePotential'][0].lower() == 'hfx-fa':
        vj1, vk1 = g.get_jk(dm=uwy.dm_hfx)
        v01a = vj1[0] + vj1[1] - vk1 
        v01a = v01a[0]
        vj1, vk1 = g.get_jk(dm=uwy.dm_fa)
        v01b = vj1 * (uwy.mol.nelectron - 1) / uwy.mol.nelectron
        v01 = numpy.array([v01a, v01b])

    elif opts['ReferencePotential'][0].lower() == 'sfa':
        vj1, vk1 = g.get_jk(dm=uwy.dm_fa)
        v01 = vj1 * uwy.sfa_factor
        v01 = numpy.array([v01, v01])

    elif opts['ReferencePotential'][0].lower() == 'shfx':
        vj1, vk1 = g.get_jk(dm=uwy.dm_hfx)
        v01 = vj1[0] + vj1[1] - vk1 * uwy.shfx_factor
        # v01 *= uwy.shfx_factor


    else:
        raise NotImplementedError
    
    vbg1 = numpy.array([
        build_vbg1(uwy.b_potential[0], uwy.integrals_3c1e_ip_ovlp),
        build_vbg1(uwy.b_potential[1], uwy.integrals_3c1e_ip_ovlp),
        ])
    
    return v01, vbg1
    

def Pulay_force(uwy, g, opts, dm, dme, debug=False):
    h1 = g.get_hcore()
    # h1 = (h1 + h1.transpose(0, 2, 1)) * 0.5
    s1 = g.get_ovlp()
    
    v01, vbg1 = get_veff1(uwy, g, dm, opts)
    veff1 = v01 + vbg1
    f1 = h1 + veff1

    dme_sf = dme[0] + dme[1]
    dm_sf = dm[0] + dm[1]

    offsetdic = uwy.mol.offset_nr_by_atom()

    Pulay_tf = numpy.zeros([uwy.mol.natm, 3])
    Pulay_ts = numpy.zeros([uwy.mol.natm, 3])
    if debug:
        th = numpy.zeros([uwy.mol.natm, 3])
        t0 = numpy.zeros([uwy.mol.natm, 3])
        tbg = numpy.zeros([uwy.mol.natm, 3])
    
    for ia in range(uwy.mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        Pulay_tf[ia] += einsum('sxij,sij->x', f1[:, :, p0:p1], dm[:, p0:p1]) * 2
        Pulay_ts[ia] -= einsum('xij,ij->x', s1[:, p0:p1], dme_sf[p0:p1]) * 2
        
        if debug:
            th[ia] += einsum('xij,ij->x', h1[:, p0:p1], dm_sf[p0:p1]) * 2 
            t0[ia] += einsum('sxij,sij->x', v01[:, :, p0:p1], dm[:, p0:p1]) * 2 
            tbg[ia] += einsum('sxij,sij->x', vbg1[:, :, p0:p1], dm[:, p0:p1]) * 2 
    
    if debug:
        print_mat(th, 'Pulay_th')
        print_mat(t0, 'Pulay_t0')
        print_mat(tbg, 'Pulay_tbg')
        print_mat(Pulay_tf, 'Pulay_tf')
        print_mat(Pulay_ts, 'Pulay_ts')
        print_mat(Pulay_tf+Pulay_ts, 'Pulay')
    
    return Pulay_tf + Pulay_ts


def calc_force(uwy, opts, debug=False):
    debug = True
    g = rks_grad.Gradients(uwy.mr)
    ## if opts['ReferencePotential'][0].lower() == 'hfx':
    ##     g = rks_grad.Gradients(uwy.mr)

    ## elif opts['ReferencePotential'][0].lower() == 'fermi-amaldi':
    ##     g = rks_grad.Gradients(uwy.mr)
    ##     
    ## elif opts['ReferencePotential'][0].lower() == 'hfx-fa':
    ##     g = rks_grad.Gradients(uwy.mr)

    ## else:
    ##     raise NotImplementedError

    HF = HF_force(uwy.mol, uwy.dm_oep[0]+uwy.dm_oep[1], debug)

    dm_rks, dme_rks = _make_rdm1e(uwy) 
    Pulay = Pulay_force(uwy, g, opts, dm_rks, dme_rks, debug)
    
    return HF + Pulay


def Pulay_term(uwy, v1, dm):
    t = numpy.zeros([uwy.mol.natm, 3])
    offsetdic = uwy.mol.offset_nr_by_atom()
    for ia in range(uwy.mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]
        t[ia] += einsum('xij,ij->x', v1[:, p0:p1], dm[p0:p1]) * 2 
    return t



