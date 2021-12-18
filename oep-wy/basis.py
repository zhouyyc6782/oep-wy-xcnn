from __future__ import print_function
from pyscf import gto, lib
import numpy as np
import ctypes
from utility import clean_array


def get_orbital_basis(atoms, basis_name):
    """
    get orbital basis from pyscf
    On output, basis is stored in internal format of pyscf, 
    i.e. 
        {
            'H': 
                [
                    [ANG, 
                        [exp, coef], 
                        [exp, coef], 
                        [exp, coef],
                        ...
                    ], 
                    [ANG, 
                        [exp, coef], 
                        [exp, coef], 
                        [exp, coef],
                        ...
                    ], 
                    ...
                ] # end of 'H'
            ...
        }
    """
    basis_tab = dict()
    for i in range(len(atoms)):
            basis_tab[atoms[i]] = basis_name
    orbital_basis = gto.format_basis(basis_tab)
    return orbital_basis


def get_potential_basis(atoms, basis_name):
    potential_basis = dict()
    for i in range(len(atoms)):
        atom = atoms[i]
        potential_basis[atom] = gto.load(basis_name, atom)
    return potential_basis


def get_overlap_g(mol, concatenate_basis, nbas_orbital_atom, nbas_potential_atom):
    """
        S_{tu}^{g} = <g_{t}|g_{u}>
    """

    nbas_potential = np.sum(nbas_potential_atom)
    aux = mol.copy()
    aux.basis = concatenate_basis
    aux.build()
    S = aux.intor("cint1e_ovlp_sph")
    Sg = np.empty([nbas_potential, nbas_potential], dtype=np.float)

    piS = 0; piSg = 0
    for ia in range(mol.natm):
        pjS = 0; pjSg = 0
        niorb = nbas_orbital_atom[ia]
        nipot = nbas_potential_atom[ia]
        for ja in range(mol.natm):
            njorb = nbas_orbital_atom[ja]
            njpot = nbas_potential_atom[ja]
            Sg[piSg:piSg+nipot, pjSg:pjSg+njpot] = \
                    S[piS+niorb:piS+niorb+nipot, pjS+njorb:pjS+njorb+njpot]

            pjS += njorb + njpot
            pjSg += njpot

        piS += niorb + nipot
        piSg += nipot

    return Sg


def get_kinetic_g(mol, concatenate_basis, nbas_orbital_atom, nbas_potential_atom):
    """
        T_{tu}^{g} = <g_{t}|hat{T}|g_{u}>
    """
    nbas_potential = np.sum(nbas_potential_atom)
    aux = mol.copy()
    aux.basis = concatenate_basis
    aux.build()
    T = aux.intor("cint1e_kin_sph")
    Tg = np.empty([nbas_potential, nbas_potential], dtype=np.float)
    piT = 0; piTg = 0
    for ia in range(mol.natm):
        pjT = 0; pjTg = 0
        niorb = nbas_orbital_atom[ia]
        nipot = nbas_potential_atom[ia]
        for ja in range(mol.natm):
            njorb = nbas_orbital_atom[ja]
            njpot = nbas_potential_atom[ja]
            Tg[piTg:piTg+nipot, pjTg:pjTg+njpot] = \
                    T[piT+niorb:piT+niorb+nipot, pjT+njorb:pjT+njorb+njpot]

            pjT += njorb + njpot
            pjTg += njpot

        piT += niorb + nipot
        piTg += nipot

    return Tg


def concatenate_basis(atoms, orbital_basis, potential_basis):
    """
    concatenate orbital basis and potential basis.
    both orbital_basis and potential_basis are in 
    internal format of PySCF. 
    On output, a combined basis is returned
    """
    
    conc_basis = {key: value[:] for key, value in orbital_basis.items()}
    uatoms = np.unique(atoms).tolist()
    for i in range(len(uatoms)):
        atom = uatoms[i]
        potential_basis_atom = potential_basis[atom]
        for basis in potential_basis_atom:
            conc_basis[atom].append(basis)

    return conc_basis


def _bas_length(basis):
    l = len(basis)
    for b in basis:
        if len(b[1]) > 2:
            l += 1
    return l


def get_integrals_3c1e(mol, atoms,
        orbital_basis_name, potential_basis_name):
    """
    generate integral table for \int xi_{i} xi_{j} \g_{t} dr
    
    """


    # get basis
    orbital_basis = get_orbital_basis(atoms, orbital_basis_name)

    potential_basis = get_potential_basis(atoms, potential_basis_name)

    # orbital alignment informations
    bas_start_orbital = np.zeros(len(atoms), dtype=np.int)
    bas_start_potential = np.zeros(len(atoms), dtype=np.int)
    for i in range(len(atoms)):
        if i == 0:
            bas_start_orbital[i] = 0
        else:
            bas_start_orbital[i] = bas_start_potential[i - 1] + _bas_length(potential_basis[atoms[i - 1]])

        bas_start_potential[i] = bas_start_orbital[i] + _bas_length(orbital_basis[atoms[i]])

    conc_basis = concatenate_basis(atoms, orbital_basis, potential_basis)
    nbas_conc = sum([_bas_length(conc_basis[atoms[i]]) for i in range(len(atoms))])
    basis_tag = np.ones(nbas_conc, dtype=np.int)
    # orbital basis: 0
    # potential basis: 1
    pi = 0
    for i in range(len(atoms)):
        di = _bas_length(orbital_basis[atoms[i]])
        basis_tag[pi:pi + di] = 0
        pi += _bas_length(conc_basis[atoms[i]])

    # assign concatenated basis to an auxiliary mole object
    aux = mol.copy()
    aux.basis = conc_basis
    aux.build()

    # components for basis function
    # s - 1; p - 3; d - 5, ...
    nbas_conc = sum([_bas_length(aux._basis[atoms[i]]) for i in range(len(atoms))])
    n_components = np.zeros([nbas_conc], dtype=np.int)
    orb_components = 0
    pot_components = 0
    k = 0
    for i in range(len(atoms)):
        atom = atoms[i]
        for basis in conc_basis[atom]:
            n_components[k] = 2 * basis[0] + 1
            if bas_start_orbital[i] <= k < bas_start_potential[i]:
                orb_components += 2 * basis[0] + 1
                if len(basis[1]) > 2:
                    orb_components += 2 * basis[0] + 1
            else:
                pot_components += 2 * basis[0] + 1
                if len(basis[1]) > 2:
                    pot_components += 2 * basis[0] + 1
            k += 1
            if len(basis[1]) > 2:
                n_components[k] = 2 * basis[0] + 1
                k += 1

    total_components = np.sum(n_components)
    assert(total_components == orb_components + pot_components)

    # let i and j go over all xi_{i}
    # let k goes over all g_{t}
    integral_table = aux.intor('int3c1e_sph')
    integrals_3c1e = np.zeros([orb_components, orb_components, pot_components])
    pi = 0; ii = 0
    for i in range(nbas_conc):
        di = n_components[i]
        if basis_tag[i] == 1: ii += di; continue

        pj = 0; jj = 0
        for j in range(nbas_conc):
            dj = n_components[j]
            if basis_tag[j] == 1: jj += dj; continue
            pk = 0; kk = 0
            for k in range(nbas_conc):
                dk = n_components[k]
                if basis_tag[k] == 0: kk += dk; continue
                integrals_3c1e[pi:pi + di, pj:pj + dj, pk:pk + dk] = integral_table[ii:ii + di, jj:jj + dj, kk:kk + dk]
                pk += dk; kk += dk;
            pj += dj; jj += dj;
        pi += di; ii += di;

    clean_array(integrals_3c1e)    
    return orbital_basis, potential_basis, conc_basis, integrals_3c1e


# int3c1e_ipovlp_sph
def get_integrals_3c1e_ip_ovlp(mol, atoms, 
        orbital_basis_name, potential_basis_name):
    """
    < nabla i | j | k > =
    \int nabla xi_{i} xi_{j} g_{t} dr
    """
    comp = 3


    # get basis
    orbital_basis = get_orbital_basis(atoms, orbital_basis_name)

    potential_basis = get_potential_basis(atoms, potential_basis_name)

    # orbital alignment informations
    bas_start_orbital = np.zeros(len(atoms), dtype=np.int)
    bas_start_potential = np.zeros(len(atoms), dtype=np.int)
    for i in range(len(atoms)):
        if i == 0:
            bas_start_orbital[i] = 0
        else:
            bas_start_orbital[i] = bas_start_potential[i - 1] + _bas_length(potential_basis[atoms[i - 1]])

        bas_start_potential[i] = bas_start_orbital[i] + _bas_length(orbital_basis[atoms[i]])

    conc_basis = concatenate_basis(atoms, orbital_basis, potential_basis)
    nbas_conc = sum([_bas_length(conc_basis[atoms[i]]) for i in range(len(atoms))])
    basis_tag = np.ones(nbas_conc, dtype=np.int)
    # orbital basis: 0
    # potential basis: 1
    pi = 0
    for i in range(len(atoms)):
        di = _bas_length(orbital_basis[atoms[i]])
        basis_tag[pi:pi + di] = 0
        pi += _bas_length(conc_basis[atoms[i]])

    # assign concatenated basis to an auxiliary mole object
    aux = mol.copy()
    aux.basis = conc_basis
    aux.build()

    # components for basis function
    # s - 1; p - 3; d - 5, ...
    nbas_conc = sum([_bas_length(aux._basis[atoms[i]]) for i in range(len(atoms))])
    n_components = np.zeros([nbas_conc], dtype=np.int)
    orb_components = 0
    pot_components = 0
    k = 0
    for i in range(len(atoms)):
        atom = atoms[i]
        for basis in conc_basis[atom]:
            n_components[k] = 2 * basis[0] + 1
            if bas_start_orbital[i] <= k < bas_start_potential[i]:
                orb_components += 2 * basis[0] + 1
                if len(basis[1]) > 2:
                    orb_components += 2 * basis[0] + 1
            else:
                pot_components += 2 * basis[0] + 1
                if len(basis[1]) > 2:
                    pot_components += 2 * basis[0] + 1
            k += 1
            if len(basis[1]) > 2:
                n_components[k] = 2 * basis[0] + 1
                k += 1
    total_components = np.sum(n_components)
    assert(total_components == orb_components + pot_components)

    # let i and j go over all xi_{i}
    # let k goes over all g_{t}

    integral_table = aux.intor('int3c1e_ipovlp_sph')
    integrals_3c1e = np.zeros([comp, orb_components, orb_components, pot_components])
    pi = 0; ii = 0
    for i in range(nbas_conc):
        di = n_components[i]
        if basis_tag[i] == 1: ii += di; continue
        pj = 0; jj = 0
        for j in range(nbas_conc):
            dj = n_components[j]
            if basis_tag[j] == 1: jj += dj; continue

            pk = 0; kk = 0
            for k in range(nbas_conc):
                dk = n_components[k]
                if basis_tag[k] == 0: kk += dk; continue
                integrals_3c1e[0, pi:pi + di, pj:pj + dj, pk:pk + dk] = integral_table[0, ii:ii + di, jj:jj + dj, kk:kk + dk]
                integrals_3c1e[1, pi:pi + di, pj:pj + dj, pk:pk + dk] = integral_table[1, ii:ii + di, jj:jj + dj, kk:kk + dk]
                integrals_3c1e[2, pi:pi + di, pj:pj + dj, pk:pk + dk] = integral_table[2, ii:ii + di, jj:jj + dj, kk:kk + dk]
                pk += dk; kk += dk
            pj += dj; jj += dj
        pi += di; ii += di

    clean_array(integrals_3c1e)    
    return orbital_basis, potential_basis, conc_basis, integrals_3c1e


def reshape_integral_table_NN_T(integrals_3c1e):
    N, T = integrals_3c1e.shape[0], integrals_3c1e.shape[2]
    table = np.empty([N * N, T])
    for i in range(T):
        table[:, i] = (integrals_3c1e[:, :, i].reshape(N * N))[:]
    return table


