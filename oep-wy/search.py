from __future__ import print_function
from math import sqrt
import numpy

from build_matrix import build_grad_general, build_grad_with_reg, build_Hess_general, build_Hess_with_reg, build_vbg
from solve_KS import solve_KS
from utility import clean_array
from constrain import *

from pyscf.grad import rks as rks_grad
from wy_force import Pulay_term
from wy_force import get_veff1


einsum = numpy.einsum
LINE_SEARCH_CONV_CRIT = 1.E-4
LINE_SEARCH_ALPHA  = 1.E-4
LINE_SEARCH_STEP_MAX = 1.E99


def solve_SVD(A, b, cutoff):
    s, U = numpy.linalg.eigh(A)
    VT = U.transpose()
    U[:, s > 0] *= -1
    s = -abs(s)

    for i in range(len(s)):
        if abs(s[i]) >= cutoff:
            s[i] = 1. / s[i]
        else:
            s[i] = 0.

    invA = einsum('ij,j->ij', VT.T, s)
    invA = einsum('ij,jk->ik', invA, U.T)
    invA = (invA + invA.T) * 0.5
    
    x = einsum('ij,j->i', invA, b)
    return x


def calc_Ws_Ts(T, v, dm, ddm):
    Ts = einsum('ij,ji->', T, dm)
    Ws = Ts + einsum('ij,ji->', v, ddm)
    return Ws, Ts


def initialize(wy):
    if 'ZeroForce' in wy.constrains:
        if wy.reference_potential == 'hfx':
            assert(wy.mr.xc.lower() == 'hf,')
            from pyscf.grad import rks as rks_grad
            wy.gmr = rks_grad.Gradients(wy.mr)
            vj, vk = wy.gmr.get_jk()
            wy.v01 = vj - vk * 0.5

            force_v0_atom = Pulay_term(wy, wy.v01, wy.dm_in)
            wy.force_v0 = numpy.sum(force_v0_atom, axis=0)

        wy.constrain_kwargs = {
                'dm0': wy.dm_in,
                'Sg': wy.Sg,
                'integrals_3c1e_ip_ovlp': wy.integrals_3c1e_ip_ovlp,
                'force_v0': -wy.force_v0,
                }
        wy.improve_b = improve_b_zf
    else:
        wy.constrain_kwargs = {}
        wy.improve_b = improve_b_none
    
    if 'LambdaRegulation' in wy.constrains:
        wy.build_gradient_matrix = build_grad_with_reg
        wy.build_Hessian_matrix = build_Hess_with_reg
    else:
        wy.build_gradient_matrix = build_grad_general
        wy.build_Hessian_matrix = build_Hess_general

    wy.b_potential = wy.b_init.copy()

    # END of initialize()


def search(wy):
    initialize(wy)

    print('Start searching...')
    print('# iter      sum abs grad         max abs grad            # elec    lambda')
    print('--------------------------------------------------------------------------------')

    converged = False
    for niter in range(wy.oep_max_iter):
        clean_array(wy.b_potential)

        wy.b_potential = wy.improve_b(wy.b_potential, **(wy.constrain_kwargs))

        wy.e, wy.c, wy.dm, wy.vbg = solve_KS(
                wy.b_potential, 
                wy.integrals_3c1e_ovlp, 
                wy.H, wy.S, 
                wy.mo_occ)

        wy.gradient = wy.build_gradient_matrix(wy)
        
        print('%5d\t%16.8e\t%16.8e\t%.6lf' % 
                (
                    niter+1, 
                    numpy.sum(abs(wy.gradient)), 
                    numpy.max(abs(wy.gradient)), 
                    numpy.einsum('ij,ji->', wy.dm, wy.S),
                    ), 
                end='')
        
        if numpy.all(abs(wy.gradient) < wy.oep_crit):
            converged = True
            print()
            break
        
        wy.Hessian = wy.build_Hessian_matrix(wy)

        update_direction = -solve_SVD(wy.Hessian, wy.gradient, wy.svd_cutoff)
        clean_array(update_direction)
        res = line_search(wy, update_direction)
        wy.b_potential, update_lambda, wy.gradient, stop_iter = res
        
        if stop_iter:
            print('\n\tSeems to find solution. Check...')
            if numpy.all(abs(wy.gradient) < wy.oep_crit): 
                converged = True
            else:
                print('\tReaches accuracy limit at current setup...Halt...')
                converged = False
            break
        else:
            print('%12.4e' % (update_lambda))
    
    print('OEP done.')

    force_v0_atom = numpy.zeros([wy.mol.natm, 3])
    offsetdic = wy.mol.offset_nr_by_atom()
    
    wy.b_potential = wy.improve_b(wy.b_potential, **(wy.constrain_kwargs))

    Ws, Ts = calc_Ws_Ts(
            wy.T, 
            wy.vn+wy.v0+wy.vbg, 
            wy.dm,
            wy.dm-wy.dm_in,
            )
    print('Stat:\tWs: %16.8e\tTs: %16.8e\tDiff: %16.8e' % (Ws, Ts, Ws - Ts))

    # END of search()


def line_search(wy, p):
    """
    Line search.

    Args:
        wy (object): WY instance.
        p (ndarray): update direction.
    """
    
    b_old = wy.b_potential
    gradient = wy.gradient
    Hessian = wy.Hessian

    slope = -numpy.dot(gradient, gradient)
    f_old = -0.5 * slope

    lambda1 = 1.0
    b_new = numpy.empty(b_old.shape, dtype=float)
    f2 = 0.0; lambda2 = 0.0

    while True:
        b_new = b_old + lambda1 * p
        wy.b_potential = b_new

        res = solve_KS(
                wy.b_potential, 
                wy.integrals_3c1e_ovlp, 
                wy.H, wy.S, 
                wy.mo_occ)
        wy.e, wy.c, wy.dm, wy.vbg = res
        
        g_new = wy.build_gradient_matrix(wy)
        f_new = 0.5 * numpy.dot(g_new, g_new)

        if lambda1 < LINE_SEARCH_CONV_CRIT:
            return b_new, lambda1, g_new, True
        if f_new <= f_old + LINE_SEARCH_ALPHA * lambda1 * slope:
            return b_new, lambda1, g_new, False
        if lambda1 == 1.0:
            tmp_lambda1 = -slope / (2.0 * (f_new - f_old - slope))
        else:
            rhs1 = f_new - f_old - lambda1 * slope
            rhs2 = f2 - f_old - lambda2 * slope
            a = (rhs1 / (lambda1 * lambda1) - rhs2 / (lambda2 * lambda2)) / (lambda1 - lambda2)
            b = (-lambda2 * rhs1 / (lambda1 * lambda1) + lambda1 * rhs2 / (lambda2 * lambda2)) / (lambda1 - lambda2)
            if abs(a) < 1.e-10:
                tmp_lambda1 = -slope / (2.0 * b)
            else:
                disc = b * b - 3.0 * a * slope
                if disc < 0.0:
                    tmp_lambda1 = 0.5 * lambda1
                elif b <= 0.0:
                    tmp_lambda1 = (-b + sqrt(disc)) / (3.0 * a)
                else:
                    tmp_lambda1 = -slope / (b + sqrt(disc))
                if tmp_lambda1 > 0.5 * lambda1:
                    tmp_lambda1 = 0.5 * lambda1
        lambda2 = lambda1
        f2 = f_new
        lambda1 = max(tmp_lambda1, 0.1 * lambda1)
    
    
