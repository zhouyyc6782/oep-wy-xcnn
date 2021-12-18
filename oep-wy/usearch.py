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
        # f = s[i] * s[i] / (s[i] * s[i] + cutoff * cutoff)
        # s[i] = f / s[i]

    invA = einsum('ij,j->ij', VT.T, s)
    invA = einsum('ij,jk->ik', invA, U.T)
    invA = (invA + invA.T) * 0.5
    
    x = einsum('ij,j->i', invA, b)
    return x


def calc_Ws_Ts(T, v, dm, ddm):
    Ts = einsum('ij,ji->', T, dm)
    Ws = Ts + einsum('ij,ji->', v, ddm)
    return Ws, Ts


def initialize(uwy):
    if False and 'ZeroForce' in wy.constrains:
        if uwy.reference_potential == 'hfx':
            assert(uwy.mr.xc.lower() == 'hf,')
            from pyscf.grad import rks as rks_grad
            uwy.gmr = rks_grad.Gradients(wy.mr)
            vj, vk = uwy.gmr.get_jk()
            uwy.v01 = vj - vk * 0.5

            force_v0_atom = Pulay_term(uwy, uwy.v01, uwy.dm_in)
            uwy.force_v0 = numpy.sum(force_v0_atom, axis=0)

        uwy.constrain_kwargs = {
                'dm0': uwy.dm_in,
                'Sg': uwy.Sg,
                'integrals_3c1e_ip_ovlp': uwy.integrals_3c1e_ip_ovlp,
                'force_v0': -uwy.force_v0,
                }
        uwy.improve_b = improve_b_zf
    else:
        uwy.constrain_kwargs = {}
        uwy.improve_b = improve_b_none
    
    if 'LambdaRegulation' in uwy.constrains:
        uwy.build_gradient_matrix = build_grad_with_reg
        uwy.build_Hessian_matrix = build_Hess_with_reg
    else:
        uwy.build_gradient_matrix = build_grad_general
        uwy.build_Hessian_matrix = build_Hess_general

    uwy.b_potential = uwy.b_init.copy()
    uwy.e = [list(), list()]
    uwy.c = [list(), list()]
    uwy.vbg = [list(), list()]

    # END of initialize()


def search(uwy, init=True):
    if init: initialize(uwy)

    idx = 0 if init else 1
    if init: print('Start searching...')
    print('# iter      sum abs grad         max abs grad            # elec    lambda')
    print('--------------------------------------------------------------------------------')

    converged = False
    for niter in range(uwy.oep_max_iter):
        clean_array(uwy.b_potential[idx])

        uwy.b_potential[idx] = uwy.improve_b(uwy.b_potential[idx], **(uwy.constrain_kwargs))

        uwy.e[idx], uwy.c[idx], uwy.dm, uwy.vbg[idx] = solve_KS(
                uwy.b_potential[idx], 
                uwy.integrals_3c1e_ovlp, 
                uwy.H[idx], uwy.S, 
                uwy.mo_occ[idx])

        uwy.gradient = uwy.build_gradient_matrix(uwy, idx=idx)
        
        print('%5d\t%16.8e\t%16.8e\t%.6lf' % 
                (
                    niter+1, 
                    numpy.sum(abs(uwy.gradient)), 
                    numpy.max(abs(uwy.gradient)), 
                    numpy.einsum('ij,ji->', uwy.dm, uwy.S),
                    ), 
                end='')
        
        if numpy.all(abs(uwy.gradient) < uwy.oep_crit):
            converged = True
            print()
            break
        
        uwy.Hessian = uwy.build_Hessian_matrix(uwy, idx) 

        update_direction = -solve_SVD(uwy.Hessian, uwy.gradient, uwy.svd_cutoff)
        clean_array(update_direction)
        res = line_search(uwy, update_direction, idx)
        uwy.b_potential[idx], update_lambda, uwy.gradient, stop_iter = res
        
        if stop_iter:
            print('\n\tSeems to find solution. Check...')
            if numpy.all(abs(uwy.gradient) < uwy.oep_crit): 
                converged = True
            else:
                print('\tReaches accuracy limit at current setup...Halt...')
                converged = False
            break
        else:
            print('%12.4e' % (update_lambda))
    
    print('OEP done.')

    force_v0_atom = numpy.zeros([uwy.mol.natm, 3])
    offsetdic = uwy.mol.offset_nr_by_atom()
    
    uwy.b_potential[idx] = uwy.improve_b(uwy.b_potential[idx], **(uwy.constrain_kwargs))

    Ws, Ts = calc_Ws_Ts(
            uwy.T, 
            uwy.vn+uwy.v0[idx]+uwy.vbg[idx], 
            uwy.dm,
            uwy.dm-uwy.dm_in[idx],
            )
    print('Stat:\tWs: %16.8e\tTs: %16.8e\tDiff: %16.8e' % (Ws, Ts, Ws - Ts))

    # END of search()


def line_search(uwy, p, idx):
    """
    Line search.

    Args:
        wy (object): WY instance.
        p (ndarray): update direction.
    """
    
    b_old = uwy.b_potential[idx]
    gradient = uwy.gradient
    Hessian = uwy.Hessian

    slope = -numpy.dot(gradient, gradient)
    f_old = -0.5 * slope

    lambda1 = 1.0
    b_new = numpy.empty(b_old.shape, dtype=float)
    f2 = 0.0; lambda2 = 0.0

    while True:
        b_new = b_old + lambda1 * p
        uwy.b_potential[idx] = b_new

        res = solve_KS(
                uwy.b_potential[idx], 
                uwy.integrals_3c1e_ovlp, 
                uwy.H[idx], uwy.S, 
                uwy.mo_occ[idx])
        uwy.e[idx], uwy.c[idx], uwy.dm, uwy.vbg[idx] = res
        
        g_new = uwy.build_gradient_matrix(uwy, idx=idx)
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
    
    
