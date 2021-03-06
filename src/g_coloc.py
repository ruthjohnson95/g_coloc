#/usr/bin/env python

from optparse import OptionParser
import sys
import numpy as np
import scipy.stats as st
import os
import pandas as pd
import logging
import math

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

COLOC_THRESH=.90

"""
    prints both to console and to outfile with file descriptor f
"""
def print_func(line, f):
    print(line)
    sys.stdout.flush()
    f.write(line)
    f.write('\n')
    return


def initialize(beta_tilde_1, beta_tilde_2, h1, h2, N1, N2):
    M = len(beta_tilde_1)
    thresh = 4.0 # fixed amount
    C00, C10, C01, C11 = 1,1,1,1 # make sure bin has at least one entry
    z1 = np.multiply(beta_tilde_1, math.sqrt(N1))
    z2 = np.multiply(beta_tilde_2, math.sqrt(N2))
    C_1_init = np.zeros(M)
    C_2_init = np.zeros(M)

    for m in range(0, M):
        z1_m = beta_tilde_1[m]
        z2_m = beta_tilde_2[m]

        if z1_m > thresh and z2_m > thresh:
            C11 += 1
            C_1_init[m] = 1
            C_2_init[m] = 1
        elif z1_m > thresh and z2_m < thresh:
            C10 += 1
            C_1_init[m] = 1
            C_2_init[m] = 0
        elif z1_m < thresh and z2_m > thresh:
            C01 += 1
            C_1_init[m] = 0
            C_2_init[m] = 1
        else:
            C00 += 1
            C_1_init[m] = 0
            C_2_init[m] = 0

    M = len(beta_tilde_1)

    p_init = np.divide([C00, C10, C01, C11], float(M))

    sigma_g_1 = h1 / (M*(p_init[1]+p_init[3]))
    sigma_g_2 = h2 / (M*(p_init[2]+p_init[3]))
    gamma_1_init = st.norm.rvs(0, sigma_g_1, size=M)
    gamma_2_init = st.norm.rvs(0, sigma_g_2, size=M)

    return p_init, gamma_1_init, gamma_2_init, C_1_init, C_2_init


def three_matrix_mul(A, B, C):
    BC = np.matmul(B,C)
    ABC = np.matmul(A, BC)
    return ABC


def calc_rm(m, z, W, gamma, C):
    # gamma and C are (2m x 1)
    gamma_C_t = np.multiply(gamma, C)
    r_m_1 = np.subtract(z, np.matmul(W, gamma_C_t))
    W_m = np.vstack([W[:,m], W[:,2*m]])
    gamma_m = [gamma[m], gamma[2*m]]
    r_m_2 = np.matmul(np.transpose(W_m), gamma_m)
    r_m = np.add(r_m_1, r_m_2)
    return r_m


def calc_mu_sigma_m(m, gamma, C, z, sigma_1_g, W, Sigma_e, trait):
    Wtt_m = W[:, trait*m] # column vector
    r_m = calc_rm(m, z, W, gamma, C)
    inv_sigma_m = 1/(sigma_1_g) + np.matmul(np.transpose(Wtt_m), np.matmul(np.linalg.inv(Sigma_e), Wtt_m))
    sigma_1_m = 1/(inv_sigma_m)
    mu_1_m = (sigma_1_m)*(three_matrix_mul(Wtt_m, np.linalg.inv(Sigma_e), r_m))

    return mu_1_m, sigma_1_m


def calc_mu_sigma_joint(m, gamma, C, z, Sigma_g, W, Sigma_e):
    r_m = calc_rm(m, z, W, gamma, C)
    W_m = np.vstack([W[:,m], W[:, 2*m]])
    A_m = three_matrix_mul(W_m, np.linalg.inv(Sigma_e), r_m)
    B_m = np.add(three_matrix_mul(W_m, np.linalg.inv(Sigma_e), np.transpose(W_m)), np.linalg.inv(Sigma_g))
    Sigma_m = np.linalg.inv(B_m)
    Mu_m = np.matmul(Sigma_m, A_m)
    return Mu_m, Sigma_m


def sample_C_gamma_m(m, p_t, C_1_t, C_2_t, gamma_1_t, gamma_2_t, h1, h2, rho, rho_e, N1, N2, Ns, W, z1, z2):

    # get number of SNPs
    M = len(z1)

    p00, p10, p01, p11 = p_t

    # genetic variance terms
    sigma_1_g = h1/(M * (p11 + p10))
    sigma_2_g = h2/(M * (p11 + p01))
    sigma_12_g = (math.sqrt(h1) * math.sqrt(h2) * rho)/(M*p11)
    sigma_21_g = sigma_12_g
    Sigma_g = [[sigma_1_g, sigma_12_g], [sigma_21_g, sigma_2_g]]

    sigma_e_1 = (1 - h1) / N1
    sigma_e_2 = (1 - h2) / N1
    cov_e = rho_e*math.sqrt(sigma_e_1)*math.sqrt(sigma_e_2)
    sigma_e_s = (cov_e*Ns)/float(N1*N2)

    sig_e_11 = np.eye(M) * sigma_e_1
    sig_e_22 = np.eye(M) * sigma_e_2
    sig_e_12 = np.eye(M) * sigma_e_s
    sig_e_21 = sig_e_12
    Sigma_e = np.block([[sig_e_11, sig_e_12], [sig_e_21, sig_e_22]])

    # posterior effect sizes
    gamma = np.hstack([gamma_1_t, gamma_2_t])
    C = np.hstack([C_1_t, C_2_t])
    z = np.hstack([z1, z2])

    trait = 1
    mu_1_m, sigma_1_m = calc_mu_sigma_m(m, gamma, C, z, sigma_1_g, W, Sigma_e, trait)
    trait = 2
    mu_2_m, sigma_2_m = calc_mu_sigma_m(m, gamma, C, z, sigma_2_g, W, Sigma_e, trait)
    mu_m, Sigma_m = calc_mu_sigma_joint(m, gamma, C, z, Sigma_g, W, Sigma_e)

    # bernoulli dist
    det1 = np.linalg.det(np.multiply(Sigma_m, 2*np.pi))
    det2 = np.linalg.det(np.multiply(Sigma_g, 2*np.pi))
    const = np.sqrt(det1/det2)
    matrix_term = np.matmul(np.transpose(mu_m), np.matmul(np.linalg.inv(Sigma_m), mu_m))
    a11 = (p11)*const*np.exp(0.50*(matrix_term))

    const = np.sqrt(sigma_1_m/sigma_1_g)
    a10 = p10*const*np.exp(sigma_1_m*0.50 * (mu_1_m * mu_1_m))

    const = np.sqrt(sigma_2_m/sigma_2_g)
    a01 = p01*const*np.exp(sigma_2_m*0.50 * (mu_2_m * mu_2_m))

    a00 = p00

    denom = a11 + a10 + a01 + a00
    a_vec = np.divide([a00, a10, a01, a11], denom)

    C_t_m = st.multinomial.rvs(n=1, p=a_vec, size=1)
    C_t_m = C_t_m.ravel()

    if C_t_m[0] == 1:
        C_1m_t = 0
        C_2m_t = 0
    elif C_t_m[1] == 1:
        C_1m_t = 1
        C_2m_t = 0
    elif C_t_m[2] == 1:
        C_1m_t = 0
        C_2m_t = 1
    else:
        C_1m_t = 1
        C_2m_t = 1

    gamma_1m_t = st.norm.rvs(mu_1_m, sigma_1_m)*C_1m_t
    gamma_2m_t = st.norm.rvs(mu_2_m, sigma_2_m)*C_2m_t

    return C_1m_t, C_2m_t, gamma_1m_t, gamma_2m_t


def gibbs(p_init, gamma_1_init, gamma_2_init, C_1_init, C_2_init, h1, h2, rho, rho_e, N1, N2, Ns, W, z1, z2, its, f):

    # get number of SNPs
    M = len(z1)

    # make lists to hold samples
    p_list = []
    C_list = np.zeros((M, 4))

    gamma_1_t = gamma_1_init
    gamma_2_t = gamma_2_init
    C_1_t = C_1_init
    C_2_t = C_2_init
    p_t = p_init

    logging.info("Starting sampler")
    for i in range(0, its):

        # keep track of causal status counts
        C00_counts, C10_counts, C01_counts, C11_counts = 0,0,0,0

        for m in range(0, M):
            # sample C and gamma
            C_1m_t, C_2m_t, gamma_1m_t, gamma_2m_t = sample_C_gamma_m(m, p_t, C_1_t, C_2_t, gamma_1_t, gamma_2_t, h1, h2, rho, rho_e, N1, N2, Ns, W, z1, z2)

            # update values
            gamma_1_t[m] = gamma_1m_t
            gamma_2_t[m] = gamma_2m_t
            C_1_t[m] = C_1m_t
            C_2_t[m] = C_2m_t

            # keep track of C values for post prob
            if C_1m_t == 0 and C_2m_t == 0:
                C00_counts += 1
            elif C_1m_t == 1 and C_2m_t == 0:
                C10_counts += 1
            elif C_1m_t == 0 and C_2m_t == 1:
                C01_counts += 1
            else:
                C11_counts += 1

            BURN = its/4
            if i >= BURN:
                # keep track of C values for post prob
                if C_1m_t == 0 and C_2m_t == 0:
                    C_list[m,0] += 1
                elif C_1m_t == 1 and C_2m_t == 0:
                    C_list[m,1] += 1
                elif C_1m_t == 0 and C_2m_t == 1:
                    C_list[m,2] += 1
                else:
                    C_list[m,3] += 1

        # end loop through SNPs

        # sample p
        alpha = np.add([C00_counts, C10_counts, C01_counts, C11_counts], np.ones(4)) # add prior
        p_t = st.dirichlet.rvs(alpha)
        p_t = p_t.ravel()
        p_list.append(p_t)

        # logging

        # TODO
        log_like = 0

        p_t_string = ""
        for p in p_t:
            p_t_string+=(str(p)+' ')

        print_func("Iteration %d: %s" % (i, p_t_string), f)
        print_func("Iteration %d (log-like): %4g" % (i, log_like), f)

        # end iterations

    # compute averages
    post_prob_C = np.empty((M, 4))
    coloc_status = np.ones(M)*5
    for m in range(0, M):
        post_prob_C[m, 0] = C_list[m,0]/float(its-BURN)
        post_prob_C[m, 1] = C_list[m,1]/float(its-BURN)
        post_prob_C[m, 2] = C_list[m,2]/float(its-BURN)
        post_prob_C[m, 3] = C_list[m,3]/float(its-BURN)

        if post_prob_C[m, 3] > COLOC_THRESH:
            coloc_status[m] = 3

    return post_prob_C, coloc_status


def main():
    parser = OptionParser()
    parser.add_option("--name", dest="name", default="sim")
    parser.add_option("--gwas_file", dest="gwas_file", default="/Users/ruthiejohnson/Development/g_coloc/data/sim.2018.txt")
    parser.add_option("--ld_half_file", dest="ld_half_file", default="/Users/ruthiejohnson/Development/g_coloc/data/ukbb.100.ld")
    parser.add_option("--seed", dest="seed", default=100)
    parser.add_option("--outdir", dest="outdir", default="/Users/ruthiejohnson/Development/g_coloc/results/")
    parser.add_option("--h1", dest="h1", default="0.15")
    parser.add_option("--h2", dest="h2", default="0.15")
    parser.add_option("--rhoG", dest="rhoG", default="0")
    parser.add_option("--rhoE", dest="rhoE", default="0")
    parser.add_option("--N1", dest="N1", default="500")
    parser.add_option("--N2", dest="N2", default="500")
    parser.add_option("--Ns", dest="Ns", default="0")
    parser.add_option("--M", dest="M", default="100")
    parser.add_option("--its", dest="its", default=100)
    (options, args) = parser.parse_args()

    # parse command line args
    name = options.name
    outdir = options.outdir
    gwas_file = options.gwas_file
    h1 = float(options.h1)
    h2 = float(options.h2)
    rhoG = float(options.rhoG)
    rhoE = float(options.rhoE)
    N1 = int(options.N1)
    N2 = int(options.N2)
    Ns = int(options.Ns)
    M = int(options.M)
    its = int(options.its)

    # set seed
    seed = int(options.seed)
    np.random.seed(seed)

    # log file
    logfile = os.path.join(outdir, name +'.'+str(seed)+'.gcoloc.log')
    f = open(logfile, 'w')

    # read in gwas
    df = pd.read_csv(gwas_file, sep=' ')
    beta_tilde_1 = np.asarray(df['BETA_STD_1_I'])
    beta_tilde_2 = np.asarray(df['BETA_STD_2_I'])
    C_true = np.asarray(df['C'])

    # read in LD file
    ld_half_file = options.ld_half_file
    W_ii = np.loadtxt(ld_half_file)
    zeros = np.zeros((M,M))
    W = np.block([[W_ii, zeros],[zeros,W_ii]])
    logging.info("Using ld half file: %s" % ld_half_file)

    # initialize values
    # TODO
    p_init, gamma_1_init, gamma_2_init, C_1_init, C_2_init = initialize(beta_tilde_1, beta_tilde_2, h1, h2, N1, N2)

    # Gibbs sampler
    post_prob_C, coloc_status = gibbs(p_init, gamma_1_init, gamma_2_init, C_1_init, C_2_init, h1, h2, rhoG, rhoE, N1, N2, Ns, W, beta_tilde_1, beta_tilde_2, its, f)

    # save posterior probability matrix
    r_df = {'C00': post_prob_C[:,0], 'C10': post_prob_C[:,1], 'C01': post_prob_C[:,2], 'C11': post_prob_C[:,3], 'COLOC': coloc_status}
    results_df = pd.DataFrame(data=r_df)
    results_file = os.path.join(outdir, name +'.'+str(seed)+'.results')
    results_df.to_csv(results_file, index=False, sep=' ')

    f.close()

    # measure accuracy
    accurate_inds = np.where(coloc_status == C_true)
    total_colocs = len(np.where(C_true == 3))
    accuracy = len(accurate_inds[0])/float(total_colocs)

    print "Percentage of colocs identified: %.4g" % accuracy

    #print "C00: %.4g (%.4g)" % (np.mean(post_prob_C[:,0]), np.var(post_prob_C[:,0]))
    #print "C10: %.4g (%.4g)" % (np.mean(post_prob_C[:,1]), np.var(post_prob_C[:,1]))
    #print "C01: %.4g (%.4g)" % (np.mean(post_prob_C[:,2]), np.var(post_prob_C[:,2]))
    #print "C11: %.4g (%.4g)" % (np.mean(post_prob_C[:,3]), np.var(post_prob_C[:,3]))

    print "Done."

if __name__== "__main__":
  main()
