#!/usr/bin/env python

from optparse import OptionParser
import sys
import numpy as np
import scipy.stats as st
import os
import pandas as pd
import logging
import math

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def truncate_eigenvalues(d):
    M = len(d)

    # order evaules in descending order
    d[::-1].sort()

    #running_sum = 0
    d_trun = np.zeros(M)

    # keep only positive evalues
    for i in range(0,M):
        if d[i] > 0:
            # keep evalue
            d_trun[i] = d[i]

    return d_trun


def truncate_matrix(V):
    # make V pos-semi-def
    d, Q = np.linalg.eigh(V, UPLO='U')

    # reorder eigenvectors from inc to dec
    idx = d.argsort()[::-1]
    Q[:] = Q[:, idx]

    # truncate small eigenvalues for stability
    d_trun = truncate_eigenvalues(d)

    # mult decomp back together to get final V_trunc
    M1 = np.matmul(Q, np.diag(d_trun))
    V_trun = np.matmul(M1, np.matrix.transpose(Q))

    return V_trun


def simulate_locus(p_vec, h1, h2, rho, rho_e, N1, N2, Ns, M, V):

    # simulate C assignments
    p00 = p_vec[0]
    p10 = p_vec[1]
    p01 = p_vec[2]
    p11 = p_vec[3]

    C = np.random.multinomial(1, p_vec, M)
    C1 = np.empty(M)
    C2 = np.empty(M)

    for m in range(0, M):
        if C[m, 0] == 1:
            C1[m] = 0
            C2[m] = 0
        elif C[m, 1] == 1:
            C1[m] = 1
            C2[m] = 0
        elif C[m, 2] == 1:
            C1[m] = 0
            C2[m] = 1
        else:
            C1[m] = 1
            C2[m] = 1

    mu = [0, 0]
    sig_11 = h1 / (M * (p11 + p10))
    sig_22 = h2 / (M * (p11 + p01))
    sig_12 = (math.sqrt(h1) * math.sqrt(h2) * rho)/(M*p11)
    sig_21 = sig_12
    cov = [[sig_11, sig_12], [sig_21, sig_22]]
    gamma = np.random.multivariate_normal(mu, cov, M)

    beta1 = np.empty(M)
    beta2 = np.empty(M)
    for m in range(0, M):
        beta1[m] = gamma[m, 0] * (C[m, 1] + C[m, 3])
        beta2[m] = gamma[m, 1] * (C[m, 2] + C[m, 3])

    # ground truth vector
    C[:, 0] = np.multiply(0, C[:,0])
    C[:, 1] = np.multiply(1, C[:,1])
    C[:, 2] = np.multiply(2, C[:,2])
    C[:, 3] = np.multiply(3, C[:,3])

    C_true = np.sum(C, axis=1) # sum across the rows

    sigma_e_1 = (1 - h1) / N1
    sigma_e_2 = (1 - h2) / N1
    cov_e = rho_e*math.sqrt(sigma_e_1)*math.sqrt(sigma_e_2)
    sigma_e_s = (cov_e*Ns)/float(N1*N2)

    sig_e_11 = np.multiply(V, sigma_e_1)
    sig_e_22 = np.multiply(V, sigma_e_2)
    sig_e_12 = np.multiply(V, sigma_e_s)
    sig_e_21 = sig_e_12
    sigma_e_cov = np.block([[sig_e_11, sig_e_12], [sig_e_21, sig_e_22]])
    mu_e = np.block([np.matmul(V, beta1), np.matmul(V, beta2)])

    z_M = st.multivariate_normal.rvs(mu_e, sigma_e_cov)
    z1 = z_M[:M]
    z2 = z_M[M:]

    return z1, z2, C_true


def main():
    parser = OptionParser()
    parser.add_option("--name", dest="name", default="sim")
    parser.add_option("--p_sim", dest="p_sim", default=".95,.01,.01,.03")
    parser.add_option("--h1_sim", dest="h1_sim", default="0.05")
    parser.add_option("--h2_sim", dest="h2_sim", default="0.05")
    parser.add_option("--rhoG_sim", dest="rhoG_sim", default="0")
    parser.add_option("--rhoE_sim", dest="rhoE_sim", default="0")
    parser.add_option("--N1", dest="N1", default="100000")
    parser.add_option("--N2", dest="N2", default="100000")
    parser.add_option("--Ns", dest="Ns", default="0")
    parser.add_option("--M", dest="M", default="100")
    parser.add_option("--ld_file", dest="ld_file", default="/Users/ruthiejohnson/Development/g_coloc/data/identity.100.ld")
    parser.add_option("--seed", dest="seed", default="2018")
    parser.add_option("--outdir", dest="outdir", default="/Users/ruthiejohnson/Development/g_coloc/data")

    (options, args) = parser.parse_args()

    # get loci simulation values
    name = options.name
    outdir = options.outdir
    seed = int(options.seed)
    h1_sim = float(options.h1_sim)
    h2_sim = float(options.h2_sim)
    rhoG_sim = float(options.rhoG_sim)
    rhoE_sim = float(options.rhoE_sim)
    N1 = int(options.N1)
    N2 = int(options.N2)
    Ns = int(options.Ns)
    M = int(options.M)

    # set seed
    np.random.seed(seed)
    
    # read in and parse p_sim
    p_sim= [float(item) for item in options.p_sim.split(',')]

    # read in LD matrix
    ld_file = options.ld_file
    if ld_file is None:
        # use identity
        logging.info("Did not provide LD matrix...simulating with no-LD")
        V = np.eye(M)
    else:
        try:
            V_raw = np.loadtxt(ld_file)
            # truncate matrix to make pos-semi def
            logging.info("Truncating matrix to ensure pos-semi-def")
            V = truncate_matrix(V_raw)

        except:
            logging.info("LD file does not exist...will simulate with no LD")
            V = np.eye(M)

    p_sim = options.p_sim
    p_sim= [float(item) for item in options.p_sim.split(',')]

    # make sure p-vec sums to 1
    if np.sum(p_sim) != 1:
        logging.info("ERROR: p vector does NOT sum to 1...exiting")
        exit(1)

    # make sure all necessary values are given
    # TODO

    z1, z2, C = simulate_locus(p_sim, h1_sim, h2_sim, rhoG_sim, rhoE_sim, N1, N2, Ns, M, V)

    # save values in dataframe
    df = {'BETA_STD_1': z1, 'BETA_STD_2': z2, 'C': C}
    gwas_df = pd.DataFrame(data=df)

    outfile = os.path.join(outdir, name +'.'+str(seed) + '.txt')
    gwas_df.to_csv(outfile, index=False, sep=' ')

    logging.info("DONE simulating gwas...files can be found at: %s" % outfile)


if __name__== "__main__":
  main()
