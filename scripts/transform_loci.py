#!/usr/bin/env python

from optparse import OptionParser
import numpy as np
import scipy.stats as st
import scipy.linalg
import math
import sys
import os
import logging
import pandas as pd
import glob

# set up global logging
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


def truncate_matrix_half(V):
    # make V pos-semi-def
    d, Q = np.linalg.eigh(V, UPLO='U')

    # reorder eigenvectors from inc to dec
    idx = d.argsort()[::-1]
    Q[:] = Q[:, idx]
    #d[:] = d.argsort()[::-1]

    # truncate small eigenvalues for stability
    d_trun = truncate_eigenvalues(d)

    d_trun_half = np.sqrt(d_trun)

    # mult decomp back together to get final V_trunc
    M1 = np.matmul(Q, np.diag(d_trun_half))
    V_trun_half = np.matmul(M1, np.matrix.transpose(Q))

    return V_trun_half


# calculate the negative-1/2 power of a matrix and then performs matrix truncation to ensure matrix is positive semi-definite
def truncate_matrix_neg_half(V):
    # make V pos-semi-def
    d, Q = np.linalg.eigh(V, UPLO='U')

    # reorder eigenvectors from inc to dec
    idx = d.argsort()[::-1]
    Q[:] = Q[:, idx]

    # truncate small eigenvalues for stability
    d_trun = truncate_eigenvalues(d)

    # square root of eigenvalues
    d_trun_half = np.sqrt(d_trun)

    # recipricol eigenvalues to do inverse
    d_trun_half_neg = np.divide(1, d_trun_half, where=d_trun_half!=0)


    # mult decomp back together to get final V_trunc
    M1 = np.matmul(Q, np.diag(d_trun_half_neg))
    V_trun_half_neg = np.matmul(M1, np.matrix.transpose(Q))

    return V_trun_half_neg


def convert_betas(z_b, ld_b):
    ld_b_neg_half = truncate_matrix_neg_half(ld_b)
    #ld_b_half = truncate_matrix_half(ld_b)
    #ld_b_neg_half = scipy.linalg.pinv(ld_b_half)
    z_b_twiddle = np.matmul(ld_b_neg_half, z_b)

    return z_b_twiddle


def main():
    parser = OptionParser()
    parser.add_option("--gwas_file", dest="gwas_file")
    parser.add_option("--ld_file", dest="ld_file")

    (options, args) = parser.parse_args()

    ld_file = options.ld_file
    gwas_file = options.gwas_file

    if gwas_file is not None and ld_file is not None:
        gwas_file_b = gwas_file
        ld_file_b = ld_file

        logging.info("gwas file: %s" % gwas_file_b)

        gwas_b = pd.read_table(gwas_file_b, sep=' ')
        z_b_1 = np.asarray(gwas_b['BETA_STD_1'])
        z_b_2 = np.asarray(gwas_b['BETA_STD_2'])
        C = np.asarray(gwas_b['C'])
        ld_b = np.loadtxt(ld_file_b)

        logging.info("Converting betas from file: %s" % os.path.basename(gwas_file_b))
        logging.info("Using ld file: %s" % os.path.basename(ld_file_b))

        z_b_twiddle_1 = convert_betas(z_b_1, ld_b)
        z_b_twiddle_2 = convert_betas(z_b_2, ld_b)

        # add converted betas to df
        gwas_b['BETA_STD_1_I'] = z_b_twiddle_1
        gwas_b['BETA_STD_2_I'] = z_b_twiddle_2

        # output file
        df = {'BETA_STD_1': z_b_1, 'BETA_STD_2': z_b_2, 'BETA_STD_1_I': z_b_twiddle_1, 'BETA_STD_2_I': z_b_twiddle_2, 'C': C}
        df_locus = pd.DataFrame(data=df)
        df_locus.to_csv(gwas_file_b, sep=' ', index=False)

        logging.info("Saving locus to: %s" % gwas_file_b)

        logging.info("FINISHED converting betas to transformed betas")

    else:

        logging.info("ERROR: user did not provide ld/gwas dir or ld/gwas filenames...exiting")
        exit(1)




if __name__== "__main__":
  main()
