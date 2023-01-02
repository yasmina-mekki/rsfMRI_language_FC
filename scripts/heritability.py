import numpy as np
import pandas as pd
import sys
import os
import tempfile

import subprocess
import time


def gcta_heritability_analysis(grm, pheno, covar, qcovar, output):
    """
    Parameters
    grm: genetic relationship matrix
    pheno: phenotype data
    covar: input discrete covariates
    qcovar: input quantitative covariates
    output: output path
    """


    cmd = " ".join(['/path/bin/gcta64', #gcta64_1.25.3
                        '--grm-gz %s' % grm,
                        '--pheno %s' % pheno,
                        '--covar %s' % covar,
                        '--qcovar %s' % qcovar,
                        '--reml',
                        '--out %s' %output,
                        #'--mpheno %d' % (1), # specify that we consider the n-th trait
                        '--thread-num 1'])

    p = subprocess.check_call(cmd, shell=True)
    

    
    
if __name__ == '__main__':
    
    
    phenotype_path     = sys.argv[1]
    covar_path         = sys.argv[2]
    qcovar_path        = sys.argv[3]
    center_covars_path = sys.argv[4]
    output_path        = sys.argv[5]
    kinship_m_path     = sys.argv[6]
    
    start_time = time.time()

    # load covars
    covar_df  = pd.read_csv(covar_path, sep="\t")
    qcovar_df = pd.read_csv(qcovar_path, sep="\t")
    covars_df = pd.merge(covar_df, qcovar_df, on=['IID', 'FID'])
    covars_df = covars_df[["FID", "IID", "Age", "Sex", "Array",
                           "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"]]
    
    # filter on individuals according the phenotype file 
    phenotype_df = pd.read_csv(phenotype_path, sep="\t")
    covars_df    = covars_df[covars_df["IID"].isin(phenotype_df["IID"])]


    # load center data
    center_covars_df = pd.read_csv(center_covars_path, sep=",")
    
    center_covars_df = pd.get_dummies(center_covars_df, columns=['Centre'])
    center_covars_df.rename(columns={'ukb_iid' : 'IID',
                                     'Centre_Cheadle (imaging)': 'Centre_Cheadle',
                                     'Centre_Newcastle (imaging)': 'Centre_Newcastle',
                                     'Centre_Reading (imaging)': 'Centre_Reading'}, inplace=True)
    
    # merge all covariates
    covars_df = pd.merge(covars_df, center_covars_df[['IID', 'Centre_Cheadle', 'Centre_Newcastle']], on=['IID'])
    
    # create covariates in gcta required format
    tmpdir = tempfile.mkdtemp()
    covars_tmp_path = os.path.join(tmpdir, 'covars.cov')
    qcovars_tmp_path = os.path.join(tmpdir, 'qcovars.cov')
    covars_df[["FID", "IID", "Sex", "Array", "Centre_Cheadle", 'Centre_Newcastle']].to_csv(covars_tmp_path, index=None, sep='\t')
    covars_df[["FID", "IID", "Age", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"]].to_csv(qcovars_tmp_path, index=None, sep='\t')
    
    

    # run the heritability analysis        
    gcta_heritability_analysis(kinship_m_path,
                               phenotype_path,
                               covars_tmp_path,
                               qcovars_tmp_path,
                               output_path+os.path.basename(phenotype_path))
            
    
    print("End -- time --- %s seconds ---" % (time.time() - start_time))       
    
    
    
