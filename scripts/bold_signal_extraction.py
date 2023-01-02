import numpy as np
import os
import sys
from nilearn import input_data



def mask_computation(func_data_clean_reg_std, subject_IID, atlas_path, outdir):    
    
    
    # init path where to extract the desired files
    subject_IID_path = os.path.join(outdir, subject_IID)
    
    # create subject s folder
    if not os.path.exists(subject_IID_path):
        os.makedirs(subject_IID_path)

    
    masker = input_data.NiftiMapsMasker(
        maps_img      = atlas_path,
        #allow_overlap = False,
        standardize   = True,
        t_r           = 0.735,
        memory_level  = 0,
        verbose       = 2)
    

    # create an output folder
    output = os.path.join(subject_IID_path, 'time_series')
    os.makedirs(output)
    
    
    try:
        time_series = masker.fit_transform(func_data_clean_reg_std)
        np.savez(os.path.join(output, subject_IID), time_series)
    except:
        print("Oops! Something went wrong here ...")  
    
    
    
if __name__ == '__main__':
    
    subject_IID = sys.argv[1]
    atlas_path  = sys.argv[2]
    outdir      = sys.argv[3]
    
    func_data_clean_reg_std = os.path.join('/neurospin/ukb/derivatives/ukb_ica/sub-'+subject_IID+'/ses-2/func/rfMRI.ica/reg_standard/filtered_func_data_clean.nii.gz')
    
    mask_computation(func_data_clean_reg_std, subject_IID, atlas_path, outdir)
        
