import numpy as np
import glob
import os
import sys
import subprocess
from nilearn import input_data
from multiprocessing import Pool



def mask_computation(parameters):    
    
    individual_nifti, subject_IID, atlas_path, outdir = parameters
    
    # init path where to extract the desired files
    subject_IID_path = os.path.join(outdir, subject_IID)
    
    # create folder for each subject
    if not os.path.exists(subject_IID_path):
        os.makedirs(subject_IID_path)
    
    # extract the desired data
    cmd = " ".join(['/usr/bin/unzip %s' % individual_nifti,
                ' fMRI/rfMRI.ica/filtered_func_data_clean.nii.gz',
                ' fMRI/rfMRI.ica/reg/*mat',
                ' fMRI/rfMRI.ica/reg/*.nii.gz',
                '-d  %s' % subject_IID_path])
    try:
        p = subprocess.check_call(cmd, shell=True)
    except:
        print("Oops! individual with no resting state fMRI data ...")
        
    
    # create reg_standard folder
    os.makedirs(os.path.join(subject_IID_path, 'fMRI/rfMRI.ica/reg_standard'))
    
    
    # => MNI space
    fname        = os.path.join(subject_IID_path, "fMRI/rfMRI.ica/filtered_func_data_clean.nii.gz")
    warp         = os.path.join(subject_IID_path, "fMRI/rfMRI.ica/reg/example_func2standard_warp")
    standard     = os.path.join(subject_IID_path, "fMRI/rfMRI.ica/reg/example_func2standard")
    outdir       = os.path.join(subject_IID_path, "fMRI/rfMRI.ica/reg_standard")
    reg_standard = os.path.join(outdir, "filtered_func_data_clean")

    
    dir_template = "/path/bin/fsl/data/standard"

    cmd = " ".join(["/data100t1/home/mekkiy/bin/fsl/bin/applywarp", #fsl5.0-applywarp
                    "--ref=%s"% standard,
                    "--in=%s"% fname,
                    "--out=%s" % reg_standard,
                    "--warp=%s"% warp,
                    "--interp=spline",
                    "--verbose"])
    try:
        p = subprocess.check_call(cmd, shell=True)
    except:
        print("Error: applywarp")
        


    mask = os.path.join(dir_template, "MNI152_T1_2mm_brain_mask")
    cmd  = " ".join(["/data100t1/home/mekkiy/bin/fsl/bin/fslmaths", #fsl5.0-fslmaths
                     reg_standard,
                     "-mas %s" % mask,
                     reg_standard])
    
    try:
        p = subprocess.check_call(cmd, shell=True)
    except:
        print("Error: fslmaths")
    
    
    
    func_data_clean_reg_std = reg_standard+'.nii.gz'
    
    # create an output folder
    output = os.path.join(subject_IID_path, 'time_series')
    os.makedirs(output)
    
    
    # extract mean bold signal for each ROIs
    masker = input_data.NiftiMapsMasker(
        maps_img      = atlas_path,
        allow_overlap = False,
        standardize   = True,
        t_r           = 0.735,
        memory_level  = 0,
        verbose       = 2)    
    
    try:
        time_series = masker.fit_transform(func_data_clean_reg_std)
        np.savez(os.path.join(output, subject_IID), time_series)
    except:
        print("Oops! Something went wrong when extracting the mean bold signal")
    
    
    
    # mv the reg_standard data
    cmd = " ".join(['mv %s ' % os.path.join(subject_IID_path, 'fMRI/rfMRI.ica/reg_standard'),
                   ' %s' % os.path.join(subject_IID_path, 'fMRI')])
    
    try:
        p = subprocess.check_call(cmd, shell=True)
    except:
        print("Error when moving the reg_standard data")
    
    
    # remove the unzipped data
    cmd = " ".join(['rm -r %s' % os.path.join(subject_IID_path, 'fMRI/rfMRI.ica')])
    
    try:
        p = subprocess.check_call(cmd, shell=True)
    except:
        print("Error when removing the unzipped data")
    
    
    
if __name__ == '__main__':
    
    rsfMRI_data_path = sys.argv[1]
    atlas_path       = sys.argv[2]
    extraction_tmp   = sys.argv[3]
    
    
    rsfMRI_data_paths = glob.glob(os.path.join(rsfMRI_data_path, '*.zip'))
    
    
    list_cmd = []
    for individual_nifti in rsfMRI_data_paths:
        subject_IID = os.path.basename(individual_nifti).split('.zip')[0]
        list_cmd.append([individual_nifti, subject_IID, atlas_path, outdir])
        
        
    pool = Pool(processes = 32)
    pool.map(mask_computation, list_cmd)
    pool.close()
    pool.join()

    
    
    
    