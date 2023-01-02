import pandas as pd
import numpy as np
import os
from stats import flood_fill_hull
from scipy.spatial import ConvexHull
from nilearn.datasets import load_mni152_brain_mask
from scipy import ndimage
import nibabel as nib




supp_data_path = '../data/'
supp_phon = pd.read_csv(os.path.join(supp_data_path, 'L_phonology.csv'))
supp_sem  = pd.read_csv(os.path.join(supp_data_path, 'L_semantic.csv'))
supp_sent = pd.read_csv(os.path.join(supp_data_path, 'L_sentence.csv'))
supp_comm = pd.read_csv(os.path.join(supp_data_path, 'L_common.csv'))

# Merging
class_l = [supp_phon, supp_sem, supp_sent, supp_comm]
labels  = ['phonology', 'semantic', 'sentence', 'common']




###########################
### Format of the peaks ###
###########################

class_dict = {}
cat_dict =  {}

for j in range(len(class_l)):
    
    cluster_df = pd.DataFrame(columns=['x', 'y', 'z', 'ROI'])
    init = 0
    cluster_dict = {}
    
    for i in class_l[j].loc[class_l[j]['x'] == 0].index:
        if init == 0:
            tmp = class_l[j].iloc[0:i, :]
            roi = tmp.loc[tmp.index[0]]['Authors, Year']
            cluster_df.loc[0] = [tmp.loc[tmp.index[0]]['x'],
                                 tmp.loc[tmp.index[0]]['y'],
                                 tmp.loc[tmp.index[0]]['z'],
                                 tmp.loc[tmp.index[0]]['Authors, Year']]
            
            cluster_dict[roi] = tmp.drop(tmp.index[0])
            init = i+1
        else:
            tmp = class_l[j].iloc[init:i, :]
            roi = tmp.loc[tmp.index[0]]['Authors, Year']
            cluster_df.loc[cluster_df.index[-1]+1] = [tmp.loc[tmp.index[0]]['x'],
                                                      tmp.loc[tmp.index[0]]['y'],
                                                      tmp.loc[tmp.index[0]]['z'],
                                                      tmp.loc[tmp.index[0]]['Authors, Year']]
            cluster_dict[roi] = tmp.drop(tmp.index[0])
            init = i+1
    
    #Get the last line of the excel file
    tmp = class_l[j].iloc[init:class_l[j].index[-1]+1, :]
    roi = tmp.loc[tmp.index[0]]['Authors, Year']
    cluster_df.loc[cluster_df.index[-1]+1] = [tmp.loc[tmp.index[0]]['x'],
                                              tmp.loc[tmp.index[0]]['y'],
                                              tmp.loc[tmp.index[0]]['z'],
                                              tmp.loc[tmp.index[0]]['Authors, Year']]
    cluster_dict[roi] = tmp.drop(tmp.index[0])
    #End
    
    class_dict[labels[j]] = cluster_dict
    cat_dict[labels[j]] = cluster_df




###########################
####### hull convex #######
###########################
 
hull_convex_dict = {}
for key in class_dict:
    for sub_key in class_dict[key]:
        mni_img  = load_mni152_brain_mask()
        img_data = np.zeros((mni_img.shape))
        elm      = class_dict[key][sub_key]
        roi_vox  = np.zeros((elm.shape[0],3))
        
        for i in range(len(elm)):
            # from mni to voxel
            mni_coords = [elm.loc[elm.index[i]]['x'], elm.loc[elm.index[i]]['y'], elm.loc[elm.index[i]]['z'], 1]
            aff_mat_inf = np.linalg.inv(mni_img.affine)
            tmp = np.dot(aff_mat_inf, mni_coords)
            roi_vox[i] = [int(tmp[0]), int(tmp[1]) , int(tmp[2])]
            img_data[int(tmp[0]), int(tmp[1]) , int(tmp[2])] = 1
        # hull convex
        out, h = flood_fill_hull(img_data)
        hull_convex_dict[sub_key] = out
        
        
###########################
######### Opening #########
###########################        
        
hull_convex_dict_opening = {}
for key in hull_convex_dict:
    hull_convex_dict_opening[key] = ndimage.binary_opening(hull_convex_dict[key]).astype(int)
    
    
    
    
###########################
######## Overlapp #########
###########################      
    
non_iverlapping_hull_convex_dict_opening = {}
for elm in list_sorted: # list_sorted : sort the ROIs according to their number of voxels.
    principal_key = list(list_nb[elm].keys())[0]
    principal_roi = hull_convex_dict_opening[principal_key]
    
    for key in hull_convex_dict_opening:
        if key == principal_key:
            continue
            
        # remove the overlap
        hull_convex_dict_opening[key][np.where((principal_roi > 0) & (hull_convex_dict_opening[key] > 0))] = 0
        
        
###########################
########## Save ###########
###########################         
        
output = '/path/hull_convex_openning'
for key in hull_convex_dict:
    if "/" in key:
        key_tmp = key.replace("/", "_")
    else:
        key_tmp = key
    img = nib.Nifti1Image(hull_convex_dict[key], mni_img.affine, mni_img.header)
    img.to_filename(os.path.join(output, 'L_'+key_tmp+'.nii.gz'))