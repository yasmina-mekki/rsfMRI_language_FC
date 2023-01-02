import numpy as np
import pandas as pd
import scipy






def flood_fill_hull(image):
    '''
    Get a
    
    input : image 3D
    output : out_img 
           : hull 
    '''
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull

def summary(df):
    '''
    Get a statistical summary of a dataframe
    
    input : dataframe
    output : summary of the dataframe
    '''
    df_summary = df.describe()
    df_summary.loc['var'] = np.var(df)
    return df_summary

def check_symmetric(matrix, rtol=1e-05, atol=1e-08):
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

def upper_tri_masking(matrix):
    '''
    Get the upper triangle of a squared matrix
    
    input : square matrix
    output : upper triangle flattened
    '''
    m = matrix.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return matrix[mask]

def upper_tri_data_frame(df):
    '''
    Get the upper triangle of a squared dataframe with keeping the columns name and index
    
    input : dataframe
    output : type       : dataframe 
             dimentsion : len(df)*(len(df)-1)/2, 3
    '''
    df_flattened = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
    np.fill_diagonal(df_flattened.values, None)
    df_flattened = df_flattened.stack().reset_index()
    return df_flattened

def fisher_z_transformation(r_value):
    '''
    Permet de dilater les valeurs de correlations (r_value)
    Renvoi des valeurs plus espacer
    
    input : correlation values
    output : correlation values fisher z transformed
    '''
    return 0.5 * (np.log(1+r_value) - np.log(1-r_value))

def cov2corr(cov):
    '''
    Convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.
    '''
    
    cov  = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    return corr