import os
import re
import pandas as pd
import numpy as np
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# NOTES:

# which data to use
prefix = 'transect'

# dask cluster location
cluster_loc = 'hpc'

# cross-validation
tuneby_group = 'Kmeans'
kfold_group = 'Kmeans'

# one of 'logo' or 'group_k'
kfold_type = 'logo'
tune_kfold_type = 'logo'

use_cuda = False
# set backend as one of 'multiprocessing' or 'dask'
backend = 'dask' 

inDIR = '../data/training/'
inFILE = 'vor_2014_2023_cln_2024_04_04_' + prefix + '_hls_idxs.csv'
nickname = 'cper_bm_' + prefix

inPATH = os.path.join(inDIR, inFILE)

outDIR = './results/'

outFILE_tmp = os.path.join(outDIR, 'tmp', re.sub('hls_idxs.csv', 'cv_' + kfold_group + '_tuneby_' + tuneby_group + '_tmp.csv', inFILE))

# unique ID column name
id_col = 'Id'
# date column name
date_col = 'Date_mean'
# dependent variable column
y_col = 'Biomass_kg_ha'

# apply transformation to dependent variable
y_col_xfrm = False
def y_col_xfrm_func(x):
    return(x*10)

# apply transformation to the output of the CPER (2022) model
cper_mod_xfrm = False
def cper_mod_xfrm_func(x):
    # convert from kg/ha to lbs/acre
    return(x * 0.892179122)

var_names = [
    'NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 'SAVI',
    'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
    'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346',
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]

def load_df(inPATH, date_cols=[date_col]):
    # Preprocessing steps here
    df = pd.read_csv(inPATH, parse_dates=date_cols)
    
    df = df[df['Season'].isin(['June', 'October'])].copy()

    df = df[~df[var_names + [y_col]].isnull().any(axis=1)].copy()
    df['sqrt_Biomass_kg_ha'] = np.sqrt(df['Biomass_kg_ha']) 
    
    df['Plot'] = df['Id'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    scaler = StandardScaler().fit(df[var_names])
    X_scaled = scaler.transform(df[var_names])
    
    pc = PLSRegression(n_components=len(var_names)//2)
    pc.fit(X_scaled, df['sqrt_Biomass_kg_ha'])
    scores = pc.transform(scaler.transform(df[var_names]))
    
    df_plot_scores = df.copy()
    for i in range(scores.shape[1]):
        df_plot_scores['PC' + str(i)] = scores[:,i]
    
    df_plot_scores = df_plot_scores.groupby(['Plot', 'Date_mean'])[[x for x in df_plot_scores.columns if 'PC' in x]].mean()#.reset_index()
    
    centroid, label, inertia = k_means(df_plot_scores, n_clusters=10)
    
    df_plot_scores['Kmeans'] = label.astype('str')
    
    df_plot_scores = df_plot_scores.reset_index()
    df = pd.merge(df, df_plot_scores[['Plot', 'Date_mean', 'Kmeans']], how='left', on=['Plot', 'Date_mean'])
    
    return df