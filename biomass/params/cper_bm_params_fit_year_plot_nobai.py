import os
import re
import pandas as pd
import numpy as np

# NOTES:

# which data to use
prefix = 'plot'

# suffix for extra analyses
suffix = 'nobai'

# dask cluster location
cluster_loc = 'hpc'

# cross-validation
tuneby_group = 'Year'
kfold_group = 'Year'

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

outFILE_tmp = os.path.join(outDIR, 'tmp', re.sub('hls_idxs.csv', 'cv_' + kfold_group + '_tuneby_' + tuneby_group + '_' + suffix + '_tmp.csv', inFILE))

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
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]

def load_df(inPATH, date_cols=[date_col]):
    # Preprocessing steps here
    df = pd.read_csv(inPATH, parse_dates=date_cols)
    
    df = df[df['Season'].isin(['June', 'October'])].copy()

    df = df[~df[var_names].isnull().any(axis=1)].copy()
    
    return df