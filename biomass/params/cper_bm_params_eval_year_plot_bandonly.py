import os

prefix = 'plot'
suffix = 'bandonly'

nickname = 'cper_bm_' + prefix
kfold_group = 'Year'
tuneby_group = 'Year'

logo_group = 'kfold'
mod_col = 'Source'

inDIR = './results/tmp'

#drop_cols = ['Date', 'lat', 'long', 'Low', 'High', 'PP_g', 'ID_yr']
drop_cols = ['Low', 'High', 'Date']
id_cols = ['kfold', 'Id', 'Pasture', 'Date_mean', 'Year', 'Season', 'Observed']
if logo_group in id_cols:
    id_cols.remove(logo_group)

# unique sub-plot ID column name
id_col_sub = None
# unique plot ID column name
id_col = 'Id'
# date column name
date_col = 'Date_mean'
# pasture column name
past_col = 'Pasture'
# grouping columns (e.g., for use when multiple dates might exist when grouping plots to pastures)
group_cols = ['Year', 'Season']

plot_group_cols = [mod_col, id_col]

# dependent variable column
y_col = 'Biomass_kg_ha'

inPATH = os.path.join(inDIR, 
                      'vor_2014_2023_cln_2024_04_04_' + prefix + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_' + suffix + '_tmp.csv')

var_names = [
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]