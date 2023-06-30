## This script implements the prediction exercises

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os
import dill
import rtree
import zipfile
from pathlib import Path
from mosaiks import transforms
from mosaiks.utils.imports import *
from shapely.geometry import Point
from scipy.spatial import distance

## load HDI measures
hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'hdi', 'HDI_indicators_and_indices_adm0_clean.p'))
hdi_subnat = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'hdi', 'HDI_indicators_and_indices_clean.p'))
hdi_np = np.array(hdi['Sub-national HDI'])
hdi_subnat_np = np.array(hdi_subnat['Sub-national HDI'])

## set upper and lower bounds
lb = hdi_np.min(axis = 0)
ub = hdi_np.max(axis = 0)
lb_subnat = hdi_subnat_np.min(axis = 0)
ub_subnat = hdi_subnat_np.max(axis = 0)

###############
## A) predict
###############

## A-1. extract weights vectors

wts = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_hdi.csv'), delimiter = ',')
wts_subnat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_hdi.csv'), delimiter = ',')

## A-2. load aggregated MOSAIKS features

eccu_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_nat.csv'), index_col = 0)
eccu_subnat_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat.csv'), index_col = 0)

## A-3. prediction!

## loop over national level and subnational level predictions
for df in (eccu_feat, eccu_subnat_feat):
    
    ## predict using national-level weights vector
    if any(df.equals(y) for y in [eccu_feat]):
        ypreds = np.dot(df.iloc[:, 1:4001], wts)
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        ypreds = np.dot(df.iloc[:, 2:4002], wts)
    
    ## bound the prediction
    ypreds[ypreds < lb] = lb
    ypreds[ypreds > ub] = ub
    
    ## store predicted values
    if any(df.equals(y) for y in [eccu_feat]):
        eccu_preds = df[['Country']]
        eccu_preds['y_preds'] = ypreds.tolist()
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        eccu_subnat_preds = df[['Country']]
        eccu_subnat_preds['Name'] = df[['NAME_1']]
        eccu_subnat_preds['y_preds'] = ypreds.tolist()
    
    ## predict using subnational-level weights vector
    if any(df.equals(y) for y in [eccu_feat]):
        ypreds = np.dot(df.iloc[:, 1:4001], wts_subnat)
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        ypreds = np.dot(df.iloc[:, 2:4002], wts_subnat)
    
    ypreds[ypreds < lb_subnat] = lb_subnat
    ypreds[ypreds > ub_subnat] = ub_subnat
    
    if any(df.equals(y) for y in [eccu_feat]):
        eccu_preds['y_preds_subnat'] = ypreds.tolist()
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        eccu_subnat_preds['y_preds_subnat'] = ypreds.tolist()

###############################
## B) clean ground truth data 
###############################

## missing HDI for AIA and MSR
eccu_hdi = hdi.loc[['ATG', 'VCT', 'TTO', 'LCA', 'GRD', 'KNA', 'DMA', 'BRB']]
eccu_hdi = eccu_hdi['Sub-national HDI'].reset_index().rename(columns = {'ISO_Code': 'Country'})

## merge national-level HDI
merged = pd.merge(eccu_preds, eccu_hdi)

## plot prediction against 
plt.clf()
tot_min = np.min([np.min(np.array(merged['y_preds'])), np.min(np.array(merged['Sub-national HDI']))])
tot_max = np.max([np.max(np.array(merged['y_preds'])), np.max(np.array(merged['Sub-national HDI']))])
fig, ax = plt.subplots()
ax.scatter(np.array(merged['Sub-national HDI']), np.array(merged['y_preds']))

## add 45 degree line and country names
plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
for i, txt in enumerate(np.array(merged['Country'])):
    ax.annotate(txt, (np.array(merged['Sub-national HDI'])[i], np.array(merged['y_preds'])[i]))

## add axis title
ax.set_xlabel('True HDI')
ax.set_ylabel('Predicted HDI')

## output the graph
fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_nat_nat_hdi.png'), bbox_inches = 'tight', pad_inches = 0.1)

## subnational maps

## initialize geodataframe
eccu_shp = gpd.GeoDataFrame([])

for iso in ['ATG', 'VCT', 'TTO', 'LCA', 'GRD', 'KNA', 'DMA', 'BRB', 'AIA', 'MSR']:
    
    ## load subnational shapefile
    shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'gadm41_{}_shp'.format(iso), 'gadm41_{}_1.shp'.format(iso)))
    
    ## append shapefiles
    eccu_shp = pd.concat([eccu_shp, shp])

## reindex 
eccu_shp = eccu_shp[['GID_0', 'NAME_1', 'geometry']].reset_index(drop = True)
eccu_shp = eccu_shp.rename(columns = {'GID_0': 'Country', 'NAME_1': 'Name'})

## merge shapefile with predicted values
merged_shp = gpd.GeoDataFrame(pd.merge(eccu_subnat_preds, eccu_shp))

## visualization
for x in ['y_preds', 'y_preds_subnat']:
    
    plt.clf()
    fig, ax = plt.subplots()
    eccu_shp.to_crs(epsg = 4326).plot(ax = ax, color = 'lightgrey')
    merged_shp.plot(column = x, ax = ax, cmap = 'RdYlGn', legend = True)
    fig.set_size_inches(30, 15, forward = True)
    if x == 'y_preds':
        fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_nat_hdi.png'), bbox_inches = 'tight', pad_inches = 0.1)
    elif x == 'y_preds_subnat':
        fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_subnat_hdi.png'), bbox_inches = 'tight', pad_inches = 0.1)

