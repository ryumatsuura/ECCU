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

## A-1. load aggregated MOSAIKS features

eccu_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_nat.csv'), index_col = 0)
eccu_subnat_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat.csv'), index_col = 0)

## A-2. prediction!

## initialize dataframe
eccu_preds = pd.DataFrame([])
eccu_subnat_preds = pd.DataFrame([])

## specify outcome variables
tasks = ['hdi', 'health', 'income', 'ed']

for task in tasks:
    
    ## extract weights
    wts = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}.csv'.format(task)), delimiter = ',')
    wts_subnat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_{}.csv'.format(task)), delimiter = ',')
    
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
            eccu_preds['Country'] = df['Country'].values.tolist()
            eccu_preds['{}_preds'.format(task)] = ypreds.tolist()
        elif any(df.equals(y) for y in [eccu_subnat_feat]):
            eccu_subnat_preds['Country'] = df['Country'].values.tolist()
            eccu_subnat_preds['Name'] = df[['NAME_1']]
            eccu_subnat_preds['{}_preds'.format(task)] = ypreds.tolist()
        
        ## predict using subnational-level weights vector
        if any(df.equals(y) for y in [eccu_feat]):
            ypreds = np.dot(df.iloc[:, 1:4001], wts_subnat)
        elif any(df.equals(y) for y in [eccu_subnat_feat]):
            ypreds = np.dot(df.iloc[:, 2:4002], wts_subnat)
        
        ypreds[ypreds < lb_subnat] = lb_subnat
        ypreds[ypreds > ub_subnat] = ub_subnat
        
        if any(df.equals(y) for y in [eccu_feat]):
            eccu_preds['{}_preds_subnat'.format(task)] = ypreds.tolist()
        elif any(df.equals(y) for y in [eccu_subnat_feat]):
            eccu_subnat_preds['{}_preds_subnat'.format(task)] = ypreds.tolist()

###############################
## B) clean ground truth data 
###############################

## missing HDI for AIA and MSR
eccu_hdi = hdi.loc[['ATG', 'VCT', 'TTO', 'LCA', 'GRD', 'KNA', 'DMA', 'BRB']].rename(columns = {'Sub-national HDI': 'hdi', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})
eccu_hdi = eccu_hdi[['hdi', 'health', 'income', 'ed']].reset_index().rename(columns = {'ISO_Code': 'Country'})

## merge national-level HDI
merged = pd.merge(eccu_preds, eccu_hdi)

for task in tasks:
    
    ## plot prediction against 
    plt.clf()
    tot_min = np.min([np.min(np.array(merged['{}_preds'.format(task)])), np.min(np.array(merged[task]))])
    tot_max = np.max([np.max(np.array(merged['{}_preds'.format(task)])), np.max(np.array(merged[task]))])
    fig, ax = plt.subplots()
    ax.scatter(np.array(merged[task]), np.array(merged['{}_preds'.format(task)]))
    
    ## add 45 degree line and country names
    plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
    for i, txt in enumerate(np.array(merged['Country'])):
        ax.annotate(txt, (np.array(merged[task])[i], np.array(merged['{}_preds'.format(task)])[i]))
    
    ## add axis title
    if task == 'hdi':
        ax.set_xlabel('True {}'.format(task.upper()))
        ax.set_ylabel('Predicted {}'.format(task.upper()))
    else:
        ax.set_xlabel('True {} Index'.format(task.capitalize()))
        ax.set_ylabel('Predicted {} Index'.format(task.capitalize()))
    
    ## output the graph
    fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_nat_nat_{}.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)

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
for task in tasks:
    for x in ['{}_preds'.format(task), '{}_preds_subnat'.format(task)]:
    
        plt.clf()
        fig, ax = plt.subplots()
        eccu_shp.to_crs(epsg = 4326).plot(ax = ax, color = 'lightgrey')
        merged_shp.plot(column = x, ax = ax, cmap = 'RdYlGn', legend = True)
        fig.set_size_inches(30, 15, forward = True)
        if x == '{}_preds'.format(task):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_nat_{}.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
        elif x == '{}_preds_subnat'.format(task):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_subnat_{}.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)

