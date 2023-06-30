## This script aggregates MOSAIKS features to national 
## and subnational levels

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

##################################
## A) aggregate MOSAIKS features
##################################

## initialize dataframes
eccu_feat = pd.DataFrame([])
eccu_subnat_feat = pd.DataFrame([])

## get features in ECCU countries
for files in os.listdir(c.features_dir):
    if files.startswith('Mosaiks_features') and files.endswith('.csv'):
        
        ## load MOSAIKS feature
        mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, files))
        
        ## load population for MOSAIKS feature coords and merge in population
        mosaiks_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'population', "".join(['pop_', files])), index_col = 0).reset_index(drop = True)
        mosaiks_feat_pop = pd.merge(mosaiks_feat, mosaiks_pop, on = ['Lon', 'Lat'])
        
        ## extract country name
        ISO = files.replace('Mosaiks_features_', '', 1).replace('.csv', '', 1).upper()
        
        ## compute the national average of MOSAIKS feature weighted by population
        feat_means = np.average(mosaiks_feat_pop.iloc[:, 3:4003], weights = mosaiks_feat_pop['population'], axis = 0)
        
        ## compile dataframe
        country_df = pd.DataFrame(feat_means).transpose()
        country_df.insert(0, 'Country', [ISO])
        
        ## append 
        eccu_feat = pd.concat([eccu_feat, country_df])
        
        ## load subnational shapefile
        shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'gadm41_{}_shp'.format(ISO), 'gadm41_{}_1.shp'.format(ISO)))
        
        ## convert MOSAIKS feature to geopandas framework
        geometry = [Point(xy) for xy in zip(mosaiks_feat_pop.Lon, mosaiks_feat_pop.Lat)]
        mosaiks_feat_gdf = gpd.GeoDataFrame(mosaiks_feat_pop.drop(['Lon', 'Lat', 'BoxLabel'], axis = 1), crs = shp.crs, geometry = geometry)
        
        ## spatially match MOSAIKS feature and subnational polygons
        sjoin = gpd.sjoin(mosaiks_feat_gdf, shp[['GID_1', 'NAME_1', 'geometry']], how = 'left', op = 'within')
        sjoin = sjoin[sjoin['NAME_1'].notna()]
        
        u, indices = np.unique(sjoin['GID_1'], return_inverse = True)
        for x in np.unique(indices):
            
            ## compute subnational average
            sjoin_group = sjoin.loc[indices == x]
            feat_subnat_means = np.average(sjoin_group.iloc[:, 0:4000], weights = sjoin_group['population'], axis = 0)
            
            ## convert results to dataframe
            subnat_df = pd.DataFrame(feat_subnat_means).transpose()
            subnat_df.insert(0, 'GID_1', [u[x]])
            subnat_df.insert(0, 'Country', [ISO])
            subnat_df = subnat_df.merge(shp[['GID_1', 'NAME_1']])
            
            ## append
            eccu_subnat_feat = pd.concat([eccu_subnat_feat, subnat_df])

## reindex
eccu_feat = eccu_feat.reset_index(drop = True)
eccu_subnat_feat = eccu_subnat_feat.reset_index(drop = True)

## save aggregated MOSAIKS features
eccu_feat.to_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_nat.csv'), sep = ',')
eccu_subnat_feat.to_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat.csv'), sep = ',')
