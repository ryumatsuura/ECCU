## This script aggregates MOSAIKS features to national 
## and subnational levels

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from pathlib import Path
from mosaiks import transforms
from mosaiks.utils.imports import *
from shapely.geometry import Point

##################################
## A) aggregate MOSAIKS features
##################################

## A-1. ECCU countries at national/subnational level

## initialize dataframes
eccu_feat = pd.DataFrame([])
eccu_subnat_feat = pd.DataFrame([])

## get features in ECCU countries
for files in os.listdir(c.features_dir):
    if files.startswith('Mosaiks_features') and files.endswith('.csv'):
        
        ## skip Barbados, Guadeloupe, and Martinique
        if files in ['Mosaiks_features_brb.csv', 'Mosaiks_features_glp.csv', 'Mosaiks_features_mtq.csv']:
            continue
        
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

## A-2. Barbados enumeration district

## load MOSAIKS feature
mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_brb.csv'))

## load population for MOSAIKS feature coords and merge in population
mosaiks_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'population', 'pop_Mosaiks_features_brb.csv'), index_col = 0).reset_index(drop = True)
mosaiks_feat_pop = pd.merge(mosaiks_feat, mosaiks_pop, on = ['Lon', 'Lat'])

## load enumeration district shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'BRB_voronoi_poly.shp'))

## convert MOSAIKS feature to geopandas framework
geometry = [Point(xy) for xy in zip(mosaiks_feat_pop.Lon, mosaiks_feat_pop.Lat)]
mosaiks_feat_gdf = gpd.GeoDataFrame(mosaiks_feat_pop.drop(['Lon', 'Lat', 'BoxLabel'], axis = 1), crs = shp.crs, geometry = geometry)

## spatially match MOSAIKS feature and subnational polygons
sjoin = gpd.sjoin(mosaiks_feat_gdf, shp[['psu', 'geometry']], how = 'left', op = 'within')
sjoin = sjoin[sjoin['psu'].notna()]

## initialize dataframes
brb_ed_feat = pd.DataFrame([])

u, indices = np.unique(sjoin['psu'], return_inverse = True)
for x in np.unique(indices):
    
    ## compute subnational average
    sjoin_group = sjoin.loc[indices == x]
    feat_ed_means = np.average(sjoin_group.iloc[:, 0:4000], weights = sjoin_group['population'], axis = 0)
    
    ## convert results to dataframe
    ed_df = pd.DataFrame(feat_ed_means).transpose()
    ed_df.insert(0, 'psu', [int(u[x])])
    
    ## append
    brb_ed_feat = pd.concat([brb_ed_feat, ed_df])

## reindex
brb_ed_feat = brb_ed_feat.reset_index(drop = True)

## compute the national average
feat_means = np.average(mosaiks_feat_gdf.iloc[:, 0:4000], weights = mosaiks_feat_gdf['population'], axis = 0)
feat_means_matrix = pd.DataFrame(np.resize(feat_means, (brb_ed_feat.shape[0], 4000)))

## demean MOSAIKS features
brb_ed_demean_feat = brb_ed_feat.iloc[:, 1:4001] - feat_means_matrix
brb_ed_demean_feat = brb_ed_demean_feat.merge(brb_ed_feat['psu'], left_index = True, right_index = True)
brb_ed_demean_feat.insert(0, 'psu', brb_ed_demean_feat.pop('psu'))

## save aggregated MOSAIKS features
brb_ed_feat.to_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_brb_ed.csv'), sep = ',')
brb_ed_demean_feat.to_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_brb_ed_demeaned.csv'), sep = ',')

## A-3. St Lucia settlements

## load MOSAIKS feature
mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_lca.csv'))

## load population for MOSAIKS feature coords and merge in population
mosaiks_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'population', 'pop_Mosaiks_features_lca.csv'), index_col = 0).reset_index(drop = True)
mosaiks_feat_pop = pd.merge(mosaiks_feat, mosaiks_pop, on = ['Lon', 'Lat'])

## load enumeration district shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'lca_admbnda_gov_2019_shp', 'lca_admbnda_adm2_gov_2019.shp'))

## convert MOSAIKS feature to geopandas framework
geometry = [Point(xy) for xy in zip(mosaiks_feat_pop.Lon, mosaiks_feat_pop.Lat)]
mosaiks_feat_gdf = gpd.GeoDataFrame(mosaiks_feat_pop.drop(['Lon', 'Lat', 'BoxLabel'], axis = 1), crs = shp.crs, geometry = geometry)

## spatially match MOSAIKS feature and subnational polygons
sjoin = gpd.sjoin(mosaiks_feat_gdf, shp[['ADM1_PCODE', 'ADM1_EN', 'ADM2_PCODE', 'ADM2_EN', 'SETTLECODE', 'geometry']], how = 'left', op = 'within')
sjoin = sjoin[sjoin['ADM1_PCODE'].notna()]
sjoin['SETTLECODE'] = sjoin['SETTLECODE'].astype(int)
sjoin['settlement'] = sjoin['ADM1_PCODE'] + sjoin['SETTLECODE'].apply(lambda x: '{0:0>9}'.format(x))

## initialize dataframes
lca_settle_feat = pd.DataFrame([])

u, indices = np.unique(sjoin['settlement'], return_inverse = True)
for x in np.unique(indices):
    
    ## compute subnational average
    sjoin_group = sjoin.loc[indices == x]
    feat_settle_means = np.average(sjoin_group.iloc[:, 0:4000], weights = sjoin_group['population'], axis = 0)
    
    ## convert results to dataframe
    settle_df = pd.DataFrame(feat_settle_means).transpose()
    settle_df.insert(0, 'ADM1_Code', sjoin_group['ADM1_PCODE'].values[0])
    settle_df.insert(1, 'Settle_Code', sjoin_group['SETTLECODE'].values[0])
    
    ## append
    lca_settle_feat = pd.concat([lca_settle_feat, settle_df])

## reindex
lca_settle_feat = lca_settle_feat.reset_index(drop = True)

## compute the national average
feat_means = np.average(mosaiks_feat_gdf.iloc[:, 0:4000], weights = mosaiks_feat_gdf['population'], axis = 0)
feat_means_matrix = pd.DataFrame(np.resize(feat_means, (lca_settle_feat.shape[0], 4000)))

## demean MOSAIKS features
lca_settle_demean_feat = lca_settle_feat.iloc[:, 2:4002] - feat_means_matrix
lca_settle_demean_feat = lca_settle_demean_feat.merge(lca_settle_feat[['ADM1_Code', 'Settle_Code']], left_index = True, right_index = True)
lca_settle_demean_feat.insert(0, 'ADM1_Code', lca_settle_demean_feat.pop('ADM1_Code'))
lca_settle_demean_feat.insert(1, 'Settle_Code', lca_settle_demean_feat.pop('Settle_Code'))

## save aggregated MOSAIKS features
lca_settle_feat.to_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_lca_settle.csv'), sep = ',')
lca_settle_demean_feat.to_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_lca_settle_demean.csv'), sep = ',')
