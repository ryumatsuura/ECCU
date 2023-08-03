## This script aggregates MOSAIKS features to national 
## and subnational levels

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv, warnings
from mosaiks import transforms
from mosaiks.utils.imports import *
from shapely.geometry import Point

def match_mosaiks_poly(shp, mosaiks_feat_gdf, col):
    
    ## spatially match MOSAIKS features with polygons
    sjoin = gpd.sjoin(mosaiks_feat_gdf, shp, how = 'left', op ='within')
    sjoin = sjoin[sjoin[col].notna()]
    
    ## initialize dataframes
    feats = pd.DataFrame([])
    
    ## loop over each index
    u, indices = np.unique(sjoin[col], return_inverse = True)
    for x in np.unique(indices):
        
        ## compute average for each polygon
        sjoin_group = sjoin.loc[indices == x]
        if sjoin_group['total_pop'].sum() > 0:
            feat_means = np.average(sjoin_group.iloc[:, 0:4000], weights = sjoin_group['total_pop'], axis = 0)
        else:
            feat_means = np.average(sjoin_group.iloc[:, 0:4000], axis = 0)
        
        ## convert results to dataframe
        df = pd.DataFrame(feat_means).transpose()
        df.insert(0, col, [u[x]])
        
        ## append 
        feats = pd.concat([feats, df])
    
    ## reindex 
    feats = feats.reset_index(drop = True)
    return feats

##################################
## A) aggregate MOSAIKS features
##################################

## A-1. ECCU countries at national/subnational level

## initialize dataframes
eccu_feat = pd.DataFrame([])
eccu_subnat_feat = pd.DataFrame([])
eccu_subnat_demean_feat = pd.DataFrame([])
eccu_mosaiks_feat = pd.DataFrame([])
eccu_mosaiks_demean_feat = pd.DataFrame([])

## get features in ECCU countries
for files in os.listdir(c.features_dir):
    if files.startswith('Mosaiks_features') and files.endswith('.csv'):
        
        ## load MOSAIKS feature
        mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, files))
        mosaiks_feat = mosaiks_feat.set_index(mosaiks_feat['Lat'].astype(str) + ':' + mosaiks_feat['Lon'].astype(str)).drop(columns = ['BoxLabel'])
        
        ## append
        eccu_mosaiks_feat = pd.concat([eccu_mosaiks_feat, mosaiks_feat.drop(columns = ['Lat', 'Lon'])])
        
        ## extract country name
        ISO = files.replace('Mosaiks_features_', '', 1).replace('.csv', '', 1).upper()
        
        ## load population for MOSAIKS feature coords and merge in population
        mosaiks_pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', "{}_mosaiks_population.pkl".format(ISO.lower())))
        mosaiks_feat_pop = mosaiks_feat.merge(mosaiks_pop['total_pop'], left_index = True, right_index = True)
        
        ## compute the national average of MOSAIKS features weighted by population
        feat_means = np.average(mosaiks_feat_pop.iloc[:, 2:4002], weights = mosaiks_feat_pop['total_pop'], axis = 0)
        
        ## compile dataframe
        country_df = pd.DataFrame(feat_means).transpose()
        country_df.insert(0, 'Country', [ISO])
        country_df = country_df.set_index('Country')
        
        ## append 
        eccu_feat = pd.concat([eccu_feat, country_df])
        
        ## load subnational shapefile
        shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'gadm41_{}_shp'.format(ISO), 'gadm41_{}_1.shp'.format(ISO)))
        
        ## convert MOSAIKS features to geopandas framework
        geometry = [Point(xy) for xy in zip(mosaiks_feat_pop.Lon, mosaiks_feat_pop.Lat)]
        mosaiks_feat_gdf = gpd.GeoDataFrame(mosaiks_feat_pop.drop(['Lon', 'Lat'], axis = 1), crs = shp.crs, geometry = geometry)
        
        ## spatially match MOSAIKS features and subnational polygons
        subnat_feat = match_mosaiks_poly(shp, mosaiks_feat_gdf, 'GID_1')
        
        ## extract subnat polygons that are not matched with MOSAIKS features
        unmatched_shp = shp.loc[~shp.GID_1.isin(subnat_feat.GID_1)]
        
        if unmatched_shp.shape[0] > 0:
            
            ## create buffer around the centroid
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                buffers = gpd.GeoDataFrame(geometry = unmatched_shp.buffer(0.00833333333333/2))
                buffers = buffers.merge(unmatched_shp['GID_1'], left_index = True, right_index = True)
            
            ## spatially match again
            buffer_subnat_feat = match_mosaiks_poly(buffers[['GID_1', 'geometry']], mosaiks_feat_gdf, 'GID_1')
            
            ## concatenate and set index
            subnat_all_feats = pd.concat([subnat_feat, buffer_subnat_feat])
            subnat_all_feats = subnat_all_feats.set_index('GID_1')
        
        else:
            subnat_all_feats = subnat_feat.set_index('GID_1')
        
        ## convert the national average to matrix and demean MOSAIKS features
        feat_means_matrix = pd.DataFrame(np.resize(feat_means, (subnat_all_feats.shape[0], 4000)), index = subnat_all_feats.index)
        subnat_demean_feats = subnat_all_feats - feat_means_matrix
        
        ## append
        eccu_subnat_feat = pd.concat([eccu_subnat_feat, subnat_all_feats])
        eccu_subnat_demean_feat = pd.concat([eccu_subnat_demean_feat, subnat_demean_feats])
        
        ## convert the national average to matrix again and demean MOSAIKS features
        feat_means_matrix = pd.DataFrame(np.resize(feat_means, (mosaiks_feat.shape[0], 4000)), index = mosaiks_feat.index)
        mosaiks_all_feats = mosaiks_feat.drop(columns = ['Lat', 'Lon'])
        mosaiks_all_feats.columns = [i for i in range(mosaiks_all_feats.shape[1])]
        mosaiks_demean_feats = mosaiks_all_feats - feat_means_matrix
        eccu_mosaiks_demean_feat = pd.concat([eccu_mosaiks_demean_feat, mosaiks_demean_feats])

## save aggregated MOSAIKS features
eccu_feat.to_pickle(os.path.join(c.features_dir, 'eccu_nat_mosaiks_features.pkl'))
eccu_subnat_feat.to_pickle(os.path.join(c.features_dir, 'eccu_subnat_mosaiks_features.pkl'))
eccu_subnat_demean_feat.to_pickle(os.path.join(c.features_dir, 'eccu_subnat_mosaiks_features_demeaned.pkl'))
eccu_mosaiks_feat.to_pickle(os.path.join(c.features_dir, 'eccu_mosaiks_mosaiks_features.pkl'))
eccu_mosaiks_demean_feat.to_pickle(os.path.join(c.features_dir, 'eccu_mosaiks_mosaiks_features_demeaned.pkl'))

## A-2. Barbados enumeration district

## load MOSAIKS feature
mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_brb.csv'))
mosaiks_feat = mosaiks_feat.set_index(mosaiks_feat['Lat'].astype(str) + ':' + mosaiks_feat['Lon'].astype(str)).drop(columns = ['BoxLabel'])

## load population for MOSAIKS feature coords and merge in population
mosaiks_pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'brb_mosaiks_population.pkl'))
mosaiks_feat_pop = mosaiks_feat.merge(mosaiks_pop['total_pop'], left_index = True, right_index = True)

## load enumeration district shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'brb_ed_voronoi_poly.shp'))

## convert MOSAIKS features to geopandas framework
geometry = [Point(xy) for xy in zip(mosaiks_feat_pop.Lon, mosaiks_feat_pop.Lat)]
mosaiks_feat_gdf = gpd.GeoDataFrame(mosaiks_feat_pop.drop(['Lon', 'Lat'], axis = 1), crs = shp.crs, geometry = geometry)

## spatially match MOSAIKS features and subnational polygons
brb_ed_feat = match_mosaiks_poly(shp, mosaiks_feat_gdf, 'psu')
brb_ed_feat['psu'] = brb_ed_feat.psu.astype(int)

## extract psu polygons that are not matched with MOSAIKS features
unmatched_shp = shp.loc[~shp.psu.isin(brb_ed_feat.psu)]

## create buffer around the centroid
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    buffers = gpd.GeoDataFrame(geometry = unmatched_shp.buffer(0.00833333333333/2))
    buffers = buffers.merge(unmatched_shp['psu'].astype(int), left_index = True, right_index = True)

## spatially match again
buffer_brb_ed_feat = match_mosaiks_poly(buffers[['psu', 'geometry']], mosaiks_feat_gdf, 'psu')
buffer_brb_ed_feat['psu'] = buffer_brb_ed_feat.psu.astype(int)

## concatenate and set index
brb_ed_all_feats = pd.concat([brb_ed_feat, buffer_brb_ed_feat])
brb_ed_all_feats = brb_ed_all_feats.set_index('psu')

## compute the national average
feat_means = np.average(mosaiks_feat_gdf.iloc[:, 0:4000], weights = mosaiks_feat_gdf['total_pop'], axis = 0)
feat_means_matrix = pd.DataFrame(np.resize(feat_means, (brb_ed_all_feats.shape[0], 4000)), index = brb_ed_all_feats.index)

## demean MOSAIKS features
brb_ed_demean_feats = brb_ed_all_feats - feat_means_matrix

## save aggregated MOSAIKS features
brb_ed_all_feats.to_pickle(os.path.join(c.features_dir, 'brb_ed_mosaiks_features.pkl'))
brb_ed_demean_feats.to_pickle(os.path.join(c.features_dir, 'brb_ed_mosaiks_features_demeaned.pkl'))

## A-3. St Lucia settlements

## load MOSAIKS feature
mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_lca.csv'))
mosaiks_feat = mosaiks_feat.set_index(mosaiks_feat['Lat'].astype(str) + ':' + mosaiks_feat['Lon'].astype(str)).drop(columns = ['BoxLabel'])

## load population for MOSAIKS feature coords and merge in population
mosaiks_pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'lca_mosaiks_population.pkl'))
mosaiks_feat_pop = mosaiks_feat.merge(mosaiks_pop['total_pop'], left_index = True, right_index = True)

## load enumeration district shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'lca_admbnda_gov_2019_shp', 'lca_admbnda_adm2_gov_2019.shp'))
shp = shp.dissolve(['ADM1_PCODE', 'SETTLECODE']).reset_index()
shp['settlement'] = shp['ADM1_PCODE'] + shp['SETTLECODE'].apply(lambda x: '{0:0>9}'.format(x))

## convert MOSAIKS features to geopandas framework
geometry = [Point(xy) for xy in zip(mosaiks_feat_pop.Lon, mosaiks_feat_pop.Lat)]
mosaiks_feat_gdf = gpd.GeoDataFrame(mosaiks_feat_pop.drop(['Lon', 'Lat'], axis = 1), crs = shp.crs, geometry = geometry)

## spatially match MOSAIKS features and subnational polygons
lca_settle_feat = match_mosaiks_poly(shp, mosaiks_feat_gdf, 'settlement')

## extract settlement polygons that are not matched with MOSAIKS features
unmatched_shp = shp.loc[~shp.settlement.isin(lca_settle_feat.settlement)]

## create buffer around the centroid
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    buffers = gpd.GeoDataFrame(geometry = unmatched_shp.buffer(0.00833333333333/2))
    buffers = buffers.merge(unmatched_shp['settlement'], left_index = True, right_index = True)

## spatially match again
buffer_lca_settle_feat = match_mosaiks_poly(buffers[['settlement', 'geometry']], mosaiks_feat_gdf, 'settlement')

## concatenate and set index
lca_settle_all_feats = pd.concat([lca_settle_feat, buffer_lca_settle_feat])
lca_settle_all_feats = lca_settle_all_feats.set_index('settlement')

## compute the national average
feat_means = np.average(mosaiks_feat_gdf.iloc[:, 0:4000], weights = mosaiks_feat_gdf['total_pop'], axis = 0)
feat_means_matrix = pd.DataFrame(np.resize(feat_means, (lca_settle_all_feats.shape[0], 4000)), index = lca_settle_all_feats.index)

## demean MOSAIKS features
lca_settle_demean_feats = lca_settle_all_feats - feat_means_matrix

## save aggregated MOSAIKS features
lca_settle_all_feats.to_pickle(os.path.join(c.features_dir, 'lca_settle_mosaiks_features.pkl'))
lca_settle_demean_feats.to_pickle(os.path.join(c.features_dir, 'lca_settle_mosaiks_features_demeaned.pkl'))

