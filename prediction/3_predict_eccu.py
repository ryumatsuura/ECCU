## This script implements the prediction exercises

## set preambles
subset_n = slice(None)
subset_feat = slice(None)

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

lambdas = lambdas_single = c.ml_model['global_lambdas']
solver = solve.ridge_regression
solver_kwargs = {'return_preds':True, 'svd_solve':False}

## extract bounds for density
c_app = getattr(c, 'population')
clip_bounds = c_app['world_bounds_pred']
lb = clip_bounds[0]
ub = clip_bounds[1]

###############
## A) predict
###############

## A-1. extract weights vectors

wts_global = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_population.csv'), delimiter = ',')
wts_cont = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_population.csv'), delimiter = ',')
wts_cont_fixed = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_fixed_population.csv'), delimiter = ',')
wts_brb = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_population.csv'), delimiter = ',')

## A-2. download MOSAIKS features

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

## add sample columns - Grenada and Trinidad and Tobago are in sample 3 
for df in (eccu_feat, eccu_subnat_feat):
    df['sample'] = 0
    df.loc[df['Country'] == 'GRD', ['sample']] = 3 
    df.loc[df['Country'] == 'TTO', ['sample']] = 3 
    df['sample_fixed'] = 3

## A-3. prediction!

## loop over national level and subnational level predictions
for df in (eccu_feat, eccu_subnat_feat):
    
    ## predict using global-scale weights vector
    if any(df.equals(y) for y in [eccu_feat]):
        ypreds = np.dot(df.iloc[:, 1:4001], wts_global)
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        ypreds = np.dot(df.iloc[:, 2:4002], wts_global)
    
    ## bound the prediction
    ypreds[ypreds < lb] = lb
    ypreds[ypreds > ub] = ub
    
    ## store predicted values
    if any(df.equals(y) for y in [eccu_feat]):
        eccu_preds = df[['Country']]
        eccu_preds['y_preds_global'] = ypreds.tolist()
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        eccu_subnat_preds = df[['Country']]
        eccu_subnat_preds['Name'] = df[['NAME_1']]
        eccu_subnat_preds['y_preds_global'] = ypreds.tolist()
    
    ## predict using continent-based weights vector
    for i in range(df.shape[0]):
        
        ## extract weight for each sample 
        mywts = wts_cont[df.loc[i, ['sample']].values[0]]
        
        ## predict
        if any(df.equals(y) for y in [eccu_feat]):
            ypreds[i] = np.dot(df.iloc[i, 1:4001], mywts)
        elif any(df.equals(y) for y in [eccu_subnat_feat]):
            ypreds[i] = np.dot(df.iloc[i, 2:4002], mywts)
        
        ## bound the prediction
        if ypreds[i] < lb:
            ypreds[i] = lb
        if ypreds[i] > ub:
            ypreds[i] = ub
    
    if any(df.equals(y) for y in [eccu_feat]):
        eccu_preds['y_preds_cont'] = ypreds.tolist()
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        eccu_subnat_preds['y_preds_cont'] = ypreds.tolist()
    
    ## predict using fixed continent-based weights vector
    for i in range(df.shape[0]):
        
        ## extract weight for each fixed sample 
        mywts = wts_cont_fixed[df.loc[i, ['sample_fixed']].values[0]]
        
        ## predict
        if any(df.equals(y) for y in [eccu_feat]):
            ypreds[i] = np.dot(df.iloc[i, 1:4001], mywts)
        elif any(df.equals(y) for y in [eccu_subnat_feat]):
            ypreds[i] = np.dot(df.iloc[i, 2:4002], mywts)
        
        ## bound the prediction
        if ypreds[i] < lb:
            ypreds[i] = lb
        if ypreds[i] > ub:
            ypreds[i] = ub
    
    if any(df.equals(y) for y in [eccu_feat]):
        eccu_preds['y_preds_cont_fixed'] = ypreds.tolist()
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        eccu_subnat_preds['y_preds_cont_fixed'] = ypreds.tolist()
    
    ## predict using BRB weights vector
    if any(df.equals(y) for y in [eccu_feat]):
        ypreds = np.dot(df.iloc[:, 1:4001], wts_brb)
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        ypreds = np.dot(df.iloc[:, 2:4002], wts_brb)
    
    ypreds[ypreds < lb] = lb
    ypreds[ypreds > ub] = ub
    
    if any(df.equals(y) for y in [eccu_feat]):
        eccu_preds['y_preds_brb'] = ypreds.tolist()
    elif any(df.equals(y) for y in [eccu_subnat_feat]):
        eccu_subnat_preds['y_preds_brb'] = ypreds.tolist()

###############################
## B) clean ground truth data 
###############################

eccu_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_density_eccu.csv'), index_col = 0)
eccu_subnat_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_density_eccu_subnat.csv'), index_col = 0)

## merge two dataframes
merged = pd.merge(eccu_preds, eccu_pop)
merged_subnat = pd.merge(eccu_subnat_preds, eccu_subnat_pop)

## national level 

## loop over different weights
for x in ['global', 'cont', 'cont_fixed', 'brb']:
    
    ## plot prediction against 
    plt.clf()
    tot_min = np.min([np.min(np.array(merged['y_preds_{}'.format(x)])), np.min(np.array(merged['Population.V1']))])
    tot_max = np.max([np.max(np.array(merged['y_preds_{}'.format(x)])), np.max(np.array(merged['Population.V1']))])
    fig, ax = plt.subplots()
    ax.scatter(np.array(merged['Population.V1']), np.array(merged['y_preds_{}'.format(x)]))
    
    ## add 45 degree line and country names
    plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
    for i, txt in enumerate(np.array(merged['Country'])):
        ax.annotate(txt, (np.array(merged['Population.V1'])[i], np.array(merged['y_preds_{}'.format(x)])[i]))
    
    ## add axis title
    ax.set_xlabel('True Population Density')
    ax.set_ylabel('Predicted Population Density')
    
    ## output the graph
    fig.savefig(os.path.join(c.out_dir, 'population', 'eccu_pop_density_{}.png'.format(x)), bbox_inches = 'tight', pad_inches = 0.1)

## subnational level

c_labels, c_indices = np.unique(merged_subnat['Country'], return_inverse = True)

## loop over different weights
for x in ['global', 'cont', 'cont_fixed', 'brb']:
    
    ## output the results 
    plt.clf()
    tot_min = np.min([np.min(np.array(merged_subnat['y_preds_{}'.format(x)])), np.min(np.array(merged_subnat['Population.V1']))])
    tot_max = np.max([np.max(np.array(merged_subnat['y_preds_{}'.format(x)])), np.max(np.array(merged_subnat['Population.V1']))])
    fig, ax = plt.subplots()
    sc = ax.scatter(np.array(merged_subnat['Population.V1']), np.array(merged_subnat['y_preds_{}'.format(x)]), c = c_indices)
    ax.legend(sc.legend_elements()[0], c_labels)
    
    ## add 45 degree line
    plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
    
    ## add axis title
    ax.set_xlabel('True Population Density')
    ax.set_ylabel('Predicted Population Density')
    
    ## output the graph
    fig.savefig(os.path.join(c.out_dir, 'population', 'eccu_subnat_pop_density_{}.png'.format(x)), bbox_inches = 'tight', pad_inches = 0.1)

