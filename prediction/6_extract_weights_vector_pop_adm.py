## This script trains the model at national and subnational scales
## and store the weights vectors

## set preambles
subset_n = slice(None)
subset_feat = slice(None)

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from mosaiks import transforms
from mosaiks.utils.imports import *

lambdas = lambdas_single = np.hstack([1, (np.logspace(-3, 6, 8) ** .3)])
solver = solve.ridge_regression
solver_kwargs = {'return_preds': True, 'svd_solve': False}

##################################
## A) compute population density
##################################

## load MOSAIKS feature aggregated to national/subnational level
mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM0_polygon_X_creation_pop_weight=True.p'))
subnat_mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM1_polygon_X_creation_pop_weight=True.p')).drop(columns = 'GDLCODE')

## load HDI measures
hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_adm0_clean.p')).loc[mosaiks_feat.index]
subnat_hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_clean.p')).loc[subnat_mosaiks_feat.index]

## load subnational shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'GDL Shapefiles V4', 'GDL Shapefiles V4.shp'))
shp = shp.set_index('GDLcode')

## compute area
shp['area_sq_km'] = shp.to_crs({'init': 'epsg:6933'})['geometry'].area / 1e6

## dissolve to the national level and compute area again
nat_shp = shp[['iso_code', 'geometry']].dissolve(['iso_code'])
nat_shp['area_sq_km'] = nat_shp.to_crs({'init': 'epsg:6933'})['geometry'].area / 1e6

## merge into area and compute population density
pop = hdi[['Population size in millions']].merge(nat_shp['area_sq_km'], left_index = True, right_index = True)
pop['pop_density'] = pop['Population size in millions'] * 1000000 / pop['area_sq_km']
pop['ln_pop_density'] = np.log(pop['pop_density'] + 1)
subnat_pop = subnat_hdi[['Population size in millions']].merge(shp['area_sq_km'], left_index = True, right_index = True)
subnat_pop['pop_density'] = subnat_pop['Population size in millions'] * 1000000 / subnat_pop['area_sq_km']
subnat_pop['ln_pop_density'] = np.log(subnat_pop['pop_density'] + 1)

## save population data for future use
pop.to_pickle(os.path.join(c.data_dir, 'int', 'population', 'global_nat_population.pkl'))
subnat_pop.to_pickle(os.path.join(c.data_dir, 'int', 'population', 'global_subnat_population.pkl'))

## load nighttime light data
nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'dmsp_nightlight_features_for_adm0_polygons_20_bins_GPW_pop_weighted.p')).loc[mosaiks_feat.index]
subnat_nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'dmsp_nightlight_features_for_hdi_polygons_20_bins_GPW_pop_weighted.p')).loc[subnat_mosaiks_feat.index]

## combine MOSAIKS and nightlight 
new_X = pd.merge(mosaiks_feat, nl, left_index = True, right_index = True)
subnat_new_X = pd.merge(subnat_mosaiks_feat, subnat_nl, left_index = True, right_index = True)

###################
## B) train model 
###################

for x in ['mosaiks_only', 'nl_only', 'both']:
    for l in ['nat', 'subnat']:
        
        ## obtain common indices
        if l == 'nat':
            if x == 'mosaiks_only':
                indices = pd.merge(pop, mosaiks_feat, left_index = True, right_index = True).index
            elif x == 'nl_only':
                indices = pd.merge(pop, nl, left_index = True, right_index = True).index
            elif x == 'both':
                indices = pd.merge(pop, new_X, left_index = True, right_index = True).index
        elif l == 'subnat':
            if x == 'mosaiks_only':
                indices = pd.merge(subnat_pop, subnat_mosaiks_feat, left_index = True, right_index = True).index
            elif x == 'nl_only':
                indices = pd.merge(subnat_pop, subnat_nl, left_index = True, right_index = True).index
            elif x == 'both':
                indices = pd.merge(subnat_pop, subnat_new_X, left_index = True, right_index = True).index
        
        ## restructure the dataset for training codes - needs to be numpy arrays
        if l == 'nat':
            Y_np = np.array(pop.loc[indices, ['ln_pop_density']])
            if x == 'mosaiks_only':
                X_np = mosaiks_feat[mosaiks_feat.index.isin(indices)].to_numpy()
            elif x == 'nl_only':
                X_np = nl[nl.index.isin(indices)].to_numpy()
            elif x == 'both':
                X_np = new_X[new_X.index.isin(indices)].to_numpy()
        elif l == 'subnat':
            Y_np = np.array(subnat_pop.loc[indices, ['ln_pop_density']])
            if x == 'mosaiks_only':
                X_np = subnat_mosaiks_feat[subnat_mosaiks_feat.index.isin(indices)].to_numpy()
            elif x == 'nl_only':
                X_np = subnat_nl[subnat_nl.index.isin(indices)].to_numpy()
            elif x == 'both':
                X_np = subnat_new_X[subnat_new_X.index.isin(indices)].to_numpy()
        
        ## set the bounds
        mins = Y_np.min(axis = 0)
        maxs = Y_np.max(axis = 0)
        solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
        
        ## split the data into training vs testing sets
        X_train, X_test, Y_train, Y_test, idxs_train, idxs_test = parse.split_data_train_test(
            X_np, Y_np, frac_test = 0.2, return_idxs = True
        )
        
        ## define limit to subsets
        Y_train = Y_train[subset_n]
        X_train = X_train[subset_n, subset_feat]
            
        kfold_results = solve.kfold_solve(
            X_train, Y_train, solve_function = solver, num_folds = 5, 
            return_model = True, lambdas = lambdas_single, **solver_kwargs
        )
        
        ## get best predictions from model
        best_lambda_idx, best_metrics, best_preds = ir.interpret_kfold_results(
            kfold_results, 'r2_score', hps = [('lambdas', lambdas_single)]
        )
        
        ## set best lambda
        best_lambda = np.array([lambdas_single[best_lambda_idx[0]]])
        
        ## retrain the model using the best lambda
        holdout_results = solve.single_solve(
            X_train[subset_n, subset_feat], X_test[:, subset_feat], Y_train[subset_n], Y_test,
            lambdas = best_lambda, return_preds = True, return_model = True, clip_bounds = [np.array([mins, maxs])]
        )
        
        wts = holdout_results['models'][0][0][0]
        np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}_{}_population.csv'.format(l, x)), wts, delimiter = ',')
        
        ## store performance metrics
        globals()[f'in_mse_{x}_{l}'] = holdout_results['metrics_train'][0][0][0]['mse']
        globals()[f'in_r2_{x}_{l}'] = holdout_results['metrics_train'][0][0][0]['r2_score']
        globals()[f'out_mse_{x}_{l}'] = holdout_results['metrics_test'][0][0][0]['mse']
        globals()[f'out_r2_{x}_{l}'] = holdout_results['metrics_test'][0][0][0]['r2_score']
        
        del Y_np, X_np, indices

for sample in ['in', 'out']:
    
    ## store mse and r-square into one csv file
    rows = [
        {'Metrics': 'National-level',
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_nat'],
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_nat'],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_nat'],
         'NL: R-square': globals()[f'{sample}_r2_nl_only_nat'],
         'Both: MSE': globals()[f'{sample}_mse_both_nat'],
         'Both: R-square': globals()[f'{sample}_r2_both_nat']},
        {'Metrics': 'Subnational-level',
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_subnat'],
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_subnat'],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_subnat'],
         'NL: R-square': globals()[f'{sample}_r2_nl_only_subnat'],
         'Both: MSE': globals()[f'{sample}_mse_both_subnat'],
         'Both: R-square': globals()[f'{sample}_r2_both_subnat']}
    ]
    
    fn = os.path.join(c.out_dir, 'metrics', 'global_adm_population_{}sample_metrics.csv'.format(sample))
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MOSAIKS: MSE', 'MOSAIKS: R-square', 'NL: MSE', 'NL: R-square', 'Both: MSE', 'Both: R-square'])
        writer.writeheader()
        writer.writerows(rows)

