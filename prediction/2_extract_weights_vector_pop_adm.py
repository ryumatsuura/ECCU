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

lambdas = lambdas_single = c.ml_model['global_lambdas']
solver = solve.ridge_regression
solver_kwargs = {'return_preds': True, 'svd_solve': False}

#######################
## A) load label data
#######################

## load MOSAIKS feature aggregated to national/subnational level
mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM0_polygon_X_creation_pop_weight=True.p'))
mosaiks_subnat_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM1_polygon_X_creation_pop_weight=True.p')).drop(columns = 'GDLCODE')

## load population data
pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_global.csv'), index_col = 0)
subnat_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_global_subnat.csv'), index_col = 0)

## load nighttime light data
nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'dmsp_nightlight_features_for_adm0_polygons_20_bins_GPW_pop_weighted.p'))
subnat_nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'dmsp_nightlight_features_for_hdi_polygons_20_bins_GPW_pop_weighted.p'))

## set index
pop = pop.set_index('iso_code')
subnat_pop = subnat_pop.set_index('GDLCODE')

###################
## B) train model 
###################

for x in ['mosaiks_only', 'nl_only', 'both']:
    for l in ['nat', 'subnat']:
        
        ## combine mosaiks and nightlight 
        if x == 'both':
            if l == 'nat':
                new_X = pd.merge(mosaiks_feat, nl, left_index = True, right_index = True)
            elif l == 'subnat':
                new_X = pd.merge(mosaiks_subnat_feat, subnat_nl, left_index = True, right_index = True)
        
        ## obtain common indices
        if x == 'mosaiks_only':
            if l == 'nat':
                indices = pd.merge(pop, mosaiks_feat, left_index = True, right_index = True).index
            elif l == 'subnat':
                indices = pd.merge(subnat_pop, mosaiks_subnat_feat, left_index = True, right_index = True).index
        elif x == 'nl_only':
            if l == 'nat':
                indices = pd.merge(pop, nl, left_index = True, right_index = True).index
            elif l == 'subnat':
                indices = pd.merge(subnat_pop, subnat_nl, left_index = True, right_index = True).index
        elif x == 'both':
            if l == 'nat':
                indices = pd.merge(pop, new_X, left_index = True, right_index = True).index
            elif l == 'subnat':
                indices = pd.merge(subnat_pop, new_X, left_index = True, right_index = True).index
        
        ## restructure the dataset for training codes - needs to be numpy arrays
        if l == 'nat':
            Y_np = np.array(pop.loc[indices, ['Population']])
            if x == 'mosaiks_only':
                X_np = mosaiks_feat[mosaiks_feat.index.isin(indices)].to_numpy()
            elif x == 'nl_only':
                X_np = nl[nl.index.isin(indices)].to_numpy()
        elif l == 'subnat':
            Y_np = np.array(subnat_pop.loc[indices, ['Population']])
            if x == 'mosaiks_only':
                X_np = mosaiks_subnat_feat[mosaiks_subnat_feat.index.isin(indices)].to_numpy()
            elif x == 'nl_only':
                X_np = subnat_nl[subnat_nl.index.isin(indices)].to_numpy()
        if x == 'both':
            X_np = new_X[new_X.index.isin(indices)].to_numpy()
        
        ## set the bounds
        mins = Y_np.min(axis = 0)
        maxs = Y_np.max(axis = 0)
        solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
        
        ## split the data into training vs testing sets
        X_train, X_test, Y_train, Y_test, idxs_train, idxs_test = parse.split_data_train_test(
            X_np, Y_np, frac_test = c.ml_model['test_set_frac'], return_idxs = True
        )
        
        ## define limit to subsets
        Y_train = Y_train[subset_n]
        X_train = X_train[subset_n, subset_feat]
            
        kfold_results = solve.kfold_solve(
            X_train, Y_train, solve_function = solver, num_folds = c.ml_model['n_folds'], 
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
        globals()[f'mse_{x}_{l}'] = holdout_results['metrics_test'][0][0][0]['mse']
        globals()[f'r2_{x}_{l}'] = holdout_results['metrics_test'][0][0][0]['r2_score']
        
        del Y_np, X_np, indices
        if x == 'both':
            del new_X

## store mse and r-square into one csv file
rows = [
    {'Metrics': 'National-level',
     'MOSAIKS: MSE': mse_mosaiks_only_nat,
     'MOSAIKS: R-square': r2_mosaiks_only_nat,
     'NL: MSE': mse_nl_only_nat,
     'NL: R-square': r2_nl_only_nat,
     'Both: MSE': mse_both_nat,
     'Both: R-square': r2_both_nat},
    {'Metrics': 'Subnational-level',
     'MOSAIKS: MSE': mse_mosaiks_only_subnat,
     'MOSAIKS: R-square': r2_mosaiks_only_subnat,
     'NL: MSE': mse_nl_only_subnat,
     'NL: R-square': r2_nl_only_subnat,
     'Both: MSE': mse_both_subnat,
     'Both: R-square': r2_both_subnat}
]

fn = os.path.join(c.out_dir, 'metrics', 'eccu_population_adm_insample_metrics.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MOSAIKS: MSE', 'MOSAIKS: R-square', 'NL: MSE', 'NL: R-square', 'Both: MSE', 'Both: R-square'])
    writer.writeheader()
    writer.writerows(rows)

