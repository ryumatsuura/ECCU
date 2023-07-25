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

 ## set index
pop = pop.set_index('iso_code')
subnat_pop = subnat_pop.set_index('GDLCODE')

## obtain common indices 
indices = pd.merge(pop, mosaiks_feat, left_index = True, right_index = True).index
subnat_indices = pd.merge(subnat_pop, mosaiks_subnat_feat, left_index = True, right_index = True).index

###################
## B) train model 
###################

for l in ['nat', 'subnat']:
    
    ## merge x and y
    if l == 'nat':
        Y, X, latlons, ids = parse.merge(pop.loc[indices, ['Population']], mosaiks_feat[mosaiks_feat.index.isin(indices)], pop.loc[indices, ['X', 'Y']].rename(columns = {'Y':'lat', 'X':'lon'}), pd.Series(pop.index, index = pop.index))
    elif l == 'subnat':
        Y, X, latlons, ids = parse.merge(subnat_pop.loc[subnat_indices, ['Population']], mosaiks_subnat_feat[mosaiks_subnat_feat.index.isin(subnat_indices)], subnat_pop.loc[subnat_indices, ['X', 'Y']].rename(columns = {'Y':'lat', 'X':'lon'}), pd.Series(subnat_pop.index, index = subnat_pop.index))
    
    ## set the bounds
    mins = Y.min(axis = 0)
    maxs = Y.max(axis = 0)
    solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
    
    ## split the data into training vs testing sets
    X_train, X_test, Y_train, Y_test, idxs_train, idxs_test = parse.split_data_train_test(
        X, Y, frac_test = c.ml_model['test_set_frac'], return_idxs = True
    )
    latlons_train = latlons[idxs_train]
    latlons_test = latlons[idxs_test]
    
    ## define limit to subsets
    Y_train = Y_train[subset_n]
    X_train = X_train[subset_n, subset_feat]
    latlons_train = latlons_train[subset_n]
    
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
    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}_population.csv'.format(l)), wts, delimiter = ',')
    
    ## store performance metrics
    if l == 'nat':
        mse_nat = holdout_results['metrics_test'][0][0][0]['mse']
        r2_nat = holdout_results['metrics_test'][0][0][0]['r2_score']
    elif l == 'subnat':
        mse_subnat = holdout_results['metrics_test'][0][0][0]['mse']
        r2_subnat = holdout_results['metrics_test'][0][0][0]['r2_score']

## store mse and r-square into one csv file
rows = [
    {'Metrics': 'National-level',
     'MSE': mse_nat,
     'R-square': r2_nat},
    {'Metrics': 'Subnational-level',
     'MSE': mse_subnat,
     'R-square': r2_subnat}
]

fn = os.path.join(c.out_dir, 'metrics', 'eccu_population_adm_insample_metrics.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MSE', 'R-square'])
    writer.writeheader()
    writer.writerows(rows)

