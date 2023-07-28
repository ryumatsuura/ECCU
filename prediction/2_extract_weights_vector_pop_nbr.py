## This script trains the model based on neighboring countries

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

############################################
## A) train model in neighboring countries
############################################

## A-1. Train by country

for country in ['brb', 'glp', 'mtq']:
    
    ## load Y and X data for Barbados/Guadeloupe/Martinique
    pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_{}.csv'.format(country.upper())), index_col = 0)
    assert pop.shape[0] == np.where(pop['population'].notnull())[0].shape[0]
    mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_{}.csv'.format(country)))
    nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', '{}_nl_features_pop_weighted.pkl'.format(country)))
    
    ## create index based on lat/lon
    pop = pop.set_index(pop['lat'].astype(str) + ':' + pop['lon'].astype(str)).drop(columns = ['lat', 'lon'])
    mosaiks_feat = mosaiks_feat.set_index(mosaiks_feat['Lat'].astype(str) + ':' + mosaiks_feat['Lon'].astype(str)).drop(columns = ['Lat', 'Lon', 'BoxLabel'])
    nl = nl.set_index(nl['lat'].astype(str) + ':' + nl['lon'].astype(str)).drop(columns = ['lat', 'lon'])
    
    ## merge mosaiks and nighttime light
    new_X = pd.merge(mosaiks_feat, nl, left_index = True, right_index = True)
        
    ## population and 
    for x in ['mosaiks_only', 'nl_only', 'both']:
        
        ## obtain common indices
        if x == 'mosaiks_only':
            indices = pd.merge(pop, mosaiks_feat, left_index = True, right_index = True).index
        elif x == 'nl_only':
            indices = pd.merge(pop, nl, left_index = True, right_index = True).index
        elif x == 'both':
            indices = pd.merge(pop, new_X, left_index = True, right_index = True).index
        
        ## restructure the dataset for training codes - needs to be numpy arrays
        pop_np = np.array(pop.loc[indices, ['population']])
        if x == 'mosaiks_only':
            X_np = mosaiks_feat[mosaiks_feat.index.isin(indices)].to_numpy()
        elif x == 'nl_only':
            X_np = nl[nl.index.isin(indices)].to_numpy()
        elif x == 'both':
            X_np = new_X[new_X.index.isin(indices)].to_numpy()
        
        ## set the boundary
        mins = pop_np.min(axis = 0)
        maxs = pop_np.max(axis = 0)
        solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
        
        X_train, X_test, Y_train, Y_test, idxs_train, idxs_test = parse.split_data_train_test(
            X_np, pop_np, frac_test = c.ml_model['test_set_frac'], return_idxs = True
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
        
        wts_ = holdout_results['models'][0][0][0]
        np.savetxt(os.path.join(c.data_dir, 'int', 'weights', '{}_{}_population.csv'.format(country, x)), wts_, delimiter = ',')
        
        ## store performance metrics
        globals()[f'mse_{x}_{country}'] = holdout_results['metrics_test'][0][0][0]['mse']
        globals()[f'r2_{x}_{country}'] = holdout_results['metrics_test'][0][0][0]['r2_score']

## A-2. Train for all neighboring countries altogether

## load Y and X data for Barbados/Guadeloupe/Martinique
Y_brb = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_BRB.csv'), index_col = 0)
X_brb = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_brb.csv'))
Y_glp = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_GLP.csv'), index_col = 0)
X_glp = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_glp.csv'))
Y_mtq = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_MTQ.csv'), index_col = 0)
X_mtq = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_mtq.csv'))

## combine all dataframes
Y_comb = pd.concat([Y_brb, Y_glp, Y_mtq], ignore_index = True)
X_comb = pd.concat([X_brb, X_glp, X_mtq], ignore_index = True)
Y_comb_pop = pd.merge(Y_comb, X_comb[['Lat', 'Lon']], left_on = ['lon', 'lat'], right_on = ['Lon', 'Lat'])[['population']]
X_comb_pop = pd.merge(X_comb, Y_comb[['lat', 'lon']], left_on = ['Lon', 'Lat'], right_on = ['lon', 'lat']).drop(columns = ['lon', 'lat'])

## remove nan from population
indices = np.where(Y_comb_pop['population'].notnull())[0]
Y_comb_pop = Y_comb_pop.reindex(indices)
X_comb_pop = X_comb_pop.reindex(indices)
Y_comb_pop, X_comb_pop, latlons_comb, ids_comb = parse.merge(Y_comb_pop, X_comb_pop.iloc[:, 3:X_comb_pop.shape[1]], X_comb_pop[['Lat', 'Lon']].rename(columns = {'Lat':'lat', 'Lon':'lon'}), pd.Series(Y_comb_pop.index, index = Y_comb_pop.index))

X_comb_train, X_comb_test, Y_comb_train, Y_comb_test, idxs_comb_train, idxs_comb_test = parse.split_data_train_test(
    X_comb_pop, Y_comb_pop, frac_test = c.ml_model['test_set_frac'], return_idxs = True
)
latlons_comb_train = latlons_comb[idxs_comb_train]
latlons_comb_test = latlons_comb[idxs_comb_test]

## define limit to subsets
Y_comb_train = Y_comb_train[subset_n]
X_comb_train = X_comb_train[subset_n, subset_feat]
latlons_comb_train = latlons_comb_train[subset_n]

kfold_results = solve.kfold_solve(
    X_comb_train, Y_comb_train, solve_function = solver, num_folds = c.ml_model['n_folds'], 
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
    X_comb_train[subset_n, subset_feat], X_comb_test[:, subset_feat], Y_comb_train[subset_n], Y_comb_test,
    lambdas = best_lambda, return_preds = True, return_model = True, clip_bounds = [np.array([mins, maxs])]
)

wts_comb = holdout_results['models'][0][0][0]
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'nbr_population.csv'), wts_comb, delimiter = ',')

## store performance metrics
mse_comb = holdout_results['metrics_test'][0][0][0]['mse']
r2_comb = holdout_results['metrics_test'][0][0][0]['r2_score']

## store mse and r-square into one csv file - by-continent = 0 (except for TTO and GRD) 
rows = [
    {'Metrics': 'Barbados-based', 'MSE': mse_brb, 'R-square': r2_brb},
    {'Metrics': 'Guadeloupe-based', 'MSE': mse_glp, 'R-square': r2_glp},
    {'Metrics': 'Martinique-based', 'MSE': mse_mtq, 'R-square': r2_mtq},
    {'Metrics': 'Neighbors-based', 'MSE': mse_comb, 'R-square': r2_comb}
]

fn = os.path.join(c.out_dir, 'metrics', 'eccu_population_nbr_insample_metrics.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MSE', 'R-square'])
    writer.writeheader()
    writer.writerows(rows)

