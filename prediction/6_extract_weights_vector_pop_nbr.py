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

lambdas = lambdas_single = np.hstack([1, (np.logspace(-3, 6, 8) ** .3)])
solver = solve.ridge_regression
solver_kwargs = {'return_preds': True, 'svd_solve': False}

## create folders if not exist
if not os.path.exists(os.path.join(c.data_dir, 'int', 'weights')):
    os.makedirs(os.path.join(c.data_dir, 'int', 'weights'))

if not os.path.exists(os.path.join(c.out_dir, 'metrics')):
    os.makedirs(os.path.join(c.out_dir, 'metrics'))

############################################
## A) train model in neighboring countries
############################################

## A-1. Train by country

for country in ['brb', 'glp', 'mtq']:
    
    ## load population data for Barbados/Guadeloupe/Martinique
    pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', '{}_mosaiks_population.pkl'.format(country)))
    
    ## load MOSAIKS and nl data
    mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_{}.csv'.format(country)))
    nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_mosaiks_nl.pkl'.format(country)))
    
    ## create index based on lat/lon
    mosaiks_feat = mosaiks_feat.set_index(mosaiks_feat['Lat'].astype(str) + ':' + mosaiks_feat['Lon'].astype(str)).drop(columns = ['Lat', 'Lon', 'BoxLabel'])
    
    ## merge mosaiks and nighttime light
    new_X = pd.merge(mosaiks_feat, nl, left_index = True, right_index = True)
    
    ## test different RHS 
    for x in ['mosaiks_only', 'nl_only', 'both']:
        
        ## obtain common indices
        if x == 'mosaiks_only':
            indices = pd.merge(pop, mosaiks_feat, left_index = True, right_index = True).index
        elif x == 'nl_only':
            indices = pd.merge(pop, nl, left_index = True, right_index = True).index
        elif x == 'both':
            indices = pd.merge(pop, new_X, left_index = True, right_index = True).index
        
        ## restructure the dataset for training codes - needs to be numpy arrays
        pop_np = np.array(pop.loc[indices, ['ln_pop_density']])
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
            X_np, pop_np, frac_test = 0.2, return_idxs = True
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
        
        wts_ = holdout_results['models'][0][0][0]
        np.savetxt(os.path.join(c.data_dir, 'int', 'weights', '{}_{}_population.csv'.format(country, x)), wts_, delimiter = ',')
        
        ## store performance metrics
        globals()[f'in_mse_{x}_{country}'] = holdout_results['metrics_train'][0][0][0]['mse']
        globals()[f'in_r2_{x}_{country}'] = holdout_results['metrics_train'][0][0][0]['r2_score']
        globals()[f'out_mse_{x}_{country}'] = holdout_results['metrics_test'][0][0][0]['mse']
        globals()[f'out_r2_{x}_{country}'] = holdout_results['metrics_test'][0][0][0]['r2_score']
        
        del pop_np, X_np, indices

## A-2. Train for all neighboring countries altogether

## load population data in neighboring countries
pop_brb = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'brb_mosaiks_population.pkl'))
pop_glp = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'glp_mosaiks_population.pkl'))
pop_mtq = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'mtq_mosaiks_population.pkl'))

## load MOSAIKS and nighttime light data
mosaiks_brb = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_brb.csv'))
nl_brb = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'brb_mosaiks_nl.pkl'))
mosaiks_glp = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_glp.csv'))
nl_glp = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'glp_mosaiks_nl.pkl'))
mosaiks_mtq = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_mtq.csv'))
nl_mtq = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'mtq_mosaiks_nl.pkl'))

## create index based on lat/lon
mosaiks_brb = mosaiks_brb.set_index(mosaiks_brb['Lat'].astype(str) + ':' + mosaiks_brb['Lon'].astype(str)).drop(columns = ['Lat', 'Lon', 'BoxLabel'])
mosaiks_glp = mosaiks_glp.set_index(mosaiks_glp['Lat'].astype(str) + ':' + mosaiks_glp['Lon'].astype(str)).drop(columns = ['Lat', 'Lon', 'BoxLabel'])
mosaiks_mtq = mosaiks_mtq.set_index(mosaiks_mtq['Lat'].astype(str) + ':' + mosaiks_mtq['Lon'].astype(str)).drop(columns = ['Lat', 'Lon', 'BoxLabel'])

## combine all dataframes
pop_comb = pd.concat([pop_brb, pop_glp, pop_mtq])
mosaiks_comb = pd.concat([mosaiks_brb, mosaiks_glp, mosaiks_mtq])
nl_comb = pd.concat([nl_brb, nl_glp, nl_mtq])

## merge mosaiks and nighttime light
new_X_comb = pd.merge(mosaiks_comb, nl_comb, left_index = True, right_index = True)

## test different RHS 
for x in ['mosaiks_only', 'nl_only', 'both']:
    
    ## obtain common indices
    if x == 'mosaiks_only':
        indices = pd.merge(pop_comb, mosaiks_comb, left_index = True, right_index = True).index
    elif x == 'nl_only':
        indices = pd.merge(pop_comb, nl_comb, left_index = True, right_index = True).index
    elif x == 'both':
        indices = pd.merge(pop_comb, new_X_comb, left_index = True, right_index = True).index
    
    ## restructure the dataset for training codes - needs to be numpy arrays
    pop_np = np.array(pop_comb.loc[indices, ['ln_pop_density']])
    if x == 'mosaiks_only':
        X_np = mosaiks_comb[mosaiks_comb.index.isin(indices)].to_numpy()
    elif x == 'nl_only':
        X_np = nl_comb[nl_comb.index.isin(indices)].to_numpy()
    elif x == 'both':
        X_np = new_X_comb[new_X_comb.index.isin(indices)].to_numpy()
    
    ## set the boundary
    mins = pop_np.min(axis = 0)
    maxs = pop_np.max(axis = 0)
    solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
    
    X_comb_train, X_comb_test, Y_comb_train, Y_comb_test, idxs_comb_train, idxs_comb_test = parse.split_data_train_test(
        X_np, pop_np, frac_test = 0.2, return_idxs = True
    )
    
    ## define limit to subsets
    Y_comb_train = Y_comb_train[subset_n]
    X_comb_train = X_comb_train[subset_n, subset_feat]
    
    kfold_results = solve.kfold_solve(
        X_comb_train, Y_comb_train, solve_function = solver, num_folds = 5, 
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
    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'nbr_{}_population.csv'.format(x)), wts_comb, delimiter = ',')
    
    ## store performance metrics
    globals()[f'in_mse_{x}_comb'] = holdout_results['metrics_train'][0][0][0]['mse']
    globals()[f'in_r2_{x}_comb'] = holdout_results['metrics_train'][0][0][0]['r2_score']
    globals()[f'out_mse_{x}_comb'] = holdout_results['metrics_test'][0][0][0]['mse']
    globals()[f'out_r2_{x}_comb'] = holdout_results['metrics_test'][0][0][0]['r2_score']
    
    del pop_np, X_np, indices

for sample in ['in', 'out']:
    
    ## store mse and r-square into one csv file
    rows = [
        {'Metrics': 'Barbados-based',
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_brb'], 
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_brb'],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_brb'], 
         'NL: R-square': globals()[f'{sample}_r2_nl_only_brb'],
         'Both: MSE': globals()[f'{sample}_mse_both_brb'], 
         'Both: R-square': globals()[f'{sample}_r2_both_brb']},
        {'Metrics': 'Guadeloupe-based', 
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_glp'], 
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_glp'],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_glp'], 
         'NL: R-square': globals()[f'{sample}_r2_nl_only_glp'],
         'Both: MSE': globals()[f'{sample}_mse_both_glp'], 
         'Both: R-square': globals()[f'{sample}_r2_both_glp']},     
        {'Metrics': 'Martinique-based', 
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_mtq'], 
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_mtq'],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_mtq'], 
         'NL: R-square': globals()[f'{sample}_r2_nl_only_mtq'],
         'Both: MSE': globals()[f'{sample}_mse_both_mtq'], 
         'Both: R-square': globals()[f'{sample}_r2_both_mtq']},          
        {'Metrics': 'Neighbors-based', 
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_comb'], 
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_comb'],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_comb'], 
         'NL: R-square': globals()[f'{sample}_r2_nl_only_comb'],
         'Both: MSE': globals()[f'{sample}_mse_both_comb'], 
         'Both: R-square': globals()[f'{sample}_r2_both_comb']}
    ]
    
    fn = os.path.join(c.out_dir, 'metrics', 'nbr_population_{}sample_metrics.csv'.format(sample))
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MOSAIKS: MSE', 'MOSAIKS: R-square', 'NL: MSE', 'NL: R-square', 'Both: MSE', 'Both: R-square'])
        writer.writeheader()
        writer.writerows(rows)

