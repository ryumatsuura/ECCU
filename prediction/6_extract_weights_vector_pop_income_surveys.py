## This script trains the model based on surveys data

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

#########################################
## A) train model for population/income
#########################################

for country in ['brb', 'lca']:
    
    ## define unit
    if country == 'brb':
        unit = 'ed'
    elif country == 'lca':
        unit = 'settle'
    
    for task in ['population', 'income']:
        
        Y = pd.read_pickle(os.path.join(c.data_dir, 'int', task, '{}_{}_{}.pkl'.format(country, unit, task)))
        
        ## load MOSAIKS and NL data
        if task == 'population':
            mosaiks_only = pd.read_pickle(os.path.join(c.features_dir, '{}_{}_mosaiks_features.pkl'.format(country, unit)))
            nl_only = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_{}_nl.pkl'.format(country, unit)))
        elif task == 'income':
            mosaiks_only = pd.read_pickle(os.path.join(c.features_dir, '{}_{}_mosaiks_features_demeaned.pkl'.format(country, unit)))
            nl_only = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_{}_nl_demeaned.pkl'.format(country, unit)))
        
        ## merge MOSAIKS and NL
        both = pd.merge(mosaiks_only, nl_only, left_index = True, right_index = True)
        
        for df in (mosaiks_only, nl_only, both):
            
            ## obtain name of features
            name = next(x for x in globals() if globals()[x] is df)
            
            ## obtain common indices
            indices = pd.merge(Y, df, left_index = True, right_index = True).index
            
            ## convert data to numpy array
            if task == 'population':
                Y_np = np.array(Y.loc[indices, ['ln_pop_density']])
            elif task == 'income':
                Y_np = np.array(Y.loc[indices, ['income']])
            X_np = df[df.index.isin(indices)].to_numpy()
            
            ## set the bounds
            mins = Y_np.min(axis = 0)
            maxs = Y_np.max(axis = 0)
            solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
            
            ## split the data into training vs testing sets
            X_train, X_test, Y_train, Y_test, idxs_train, idsx_test = parse.split_data_train_test(
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
            np.savetxt(os.path.join(c.data_dir, 'int', 'weights', '{}_{}_{}_{}.csv'.format(country, unit, name, task)), wts, delimiter = ',')
            
            ## store performance metrics
            globals()[f'in_mse_{name}'] = holdout_results['metrics_train'][0][0][0]['mse']
            globals()[f'in_r2_{name}'] = holdout_results['metrics_train'][0][0][0]['r2_score']
            globals()[f'out_mse_{name}'] = holdout_results['metrics_test'][0][0][0]['mse']
            globals()[f'out_r2_{name}'] = holdout_results['metrics_test'][0][0][0]['r2_score']
            
            del Y_np, X_np, indices
        
        for sample in ['in', 'out']:
            
            ## store mse and r-square into one csv file
            rows = [
                {'Metrics': 'Survey-based',
                 'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only'], 
                 'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only'],
                 'NL: MSE': globals()[f'{sample}_mse_nl_only'], 
                 'NL: R-square': globals()[f'{sample}_r2_nl_only'],
                 'Both: MSE': globals()[f'{sample}_mse_both'], 
                 'Both: R-square': globals()[f'{sample}_r2_both']}
            ]
            
            fn = os.path.join(c.out_dir, 'metrics', '{}_{}_{}_{}sample_metrics.csv'.format(country, unit, task, sample))
            with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
                writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MOSAIKS: MSE', 'MOSAIKS: R-square', 'NL: MSE', 'NL: R-square', 'Both: MSE', 'Both: R-square'])
                writer.writeheader()
                writer.writerows(rows)

