## This script trains the model at global scales and 
## stores the weights vectors

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

#######################
## A) load label data
#######################

## load indexed MOSAIKS feature data 
mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'grid_features.pkl'))

## save lat/lon and reset index
latlons = mosaiks_feat[['lat', 'lon']].to_numpy()
mosaiks_feat = mosaiks_feat.set_index(mosaiks_feat['lat'].astype(str) + ':' + mosaiks_feat['lon'].astype(str)).drop(columns = ['lat', 'lon', 'geometry'])

## load population and nighttime light data
pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'global_population.pkl'))
nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'global_nl.pkl'))

## combine MOSAIKS and nightlight
new_X = pd.merge(mosaiks_feat, nl, left_index = True, right_index = True)

assert pop.shape[0] == mosaiks_feat.shape[0] == nl.shape[0] == new_X.shape[0]

## store indices
ids = pop.index.values

###################################
## B) train model at global scale
###################################

for x in ['mosaiks_only', 'nl_only', 'both']:
    
    ## restructure the dataset for training codes - needs to be numpy arrays
    pop_np = np.array(pop['ln_pop_density'])
    if x == 'mosaiks_only':
        X_np = mosaiks_feat.to_numpy()
    elif x == 'nl_only':
        X_np = nl.to_numpy()
    elif x == 'both':
        X_np = new_X.to_numpy()
    
    ## set the bounds
    mins = pop_np.min(axis = 0)
    maxs = pop_np.max(axis = 0)
    solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
    
    ## split the data into training vs testing sets
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
    
    wts_global = holdout_results['models'][0][0][0]
    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}_population.csv'.format(x)), wts_global, delimiter = ',')
    
    ## store performance metrics
    globals()[f'in_mse_{x}_global'] = holdout_results['metrics_train'][0][0][0]['mse']
    globals()[f'in_r2_{x}_global'] = holdout_results['metrics_train'][0][0][0]['r2_score']
    globals()[f'out_mse_{x}_global'] = holdout_results['metrics_test'][0][0][0]['mse']
    globals()[f'out_r2_{x}_global'] = holdout_results['metrics_test'][0][0][0]['r2_score']
    
    del pop_np, X_np

#####################################
## C) train model in each continent
#####################################

## split latlons into continents
latlonsdf_samp = parse.split_world_sample(pd.DataFrame(latlons, index = ids, columns = ['lat', 'lon']))

for x in ['mosaiks_only', 'nl_only', 'both']:
    
    ## select the dataset for training codes
    Y_df = pop[['ln_pop_density']]
    if x == 'mosaiks_only':
        X_df = mosaiks_feat
    elif x == 'nl_only':
        X_df = nl
    elif x == 'both':
        X_df = new_X
    
    ## save kfold results from all samples in one list
    N = 6
    kfold_results = [None] * N
    ll_train = [None] * N
    best_preds = [None] * N
    mykwargs = [None] * N
    best_lambda_idxs = [None] * N
    idxs_train = [None] * N
    idxs_test = [None] * N
    globalclipping = True
    
    for s in range(N):
        ids_samp = np.where(latlonsdf_samp['samp'] == s)
        if len(ids_samp[0]) >= 500:
            (
                kfold_results[s], ll_train[s], idxs_train[s], idxs_test[s], mykwargs[s],
            ) = solve.split_world_sample_solve(
                X_df, Y_df, latlonsdf_samp, sample = s, subset_n = subset_n, subset_feat = subset_feat,
                num_folds = 5, solve_function = solver, globalclipping = globalclipping,
                lambdas = lambdas, **solver_kwargs
            )
            best_lambda_idxs[s], _, best_preds[s] = ir.interpret_kfold_results(
                kfold_results[s], 'r2_score', hps = [('lambdas', lambdas)],
            )
        else:
            print('----Cannot estimate ridge regression with cross-val: Too few observations!----')
            
    ## remove none-type entries from list
    kfold_results_clean = [res for res in kfold_results if res != None]
    best_preds_clean = [res for res in best_preds if res != None]
    
    myN = len(best_preds_clean)
    
    ## initialize weights vector
    wts_cont = [None] * myN
    globals()[f'in_mse_{x}_cont'] = [None] * myN
    globals()[f'in_r2_{x}_cont'] = [None] * myN
    globals()[f'out_mse_{x}_cont'] = [None] * myN
    globals()[f'out_r2_{x}_cont'] = [None] * myN
    
    for sample in range(myN):
        
        ## initialize the weights vector for each continent
        wts_this_cont = [None]
        
        ## cut sample for each continent
        ids_samp = np.where(latlonsdf_samp['samp'] == sample)
        X_samp = X_df.iloc[ids_samp]
        Y_samp = Y_df.iloc[ids_samp]
        
        ## split into train vs test sets
        X_train = X_samp.iloc[idxs_train[sample]].values
        X_test = X_samp.iloc[idxs_test[sample]].values
        Y_train = Y_samp.iloc[idxs_train[sample]].values
        Y_test = Y_samp.iloc[idxs_test[sample]].values
        
        ## retrain the model using the best lambda
        holdout_results = solve.single_solve(
            X_train[subset_n, subset_feat], X_test[:, subset_feat],
            Y_train[subset_n], Y_test, lambdas = np.array([mykwargs[sample]['lambdas'][best_lambda_idxs[sample]]]),
            return_preds = True, return_model = True, clip_bounds = [mykwargs[sample]['clip_bounds'][0]],
            svd_solve = False,
        )
        
        ## store sets of weights
        wts_cont[sample] = holdout_results['models'][0][0][0]
        
        ## store performance metrics
        globals()[f'in_mse_{x}_cont'][sample] = holdout_results['metrics_train'][0][0][0]['mse']
        globals()[f'in_r2_{x}_cont'][sample] = holdout_results['metrics_train'][0][0][0]['r2_score']
        globals()[f'out_mse_{x}_cont'][sample] = holdout_results['metrics_test'][0][0][0]['mse']
        globals()[f'out_r2_{x}_cont'][sample] = holdout_results['metrics_test'][0][0][0]['r2_score']
    
    ## save weights vector for each continent
    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_{}_population.csv'.format(x)), wts_cont, delimiter = ',')

################################################
## D) train model in each continent - modified
################################################

## move Grenada and Trinidad and Tobago to north america
latlonsdf_samp.loc[(latlonsdf_samp['lon'] >= -65) & (latlonsdf_samp['lon'] < -60) & (latlonsdf_samp['lat'] >= 10) & (latlonsdf_samp['lat'] < 12.5), 'samp'] = 0

for x in ['mosaiks_only', 'nl_only', 'both']:
    
    ## select the dataset for training codes
    Y_df = pop[['ln_pop_density']]
    if x == 'mosaiks_only':
        X_df = mosaiks_feat
    elif x == 'nl_only':
        X_df = nl
    elif x == 'both':
        X_df = new_X
    
    ## save kfold results from all samples in one list
    N = 6
    kfold_results = [None] * N
    ll_train = [None] * N
    best_preds = [None] * N
    mykwargs = [None] * N
    best_lambda_idxs = [None] * N
    idxs_train = [None] * N
    idxs_test = [None] * N
    globalclipping = True
    
    for s in range(N):
        ids_samp = np.where(latlonsdf_samp['samp'] == s)
        if len(ids_samp[0]) >= 500:
            (
                kfold_results[s], ll_train[s], idxs_train[s], idxs_test[s], mykwargs[s],
            ) = solve.split_world_sample_solve(
                X_df, Y_df, latlonsdf_samp, sample = s, subset_n = subset_n, subset_feat = subset_feat,
                num_folds = 5, solve_function = solver, globalclipping = globalclipping,
                lambdas = lambdas, **solver_kwargs
            )
            best_lambda_idxs[s], _, best_preds[s] = ir.interpret_kfold_results(
                kfold_results[s], 'r2_score', hps = [('lambdas', lambdas)],
            )
        else:
            print('----Cannot estimate ridge regression with cross-val: Too few observations!----')
            
    ## remove none-type entries from list
    kfold_results_clean = [res for res in kfold_results if res != None]
    best_preds_clean = [res for res in best_preds if res != None]
    
    myN = len(best_preds_clean)
    
    ## initialize weights vector
    wts_cont = [None] * myN
    globals()[f'in_mse_{x}_cont_fixed'] = [None] * myN
    globals()[f'in_r2_{x}_cont_fixed'] = [None] * myN
    globals()[f'out_mse_{x}_cont_fixed'] = [None] * myN
    globals()[f'out_r2_{x}_cont_fixed'] = [None] * myN
    
    for sample in range(myN):
        
        ## initialize the weights vector for each continent
        wts_this_cont = [None]
        
        ## cut sample for each continent
        ids_samp = np.where(latlonsdf_samp['samp'] == sample)
        X_samp = X_df.iloc[ids_samp]
        Y_samp = Y_df.iloc[ids_samp]
        
        ## split into train vs test sets
        X_train = X_samp.iloc[idxs_train[sample]].values
        X_test = X_samp.iloc[idxs_test[sample]].values
        Y_train = Y_samp.iloc[idxs_train[sample]].values
        Y_test = Y_samp.iloc[idxs_test[sample]].values
        
        ## retrain the model using the best lambda
        holdout_results = solve.single_solve(
            X_train[subset_n, subset_feat], X_test[:, subset_feat],
            Y_train[subset_n], Y_test, lambdas = np.array([mykwargs[sample]['lambdas'][best_lambda_idxs[sample]]]),
            return_preds = True, return_model = True, clip_bounds = [mykwargs[sample]['clip_bounds']][0],
            svd_solve = False,
        )
        
        ## store sets of weights
        wts_cont[sample] = holdout_results['models'][0][0][0]
        
        ## store performance metrics
        globals()[f'in_mse_{x}_cont_fixed'][sample] = holdout_results['metrics_train'][0][0][0]['mse']
        globals()[f'in_r2_{x}_cont_fixed'][sample] = holdout_results['metrics_train'][0][0][0]['r2_score']
        globals()[f'out_mse_{x}_cont_fixed'][sample] = holdout_results['metrics_test'][0][0][0]['mse']
        globals()[f'out_r2_{x}_cont_fixed'][sample] = holdout_results['metrics_test'][0][0][0]['r2_score']
    
    ## save weights vector for each continent
    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_fixed_{}_population.csv'.format(x)), wts_cont, delimiter = ',')

for sample in ['in', 'out']:
    
    ## store mse and r-square into one csv file - by-continent = 0 (except for TTO and GRD) 
    rows = [
        {'Metrics': 'Global-scale', 
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_global'],
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_global'],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_global'],
         'NL: R-square': globals()[f'{sample}_r2_nl_only_global'],
         'Both: MSE': globals()[f'{sample}_mse_both_global'],
         'Both: R-square': globals()[f'{sample}_r2_both_global']},
        {'Metrics': 'By-continent', 
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_cont'][0],
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_cont'][0],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_cont'][0],
         'NL: R-square': globals()[f'{sample}_r2_nl_only_cont'][0],
         'Both: MSE': globals()[f'{sample}_mse_both_cont'][0],
         'Both: R-square': globals()[f'{sample}_r2_both_cont'][0]},
        {'Metrics': 'By-continent fixed', 
         'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_cont_fixed'][3],
         'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_cont_fixed'][3],
         'NL: MSE': globals()[f'{sample}_mse_nl_only_cont_fixed'][3],
         'NL: R-square': globals()[f'{sample}_r2_nl_only_cont_fixed'][3],
         'Both: MSE': globals()[f'{sample}_mse_both_cont_fixed'][3],
         'Both: R-square': globals()[f'{sample}_r2_both_cont_fixed'][3]}
    ]
    
    fn = os.path.join(c.out_dir, 'metrics', 'global_grid_population_{}sample_metrics.csv'.format(sample))
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MOSAIKS: MSE', 'MOSAIKS: R-square', 'NL: MSE', 'NL: R-square', 'Both: MSE', 'Both: R-square'])
        writer.writeheader()
        writer.writerows(rows)

