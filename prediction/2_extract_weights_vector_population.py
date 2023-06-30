## This script trains the model at different scales and 
## stores the weights vectors

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

#######################
## A) load label data
#######################

## load indexed MOSAIKS feature data 
merged = pd.read_pickle(os.path.join(c.data_dir, 'int', 'feature_matrices', 'grid_features.pkl'))

## load population density data
c.grid['area'] = 'WORLD'
c.sampling['n_samples'] = 1000000
c = io.get_filepaths(c, 'population')
Y = io.get_Y(c, 'population')
Y, _ = transforms.dropna_Y(Y, 'population')
_, Y, _ = getattr(transforms, f'transform_population')(Y, Y, Y, True)

## choose only those that match valid images
Y = Y.loc[Y.index.isin(merged.index)]

## merge x and y
Y, X, latlons, ids = parse.merge(Y, merged.iloc[:, 0:4000], merged[['Y', 'X']].rename(columns = {'Y':'lat', 'X':'lon'}), pd.Series(Y.index, index = Y.index))
del merged

## set the bounds
mins = Y.min(axis = 0)
maxs = Y.max(axis = 0)
solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T

###################################
## B) train model at global scale
###################################

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

wts_global = holdout_results['models'][0][0][0]
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_population.csv'), wts_global, delimiter = ',')

#####################################
## C) train model in each continent
#####################################

X_df = pd.DataFrame(X, index = ids)
Y_df = pd.DataFrame(Y, index = ids)

## split latlons into continents
latlonsdf_samp = parse.split_world_sample(
    pd.DataFrame(latlons, index = ids, columns = ['lat', 'lon'])
)

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

for x in range(N):
    ids_samp = np.where(latlonsdf_samp['samp'] == x)
    if len(ids_samp[0]) >= 500:
        (
            kfold_results[x], ll_train[x], idxs_train[x], idxs_test[x], mykwargs[x],
        ) = solve.split_world_sample_solve(
            X_df, Y_df, latlonsdf_samp, sample = x, subset_n = subset_n, subset_feat = subset_feat,
            num_folds = c.ml_model['n_folds'], solve_function = solver, globalclipping = globalclipping,
            lambdas = lambdas, **solver_kwargs
        )
        best_lambda_idxs[x], _, best_preds[x] = ir.interpret_kfold_results(
            kfold_results[x], 'r2_score', hps = [('lambdas', lambdas)],
        )
    else:
        print('----Cannot estimate ridge regression with cross-val: Too few observations!----')

## remove none-type entries from list
kfold_results_clean = [x for x in kfold_results if x != None]
best_preds_clean = [x for x in best_preds if x != None]

myN = len(best_preds_clean)

## initialize weights vector
wts_by_cont = [None] * myN

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
    wts_by_cont[sample] = holdout_results['models'][0][0][0]

## save weights vector for each continent
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_population.csv'), wts_by_cont, delimiter = ',')

################################################
## D) train model in each continent - modified
################################################

## move Grenada and Trinidad and Tobago to north america
latlonsdf_samp.loc[(latlonsdf_samp['lon'] >= -65) & (latlonsdf_samp['lon'] < -60) & (latlonsdf_samp['lat'] >= 10) & (latlonsdf_samp['lat'] < 12.5), 'samp'] = 0

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

for x in range(N):
    ids_samp = np.where(latlonsdf_samp['samp'] == x)
    if len(ids_samp[0]) >= 500:
        (
            kfold_results[x], ll_train[x], idxs_train[x], idxs_test[x], mykwargs[x],
        ) = solve.split_world_sample_solve(
            X_df, Y_df, latlonsdf_samp, sample = x, subset_n = subset_n, subset_feat = subset_feat,
            num_folds = c.ml_model['n_folds'], solve_function = solver, globalclipping = globalclipping,
            lambdas = lambdas, **solver_kwargs
        )
        best_lambda_idxs[x], _, best_preds[x] = ir.interpret_kfold_results(
            kfold_results[x], 'r2_score', hps = [('lambdas', lambdas)],
        )
    else:
        print('----Cannot estimate ridge regression with cross-val: Too few observations!----')

## remove none-type entries from list
kfold_results_clean = [x for x in kfold_results if x != None]
best_preds_clean = [x for x in best_preds if x != None]

myN = len(best_preds_clean)

## initialize weights vector
wts_by_cont = [None] * myN

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
    wts_by_cont[sample] = holdout_results['models'][0][0][0]

## save weights vector for each continent
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_fixed_population.csv'), wts_by_cont, delimiter = ',')

###############################
## E) train model in Barbados
###############################

## E-1. load Barbados population data

## load Y and X data for Barbados
Y_brb = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_BRB.csv'), index_col = 0)
X_brb = pd.read_csv(os.path.join(c.data_dir, 'int', 'feature_matrices', 'Mosaiks_features_brb.csv'))
Y_brb = pd.merge(Y_brb, X_brb[['Lat', 'Lon']], left_on = ['lon', 'lat'], right_on = ['Lon', 'Lat'])[['population']]

## remove nan from population
indices = np.where(Y_brb['population'].notnull())[0]
Y_brb = Y_brb.reindex(indices)
X_brb = X_brb.reindex(indices)
Y_brb, X_brb, latlons_brb, ids_brb = parse.merge(Y_brb, X_brb.iloc[:, 3:X_brb.shape[1]], X_brb[['Lat', 'Lon']].rename(columns = {'Lat':'lat', 'Lon':'lon'}), pd.Series(Y_brb.index, index = Y_brb.index))

## E-2. train the model

X_brb_train, X_brb_test, Y_brb_train, Y_brb_test, idxs_brb_train, idxs_brb_test = parse.split_data_train_test(
    X_brb, Y_brb, frac_test = c.ml_model['test_set_frac'], return_idxs = True
)
latlons_brb_train = latlons_brb[idxs_brb_train]
latlons_brb_test = latlons_brb[idxs_brb_test]

## define limit to subsets
Y_brb_train = Y_brb_train[subset_n]
X_brb_train = X_brb_train[subset_n, subset_feat]
latlons_brb_train = latlons_brb_train[subset_n]

kfold_results = solve.kfold_solve(
    X_brb_train, Y_brb_train, solve_function = solver, num_folds = c.ml_model['n_folds'], 
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
    X_brb_train[subset_n, subset_feat], X_brb_test[:, subset_feat], Y_brb_train[subset_n], Y_brb_test,
    lambdas = best_lambda, return_preds = True, return_model = True, clip_bounds = [np.array([mins, maxs])]
)

wts_brb = holdout_results['models'][0][0][0]
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_population.csv'), wts_brb, delimiter = ',')

