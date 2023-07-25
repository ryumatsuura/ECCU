## This script trains the model at different scales and 
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

lambdas = lambdas_single = c.ml_model['global_lambdas']
solver = solve.ridge_regression
solver_kwargs = {'return_preds': True, 'svd_solve': False}

#######################
## A) load label data
#######################

## load indexed MOSAIKS feature data 
feat = pd.read_pickle(os.path.join(c.features_dir, 'grid_features.pkl'))

## load population data
pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_global.csv'), index_col = 0)

## drop NaN in population data
pop = pop.loc[pop['population'].isnull() == False]
feat = feat.merge(pop[['lat', 'lon']], left_on = ['lat', 'lon'], right_on = ['lat', 'lon'])

## convert to numpy array
Y = np.array(pop['population'])
X = feat.iloc[:, 3:4003].to_numpy()
latlons = pop[['lat', 'lon']].to_numpy()
ids = pop.index.values

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

## store performance metrics
mse_global = holdout_results['metrics_test'][0][0][0]['mse']
r2_global = holdout_results['metrics_test'][0][0][0]['r2_score']

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
mse_by_cont = [None] * myN
r2_by_cont = [None] * myN

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
    
    ## store performance metrics
    mse_by_cont[sample] = holdout_results['metrics_test'][0][0][0]['mse']
    r2_by_cont[sample] = holdout_results['metrics_test'][0][0][0]['r2_score']

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
mse_by_cont_fixed = [None] * myN
r2_by_cont_fixed = [None] * myN

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
    
    ## store performance metrics
    mse_by_cont_fixed[sample] = holdout_results['metrics_test'][0][0][0]['mse']
    r2_by_cont_fixed[sample] = holdout_results['metrics_test'][0][0][0]['r2_score']

## save weights vector for each continent
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_fixed_population.csv'), wts_by_cont, delimiter = ',')

############################################
## E) train model in neighboring countries
############################################

## E-1. Train by country

for country in ['brb', 'glp', 'mtq']:
    
    ## load Y and X data for Barbados/Guadeloupe/Martinique
    Y_ = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_{}.csv'.format(country.upper())), index_col = 0)
    X_ = pd.read_csv(os.path.join(c.features_dir, 'Mosaiks_features_{}.csv'.format(country)))
    Y_pop = pd.merge(Y_, X_[['Lat', 'Lon']], left_on = ['lon', 'lat'], right_on = ['Lon', 'Lat'])[['population']]
    X_pop = pd.merge(X_, Y_[['lat', 'lon']], left_on = ['Lon', 'Lat'], right_on = ['lon', 'lat']).drop(columns = ['lon', 'lat'])
    
    ## remove nan from population
    indices = np.where(Y_pop['population'].notnull())[0]
    Y_pop = Y_pop.reindex(indices)
    X_pop = X_pop.reindex(indices)
    Y_pop, X_pop, latlons_, ids_ = parse.merge(Y_pop, X_pop.iloc[:, 3:X_pop.shape[1]], X_pop[['Lat', 'Lon']].rename(columns = {'Lat':'lat', 'Lon':'lon'}), pd.Series(Y_pop.index, index = Y_pop.index))
    
    X__train, X__test, Y__train, Y__test, idxs__train, idxs__test = parse.split_data_train_test(
        X_pop, Y_pop, frac_test = c.ml_model['test_set_frac'], return_idxs = True
    )
    latlons__train = latlons_[idxs__train]
    latlons__test = latlons_[idxs__test]
    
    ## define limit to subsets
    Y__train = Y__train[subset_n]
    X__train = X__train[subset_n, subset_feat]
    latlons__train = latlons__train[subset_n]
    
    kfold_results = solve.kfold_solve(
        X__train, Y__train, solve_function = solver, num_folds = c.ml_model['n_folds'], 
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
        X__train[subset_n, subset_feat], X__test[:, subset_feat], Y__train[subset_n], Y__test,
        lambdas = best_lambda, return_preds = True, return_model = True, clip_bounds = [np.array([mins, maxs])]
    )
    
    wts_ = holdout_results['models'][0][0][0]
    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', '{}_population.csv'.format(country)), wts_, delimiter = ',')
    
    ## store performance metrics
    if country == 'brb':
        mse_brb = holdout_results['metrics_test'][0][0][0]['mse']
        r2_brb = holdout_results['metrics_test'][0][0][0]['r2_score']
    elif country == 'glp':
        mse_glp = holdout_results['metrics_test'][0][0][0]['mse']
        r2_glp = holdout_results['metrics_test'][0][0][0]['r2_score']
    elif country == 'mtq':
        mse_mtq = holdout_results['metrics_test'][0][0][0]['mse']
        r2_mtq = holdout_results['metrics_test'][0][0][0]['r2_score']

## E-2. Train for all neighboring countries altogether

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
    {'Metrics': 'Global-scale', 'MSE': mse_global, 'R-square': r2_global},
    {'Metrics': 'By-continent', 'MSE': mse_by_cont[0], 'R-square': r2_by_cont[0]},
    {'Metrics': 'By-continent fixed', 'MSE': mse_by_cont_fixed[3], 'R-square': r2_by_cont_fixed[3]},
    {'Metrics': 'Barbados-based', 'MSE': mse_brb, 'R-square': r2_brb},
    {'Metrics': 'Guadeloupe-based', 'MSE': mse_glp, 'R-square': r2_glp},
    {'Metrics': 'Martinique-based', 'MSE': mse_mtq, 'R-square': r2_mtq},
    {'Metrics': 'Neighbors-based', 'MSE': mse_comb, 'R-square': r2_comb}
]

fn = os.path.join(c.out_dir, 'metrics', 'eccu_population_grid_insample_metrics.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MSE', 'R-square'])
    writer.writeheader()
    writer.writerows(rows)

