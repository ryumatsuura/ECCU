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

## load population and nighttime light data
pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_global.csv'), index_col = 0)
nl = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'outcome_sampled_nightlights_global.csv'), index_col = 0)

## drop NaN in population data
pop = pop.loc[pop['population'].isnull() == False]
feat = feat.merge(pop[['lat', 'lon']], left_on = ['lat', 'lon'], right_on = ['lat', 'lon'])
nl = nl.merge(pop[['lat', 'lon']], left_on = ['lat', 'lon'], right_on = ['lat', 'lon'])

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

## store mse and r-square into one csv file - by-continent = 0 (except for TTO and GRD) 
rows = [
    {'Metrics': 'Global-scale', 'MSE': mse_global, 'R-square': r2_global},
    {'Metrics': 'By-continent', 'MSE': mse_by_cont[0], 'R-square': r2_by_cont[0]},
    {'Metrics': 'By-continent fixed', 'MSE': mse_by_cont_fixed[3], 'R-square': r2_by_cont_fixed[3]}
]

fn = os.path.join(c.out_dir, 'metrics', 'eccu_population_grid_insample_metrics.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MSE', 'R-square'])
    writer.writeheader()
    writer.writerows(rows)

