## This script trains the model for HDI predictions
## and stores the weights vectors

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
solver_kwargs = {'return_preds':True, 'svd_solve':False}

## define function for weighted average
def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

## specify outcome variables
tasks = ['hdi', 'gni', 'health', 'income', 'ed']

#######################
## A) load label data
#######################

## load MOSAIKS feature aggregated to national/subnational level
mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM0_polygon_X_creation_pop_weight=True.p'))
mosaiks_subnat_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM1_polygon_X_creation_pop_weight=True.p')).drop(columns = 'GDLCODE')

## extract iso code from subnational code
mosaiks_subnat_feat['iso_code'] = mosaiks_subnat_feat.index.values
mosaiks_subnat_feat['iso_code'] = mosaiks_subnat_feat.iso_code.str[0:3]

## demean subnational mosaiks feature
mosaiks_subnat_demean_feat = pd.DataFrame(index = mosaiks_subnat_feat.index)
for column in mosaiks_feat.columns:
    merged = mosaiks_subnat_feat[['iso_code', column]].merge(mosaiks_feat[column], left_on = 'iso_code', right_index = True, suffixes = ('_subnat', '_nat'))
    merged[column] = merged['{}_subnat'.format(column)] - merged['{}_nat'.format(column)]
    mosaiks_subnat_demean_feat = mosaiks_subnat_demean_feat.merge(merged[column], left_index = True, right_index = True)

## remove iso code from subnat
mosaiks_subnat_feat = mosaiks_subnat_feat.drop(columns = ['iso_code'])

## load HDI measures
hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'hdi', 'HDI_indicators_and_indices_adm0_clean.p')).loc[mosaiks_feat.index]
hdi_subnat = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'hdi', 'HDI_indicators_and_indices_clean.p')).loc[mosaiks_subnat_feat.index]

## rename columns 
hdi = hdi.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})
hdi_subnat = hdi_subnat.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})

for task in tasks:
    avg = hdi_subnat.groupby('ISO_Code').apply(w_avg, task, 'Population size in millions')
    hdi_subnat = hdi_subnat.reset_index().merge(avg.rename('avg'), left_on = 'ISO_Code', right_on = 'ISO_Code').set_index('GDLCODE')
    hdi_subnat['demeaned_{}'.format(task)] = hdi_subnat[task] - hdi_subnat['avg']
    hdi_subnat = hdi_subnat.drop(columns = ['avg'])

###################################
## B) train model at global scale
###################################

for task in tasks:
    for y in ['level', 'demeaned']:
        for level in ['nat', 'subnat']:
            
            ## skip national-demeaned combination
            if level == 'nat' and y == 'demeaned':
                continue
            
            ## restructure the dataset for training codes - needs to be numpy arrays 
            if level == 'nat':
                hdi_np = np.array(hdi[task])
                mosaiks_feat_np = mosaiks_feat.to_numpy()
            elif level == 'subnat':
                if y == 'level':
                    hdi_np = np.array(hdi_subnat[task])
                    mosaiks_feat_np = mosaiks_subnat_feat.to_numpy()
                elif y == 'demeaned':
                    hdi_np = np.array(hdi_subnat.loc[hdi_subnat['Level'] == 'Subnat']['{}_{}'.format(y, task)])
                    mosaiks_feat_np = mosaiks_subnat_demean_feat.loc[hdi_subnat.loc[hdi_subnat['Level'] == 'Subnat'].index].to_numpy()
            
            ## set the bounds
            mins = hdi_np.min(axis = 0)
            maxs = hdi_np.max(axis = 0)
            solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
            
            ## split the data into training vs testing sets
            X_train, X_test, Y_train, Y_test, idxs_train, idsx_test = parse.split_data_train_test(
                mosaiks_feat_np, hdi_np, frac_test = c.ml_model['test_set_frac'], return_idxs = True
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
            if level == 'nat':
                np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}.csv'.format(task)), wts, delimiter = ',')
            elif level == 'subnat':
                if y == 'level':
                    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_{}.csv'.format(task)), wts, delimiter = ',')
                elif y == 'demeaned':
                    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_{}_{}.csv'.format(task, y)), wts, delimiter = ',')

