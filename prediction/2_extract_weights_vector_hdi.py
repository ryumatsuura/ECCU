## This script trains the model for HDI predictions
## and stores the weights vectors

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

## load MOSAIKS feature aggregated to national/subnational level
mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM0_polygon_X_creation_pop_weight=True.p'))
mosaiks_subnat_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM1_polygon_X_creation_pop_weight=True.p')).drop(columns = 'GDLCODE')

## load HDI measures
hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'hdi', 'HDI_indicators_and_indices_adm0_clean.p')).loc[mosaiks_feat.index]
hdi_subnat = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'hdi', 'HDI_indicators_and_indices_clean.p')).loc[mosaiks_subnat_feat.index]

## rename columns 
hdi = hdi.rename(columns = {'Sub-national HDI': 'hdi', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})
hdi_subnat = hdi_subnat.rename(columns = {'Sub-national HDI': 'hdi', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})

###################################
## B) train model at global scale
###################################

## specify outcome variables
tasks = ['hdi', 'health', 'income', 'ed']

for task in tasks:
    for level in ['nat', 'subnat']:
        
        ## restructure the dataset for training codes - needs to be numpy arrays 
        if level == 'nat':
            hdi_np = np.array(hdi[task])
            mosaiks_feat_np = mosaiks_feat.iloc[:, 0:4000].to_numpy()
        elif level == 'subnat':
            hdi_np = np.array(hdi_subnat[task])
            mosaiks_feat_np = mosaiks_subnat_feat.iloc[:, 0:4000].to_numpy()
        
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
            np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_{}.csv'.format(task)), wts, delimiter = ',')

