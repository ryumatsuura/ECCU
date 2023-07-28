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

##lambdas = lambdas_single = c.ml_model['global_lambdas']
lambdas = lambdas_single = np.hstack([1, (np.logspace(-3, 6, 8) ** .3)])
##lambdas = lambdas_single = c.ml_model['global_lambdas']
solver = solve.ridge_regression
solver_kwargs = {'return_preds': True, 'svd_solve': False}

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

## load nighttime light data
nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'dmsp_nightlight_features_for_adm0_polygons_20_bins_GPW_pop_weighted.p')).loc[mosaiks_feat.index]
subnat_nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'dmsp_nightlight_features_for_hdi_polygons_20_bins_GPW_pop_weighted.p')).loc[mosaiks_subnat_feat.index]

## demean 
for df in (mosaiks_subnat_feat, subnat_nl):
    
    ## identify national-level features
    if any(df.equals(y) for y in [mosaiks_subnat_feat]):
        nat_feat = mosaiks_feat
    elif any(df.equals(y) for y in [subnat_nl]):
        nat_feat = nl
    
    ## extract iso code from subnational code
    df['iso_code'] = df.index.values
    df['iso_code'] = df.iso_code.str[0:3]
    
    ## demean subnational mosaiks feature
    if any(df.equals(y) for y in [mosaiks_subnat_feat]):
        mosaiks_subnat_demean_feat = pd.DataFrame(index = df.index)
    elif any(df.equals(y) for y in [subnat_nl]):
        subnat_demean_nl = pd.DataFrame(index = df.index)
    for column in nat_feat.columns:
        merged = df[['iso_code', column]].merge(nat_feat[column], left_on = 'iso_code', right_index = True, suffixes = ('_subnat', '_nat'))
        merged[column] = merged['{}_subnat'.format(column)] - merged['{}_nat'.format(column)]
        if any(df.equals(y) for y in [mosaiks_subnat_feat]):
            mosaiks_subnat_demean_feat = mosaiks_subnat_demean_feat.merge(merged[column], left_index = True, right_index = True)
        elif any(df.equals(y) for y in [subnat_nl]):
            subnat_demean_nl = subnat_demean_nl.merge(merged[column], left_index = True, right_index = True)
    
    del nat_feat, merged

## remove iso code from subnat
mosaiks_subnat_feat = mosaiks_subnat_feat.drop(columns = ['iso_code'])
subnat_nl = subnat_nl.drop(columns = ['iso_code'])

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
        for x in ['mosaiks_only', 'nl_only', 'both']:
            for l in ['nat', 'subnat']:
                
                ## skip national-demeaned combination
                if l == 'nat' and y == 'demeaned':
                    continue
                
                ## combine mosaiks and nightlight
                if y == 'level':
                    if x == 'both':
                        if l == 'nat':
                            new_X = pd.merge(mosaiks_feat, nl, left_index = True, right_index = True)
                        elif l == 'subnat':
                            new_X = pd.merge(mosaiks_subnat_feat, subnat_nl, left_index = True, right_index = True)
                elif y == 'demeaned':
                    if x == 'both':
                        new_X = pd.merge(mosaiks_subnat_demean_feat, subnat_demean_nl, left_index = True, right_index = True)
                
                ## restructure the dataset for training codes - needs to be numpy arrays 
                if l == 'nat':
                    hdi_np = np.array(hdi[task])
                    if x == 'mosaiks_only':
                        X_np = mosaiks_feat.to_numpy()
                    elif x == 'nl_only':
                        X_np = nl.to_numpy()
                    elif x == 'both':
                        X_np = new_X.to_numpy()
                elif l == 'subnat':
                    if y == 'level':
                        hdi_np = np.array(hdi_subnat[task])
                        if x == 'mosaiks_only':
                            X_np = mosaiks_subnat_feat.to_numpy()
                        elif x == 'nl_only':
                            X_np = subnat_nl.to_numpy()
                        elif x == 'both':
                            X_np = new_X.to_numpy()
                    elif y == 'demeaned':
                        hdi_np = np.array(hdi_subnat.loc[hdi_subnat['Level'] == 'Subnat']['{}_{}'.format(y, task)])
                        if x == 'mosaiks_only':
                            X_np = mosaiks_subnat_demean_feat.loc[hdi_subnat.loc[hdi_subnat['Level'] == 'Subnat'].index].to_numpy()
                        elif x == 'nl_only':
                            X_np = subnat_demean_nl.loc[hdi_subnat.loc[hdi_subnat['Level'] == 'Subnat'].index].to_numpy()
                        elif x == 'both':
                            X_np = new_X.loc[hdi_subnat.loc[hdi_subnat['Level'] == 'Subnat'].index].to_numpy()
                
                ## set the bounds
                mins = hdi_np.min(axis = 0)
                maxs = hdi_np.max(axis = 0)
                solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
                
                ## split the data into training vs testing sets
                X_train, X_test, Y_train, Y_test, idxs_train, idsx_test = parse.split_data_train_test(
                    X_np, hdi_np, frac_test = c.ml_model['test_set_frac'], return_idxs = True
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
                if y == 'level':
                        np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}_{}_{}.csv'.format(l, x, task)), wts, delimiter = ',')
                elif y == 'demeaned':
                    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}_{}_{}_{}.csv'.format(l, x, y, task)), wts, delimiter = ',')
                
                ## store performance metrics
                globals()[f'mse_{x}_{l}_{y}'] = holdout_results['metrics_test'][0][0][0]['mse']
                globals()[f'r2_{x}_{l}_{y}'] = holdout_results['metrics_test'][0][0][0]['r2_score']
                
                del hdi_np, X_np
                if x == 'both':
                    del new_X
    
    ## store mse and r-square into one csv file for each task
    rows = [
        {'Metrics': 'National-level',
         'MOSAIKS: MSE': mse_mosaiks_only_nat_level,
         'MOSAIKS: R-square': r2_mosaiks_only_nat_level,
         'NL: MSE': mse_nl_only_nat_level,
         'NL: R-square': r2_nl_only_nat_level,
         'Both: MSE': mse_both_nat_level,
         'Both: R-square': r2_both_nat_level},
        {'Metrics': 'Subnational-level',
         'MOSAIKS: MSE': mse_mosaiks_only_subnat_level,
         'MOSAIKS: R-square': r2_mosaiks_only_subnat_level,
         'NL: MSE': mse_nl_only_subnat_level,
         'NL: R-square': r2_nl_only_subnat_level,
         'Both: MSE': mse_both_subnat_level,
         'Both: R-square': r2_both_subnat_level},
        {'Metrics': 'Subnational-demeaned',
         'MOSAIKS: MSE': mse_mosaiks_only_subnat_demeaned,
         'MOSAIKS: R-square': r2_mosaiks_only_subnat_demeaned,
         'NL: MSE': mse_nl_only_subnat_demeaned,
         'NL: R-square': r2_nl_only_subnat_demeaned,
         'Both: MSE': mse_both_subnat_demeaned,
         'Both: R-square': r2_both_subnat_demeaned}
    ]
    
    fn = os.path.join(c.out_dir, 'metrics', 'eccu_{}_adm_insample_metrics.csv'.format(task))
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MOSAIKS: MSE', 'MOSAIKS: R-square', 'NL: MSE', 'NL: R-square', 'Both: MSE', 'Both: R-square'])
        writer.writeheader()
        writer.writerows(rows)

