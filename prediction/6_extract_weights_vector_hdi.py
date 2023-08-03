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

lambdas = lambdas_single = np.hstack([1, (np.logspace(-3, 6, 8) ** .3)])
solver = solve.ridge_regression
solver_kwargs = {'return_preds': True, 'svd_solve': False}

## define function for weighted average
def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

## specify outcome variables
tasks = ['hdi', 'gni', 'health', 'income', 'ed']

## create folders if not exist
if not os.path.exists(os.path.join(c.data_dir, 'int', 'weights')):
    os.makedirs(os.path.join(c.data_dir, 'int', 'weights'))

if not os.path.exists(os.path.join(c.out_dir, 'metrics')):
    os.makedirs(os.path.join(c.out_dir, 'metrics'))

#######################
## A) load label data
#######################

## load MOSAIKS feature aggregated to national/subnational level
mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM0_polygon_X_creation_pop_weight=True.p'))
subnat_mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'GDL_ADM1_polygon_X_creation_pop_weight=True.p')).drop(columns = 'GDLCODE')

## load nighttime light data
nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'dmsp_nightlight_features_for_adm0_polygons_20_bins_GPW_pop_weighted.p')).loc[mosaiks_feat.index]
subnat_nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'dmsp_nightlight_features_for_hdi_polygons_20_bins_GPW_pop_weighted.p')).loc[subnat_mosaiks_feat.index]

## demean 
for df in (subnat_mosaiks_feat, subnat_nl):
    
    ## identify national-level features
    if any(df.equals(y) for y in [subnat_mosaiks_feat]):
        nat_feat = mosaiks_feat
    elif any(df.equals(y) for y in [subnat_nl]):
        nat_feat = nl
    
    ## extract iso code from subnational code
    df['iso_code'] = df.index.values
    df['iso_code'] = df.iso_code.str[0:3]
    
    ## demean subnational mosaiks feature
    if any(df.equals(y) for y in [subnat_mosaiks_feat]):
        subnat_demean_mosaiks_feat = pd.DataFrame(index = df.index)
    elif any(df.equals(y) for y in [subnat_nl]):
        subnat_demean_nl = pd.DataFrame(index = df.index)
    for column in nat_feat.columns:
        merged = df[['iso_code', column]].merge(nat_feat[column], left_on = 'iso_code', right_index = True, suffixes = ('_subnat', '_nat'))
        merged[column] = merged['{}_subnat'.format(column)] - merged['{}_nat'.format(column)]
        if any(df.equals(y) for y in [subnat_mosaiks_feat]):
            subnat_demean_mosaiks_feat = subnat_demean_mosaiks_feat.merge(merged[column], left_index = True, right_index = True)
        elif any(df.equals(y) for y in [subnat_nl]):
            subnat_demean_nl = subnat_demean_nl.merge(merged[column], left_index = True, right_index = True)
    
    del nat_feat, merged

## remove iso code from subnat
subnat_mosaiks_feat = subnat_mosaiks_feat.drop(columns = ['iso_code'])
subnat_nl = subnat_nl.drop(columns = ['iso_code'])

## load HDI measures
hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_adm0_clean.p')).loc[mosaiks_feat.index]
subnat_hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_clean.p')).loc[subnat_mosaiks_feat.index]

## rename columns 
hdi = hdi.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})
subnat_hdi = subnat_hdi.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})

for task in tasks:
    avg = subnat_hdi.groupby('ISO_Code').apply(w_avg, task, 'Population size in millions')
    subnat_hdi = subnat_hdi.reset_index().merge(avg.rename('avg'), left_on = 'ISO_Code', right_on = 'ISO_Code').set_index('GDLCODE')
    subnat_hdi['demeaned_{}'.format(task)] = subnat_hdi[task] - subnat_hdi['avg']
    subnat_hdi = subnat_hdi.drop(columns = ['avg'])

## combine MOSAIKS and nightlight
new_X = pd.merge(mosaiks_feat, nl, left_index = True, right_index = True)
subnat_new_X = pd.merge(subnat_mosaiks_feat, subnat_nl, left_index = True, right_index = True)
subnat_demean_new_X = pd.merge(subnat_demean_mosaiks_feat, subnat_demean_nl, left_index = True, right_index = True)

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
                        hdi_np = np.array(subnat_hdi[task])
                        if x == 'mosaiks_only':
                            X_np = subnat_mosaiks_feat.to_numpy()
                        elif x == 'nl_only':
                            X_np = subnat_nl.to_numpy()
                        elif x == 'both':
                            X_np = subnat_new_X.to_numpy()
                    elif y == 'demeaned':
                        hdi_np = np.array(subnat_hdi.loc[subnat_hdi['Level'] == 'Subnat']['{}_{}'.format(y, task)])
                        if x == 'mosaiks_only':
                            X_np = subnat_demean_mosaiks_feat.loc[subnat_hdi.loc[subnat_hdi['Level'] == 'Subnat'].index].to_numpy()
                        elif x == 'nl_only':
                            X_np = subnat_demean_nl.loc[subnat_hdi.loc[subnat_hdi['Level'] == 'Subnat'].index].to_numpy()
                        elif x == 'both':
                            X_np = subnat_demean_new_X.loc[subnat_hdi.loc[subnat_hdi['Level'] == 'Subnat'].index].to_numpy()
                
                ## set the bounds
                mins = hdi_np.min(axis = 0)
                maxs = hdi_np.max(axis = 0)
                solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T
                
                ## split the data into training vs testing sets
                X_train, X_test, Y_train, Y_test, idxs_train, idsx_test = parse.split_data_train_test(
                    X_np, hdi_np, frac_test = 0.2, return_idxs = True
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
                if y == 'level':
                        np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}_{}_{}.csv'.format(l, x, task)), wts, delimiter = ',')
                elif y == 'demeaned':
                    np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'global_{}_{}_{}_{}.csv'.format(l, x, task, y)), wts, delimiter = ',')
                
                ## store performance metrics
                globals()[f'in_mse_{x}_{l}_{y}'] = holdout_results['metrics_train'][0][0][0]['mse']
                globals()[f'in_r2_{x}_{l}_{y}'] = holdout_results['metrics_train'][0][0][0]['r2_score']
                globals()[f'out_mse_{x}_{l}_{y}'] = holdout_results['metrics_test'][0][0][0]['mse']
                globals()[f'out_r2_{x}_{l}_{y}'] = holdout_results['metrics_test'][0][0][0]['r2_score']
                
                del hdi_np, X_np
    
    for sample in ['in', 'out']:
        
        ## store mse and r-square into one csv file for each task
        rows = [
            {'Metrics': 'National-level',
             'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_nat_level'],
             'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_nat_level'],
             'NL: MSE': globals()[f'{sample}_mse_nl_only_nat_level'],
             'NL: R-square': globals()[f'{sample}_r2_nl_only_nat_level'],
             'Both: MSE': globals()[f'{sample}_mse_both_nat_level'],
             'Both: R-square': globals()[f'{sample}_r2_both_nat_level']},
            {'Metrics': 'Subnational-level',
             'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_subnat_level'],
             'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_subnat_level'],
             'NL: MSE': globals()[f'{sample}_mse_nl_only_subnat_level'],
             'NL: R-square': globals()[f'{sample}_r2_nl_only_subnat_level'],
             'Both: MSE': globals()[f'{sample}_mse_both_subnat_level'],
             'Both: R-square': globals()[f'{sample}_r2_both_subnat_level']},
            {'Metrics': 'Subnational-demeaned',
             'MOSAIKS: MSE': globals()[f'{sample}_mse_mosaiks_only_subnat_demeaned'],
             'MOSAIKS: R-square': globals()[f'{sample}_r2_mosaiks_only_subnat_demeaned'],
             'NL: MSE': globals()[f'{sample}_mse_nl_only_subnat_demeaned'],
             'NL: R-square': globals()[f'{sample}_r2_nl_only_subnat_demeaned'],
             'Both: MSE': globals()[f'{sample}_mse_both_subnat_demeaned'],
             'Both: R-square': globals()[f'{sample}_r2_both_subnat_demeaned']}
        ]
        
        fn = os.path.join(c.out_dir, 'metrics', 'global_adm_{}_{}sample_metrics.csv'.format(task, sample))
        with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MOSAIKS: MSE', 'MOSAIKS: R-square', 'NL: MSE', 'NL: R-square', 'Both: MSE', 'Both: R-square'])
            writer.writeheader()
            writer.writerows(rows)

