## This script trains the prediction model based on
## St. Lucia census data

## set preambles
subset_n = slice(None)
subset_feat = slice(None)

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from pathlib import Path
from mosaiks import transforms
from mosaiks.utils.imports import *
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

lambdas = lambdas_single = c.ml_model['global_lambdas']
solver = solve.ridge_regression
solver_kwargs = {'return_preds':True, 'svd_solve':False}

###################################
## A) load label and MOSAIKS data
###################################

## load LCA data
lca_2010 = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Saint Lucia Census and Labor Survey', '2010 Census Dataset', 'person_house_merged.dta'), convert_categoricals = False)
lca_2010 = lca_2010[['district', 'ed', 'hh', 'p34_per_num', 'hhincome', 'settlement']]
lca_2010 = lca_2010.drop_duplicates(subset = ['district', 'ed', 'hh', 'p34_per_num'], keep = False)

## fill in missing household income and settlement code - only one non-missing value for each household
lca_2010['hhincome'] = lca_2010['hhincome'].fillna(lca_2010.groupby(['district', 'ed', 'hh'])['hhincome'].transform('mean')).astype(int)
lca_2010['settlement'] = lca_2010['settlement'].fillna(lca_2010.groupby(['district', 'ed', 'hh'])['settlement'].transform('mean')).astype(int)

## collapse down to household level
lca_2010_hh = lca_2010.drop(columns = ['p34_per_num']).drop_duplicates()

## create admin code - aggregate castries to district 13
lca_2010_hh['adm2code'] = lca_2010_hh['settlement'].apply(lambda x: '{0:0>9}'.format(x))
lca_2010_hh['adm1code'] = 'LC' + lca_2010_hh['adm2code'].str[0:2]
lca_2010_hh.loc[lca_2010_hh['adm1code'].isin(['LC01', 'LC02', 'LC03']), ['adm1code']] = 'LC13'

## aggregate household income to settlement level and standardize it
lca_2010_settle = lca_2010_hh.groupby(['adm1code', 'settlement'])['hhincome'].sum()
lca_2010_settle = lca_2010_settle.to_frame('agg_income').reset_index().merge(lca_2010_hh.groupby(['adm1code', 'settlement']).size().to_frame('hh_num').reset_index(), left_on = ['adm1code', 'settlement'], right_on = ['adm1code', 'settlement'])
lca_2010_settle['avg_income'] = lca_2010_settle['agg_income'] / lca_2010_settle['hh_num']
lca_2010_settle['income'] = StandardScaler().fit_transform(lca_2010_settle[['avg_income']])

## load demeaned mosaiks feat
mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_lca_settle_demeaned.csv'), index_col = 0)

## select enumeration districts that match with MOSAIKS data
Y = lca_2010_settle.merge(mosaiks_feat[['ADM1_Code', 'Settle_Code']], left_on = ['adm1code', 'settlement'], right_on = ['ADM1_Code', 'Settle_Code'])
X = mosaiks_feat.merge(Y[['ADM1_Code', 'Settle_Code']], left_on = ['ADM1_Code', 'Settle_Code'], right_on = ['ADM1_Code', 'Settle_Code'])

###################
## B) train model 
###################

## convert data to numpy array
Y_np = np.array(Y['income'])
X_np = X.iloc[:, 2:4002].to_numpy()

## set the bounds
mins = Y_np.min(axis = 0)
maxs = Y_np.max(axis = 0)
solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T

## split the data into training vs testing sets
X_train, X_test, Y_train, Y_test, idxs_train, idsx_test = parse.split_data_train_test(
    X_np, Y_np, frac_test = c.ml_model['test_set_frac'], return_idxs = True
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
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'lca_settle_income.csv'), wts, delimiter = ',')

###############
## C) predict
###############

# wts = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'lca_settle_income.csv'), delimiter = ',')

## C-1. parish level for ECCU countries

eccu_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_nat.csv'), index_col = 0)
eccu_subnat_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat.csv'), index_col = 0)
eccu_subnat_demean_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat_demeaned.csv'), index_col = 0)

## initialize dataframe
eccu_subnat_preds = pd.DataFrame([])

## add country name and subnational unit name
eccu_subnat_preds['Country'] = eccu_subnat_demean_feat['Country'].values.tolist()
eccu_subnat_preds['Name'] = eccu_subnat_demean_feat['NAME_1']

## predict 
ypreds = np.dot(eccu_subnat_demean_feat.iloc[:, 2:4002], wts)
    
## bound the prediction
ypreds[ypreds < mins] = mins
ypreds[ypreds > maxs] = maxs

## store predicted values
eccu_subnat_preds['income_preds'] = ypreds.tolist()

##############################
## D) clean LCA labor survey
##############################

## D-1. clean ground truth data

## store max income from 2010 LCA data
max_income = np.max(lca_2010.hhincome)

## district key for census
##dist_key = {1: 'Castries City', 2: 'Castries Suburban', 3: 'Castries Rural', 4: 'Anse La Raye', 5: 'Canaries', 6: 'Soufriere', 
##            7: 'Choiseul', 8: 'Laborie', 9: 'Vieux-Fort', 10: 'Micoud', 11: 'Dennery', 12: 'Gros Islet'}
dist_key = pd.DataFrame.from_dict({'LC04': 'ANSE-LA-RAYE', 'LC05': 'CANARIES', 'LC06': 'SOUFRIERE', 'LC07': 'CHOISEUL', 'LC08': 'LABORIE', 'LC09': 'VIEUX-FORT', 
                                   'LC10': 'MICOUD', 'LC11': 'DENNERY', 'LC12': 'GROS-ISLET', 'LC13': 'CASTRIES'}, orient = 'index', columns = ['NAME'])

## load LCA data
lfs_2016 = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Saint Lucia Census and Labor Survey', 'LFS', 'LCA_2016.dta'))
lfs_2016 = lfs_2016[['district', 'ed', 'wt', 'income']]

## recreate district variable
lfs_2016['NAME'] = lfs_2016.district.to_list()

## I can separate ANSE-LA-RAYE and CANARIES districts based on ed id's
## ed = 5200-6205 in Anse La Raye district / 6301-6504 in Canaries district
lfs_2016.loc[(lfs_2016['ed'] >= 5200) & (lfs_2016['ed'] <= 6205), 'NAME'] = 'ANSE-LA-RAYE'
lfs_2016.loc[(lfs_2016['ed'] >= 6301) & (lfs_2016['ed'] <= 6504), 'NAME'] = 'CANARIES'
lfs_2016.loc[(lfs_2016['district'] == 'CASTRIES CITY') | (lfs_2016['district'] == 'CASTRIES RURAL'), 'NAME'] = 'CASTRIES'
lfs_2016 = lfs_2016.drop(columns = ['district'])

## recreate country and district columns
lfs_2016['Country'] = 'LCA'
lfs_2016['District'] = lfs_2016.NAME.str.capitalize()
lfs_2016.loc[lfs_2016['District'] == 'Vieux-fort', 'District'] = 'Vieux Fort'
lfs_2016.loc[lfs_2016['District'] == 'Anse-la-raye', 'District'] = 'Anse-la-Raye'
lfs_2016.loc[lfs_2016['District'] == 'Gros-islet', 'District'] = 'Gros Islet'
lfs_2016.loc[lfs_2016['District'] == 'Soufriere', 'District'] = 'Soufrière'

## create income variable for the upper bound
lfs_2016['income_ub'] = 0
lfs_2016.loc[(lfs_2016['income'] == 0), 'income_ub'] = 100
lfs_2016.loc[(lfs_2016['income'] == 100), 'income_ub'] = 300
lfs_2016.loc[(lfs_2016['income'] == 300), 'income_ub'] = 600
lfs_2016.loc[(lfs_2016['income'] == 600), 'income_ub'] = 1000
lfs_2016.loc[(lfs_2016['income'] == 1000), 'income_ub'] = 1600
lfs_2016.loc[(lfs_2016['income'] == 1600), 'income_ub'] = 3000
lfs_2016.loc[(lfs_2016['income'] == 3000), 'income_ub'] = 5000
lfs_2016.loc[(lfs_2016['income'] == 5000), 'income_ub'] = 7500
lfs_2016.loc[(lfs_2016['income'] == 7500), 'income_ub'] = max_income

## create income variable for the midpoint
lfs_2016['income_mid'] = ((lfs_2016['income'] + lfs_2016['income_ub']) / 2).astype(int)

## aggregate income to district level - use lower bound
dist_tot_weight = lfs_2016[['Country', 'District', 'wt']].groupby(['Country', 'District']).agg(sum)
agg_income_lb = lfs_2016.income.mul(lfs_2016.wt).groupby(lfs_2016['District']).sum()
agg_income_ub = lfs_2016.income_ub.mul(lfs_2016.wt).groupby(lfs_2016['District']).sum()
agg_income_mid = lfs_2016.income_mid.mul(lfs_2016.wt).groupby(lfs_2016['District']).sum()
df = dist_tot_weight.merge(agg_income_lb.rename('agg_income_lb'), left_index = True, right_index = True)
df = df.merge(agg_income_ub.rename('agg_income_ub'), left_index = True, right_index = True)
df = df.merge(agg_income_mid.rename('agg_income_mid'), left_index = True, right_index = True)
for income in ['lb', 'ub', 'mid']:
    df['avg_income_{}'.format(income)]  = df['agg_income_{}'.format(income)] / df['wt']
    df['income_{}'.format(income)] = StandardScaler().fit_transform(df[['avg_income_{}'.format(income)]])

## match predicted value and ground-truth at district level
merged_dist = eccu_subnat_preds.merge(df[['income_lb', 'income_ub', 'income_mid']], left_on = ['Country', 'Name'], right_index = True)

## D-2. comapre ground-truth against predicted values

for col in ['income_lb', 'income_ub', 'income_mid']:
    
    ## plot prediction against ground truth
    plt.clf()
    tot_min = np.min([np.min(np.array(merged_dist['income_preds'])), np.min(np.array(merged_dist[col]))])
    tot_max = np.max([np.max(np.array(merged_dist['income_preds'])), np.max(np.array(merged_dist[col]))])
    fig, ax = plt.subplots()
    ax.scatter(np.array(merged_dist[col]), np.array(merged_dist['income_preds']))
    
    ## add 45 degree line and country names
    plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
    
    ## add axis title
    ax.set_xlabel('True Demeaned Income')
    ax.set_ylabel('Predicted Demeaned Income')
    
    ## output the graph
    fig.savefig(os.path.join(c.out_dir, 'income', 'lca_{}_lca_dist.png'.format(col)), bbox_inches = 'tight', pad_inches = 0.1)

#########################
## E) descriptive stats 
#########################

## E-1. compare census and survey data and population density 

## aggregate household income to settlement level and standardize it
lca_2010_dist = lca_2010_hh.groupby(['adm1code'])['hhincome'].sum()
lca_2010_dist = lca_2010_dist.to_frame('agg_income').reset_index().merge(lca_2010_hh.groupby(['adm1code']).size().to_frame('hh_num').reset_index(), left_on = ['adm1code'], right_on = ['adm1code'])
lca_2010_dist['avg_income'] = lca_2010_dist['agg_income'] / lca_2010_dist['hh_num']
lca_2010_dist['income'] = StandardScaler().fit_transform(lca_2010_dist[['avg_income']])

## merge in district name
lca_2010_dist['Country'] = 'LCA'
lca_2010_dist = lca_2010_dist.merge(dist_key, left_on = ['adm1code'], right_index = True)
lca_2010_dist['District'] = lca_2010_dist.NAME.str.capitalize()
lca_2010_dist.loc[lca_2010_dist['District'] == 'Vieux-fort', 'District'] = 'Vieux Fort'
lca_2010_dist.loc[lca_2010_dist['District'] == 'Anse-la-raye', 'District'] = 'Anse-la-Raye'
lca_2010_dist.loc[lca_2010_dist['District'] == 'Gros-islet', 'District'] = 'Gros Islet'
lca_2010_dist.loc[lca_2010_dist['District'] == 'Soufriere', 'District'] = 'Soufrière'

## merge 2010 census with 2016 labor force survey
merged = lca_2010_dist.merge(df, left_on = ['Country', 'District'], right_index = True)

## merge with population density
lca_pop_density = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_lca_subnat.csv'), index_col = 0)
merged = merged.merge(lca_pop_density[['Name', 'Population']], left_on = 'District', right_on = 'Name')

## plot income against population density
for income in ['income', 'income_lb', 'income_ub', 'income_mid']:
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(np.array(merged['Population']), np.array(merged[income]))
    xmin = np.min(np.array(merged['Population']))
    p1, p0 = np.polyfit(np.array(merged['Population']), np.array(merged[income]), deg = 1)
    newp0 = p0 + xmin * p1
    ax.axline(xy1 = (xmin, newp0), slope = p1, color = 'r', lw = 2)
    ax.set_xlabel('Log Population Density')
    if income == 'income':
        ax.set_ylabel('Standardized Income per Capita from Census')
    else:
        ax.set_ylabel('Standardized Income per Capita from LFS')
    stat = (f"$r$ = {np.corrcoef(merged['Population'], merged[income])[0][1]:.2f}")
    bbox = dict(boxstyle = 'round', fc = 'blanchedalmond', alpha = 0.5)
    ax.text(0.95, 0.07, stat, fontsize = 12, bbox = bbox, transform = ax.transAxes, horizontalalignment = 'right')
    if income == 'income':
        fig.savefig(os.path.join(c.out_dir, 'population', 'lca_population_income_census.png'), bbox_inches = 'tight', pad_inches = 0.1)
    else:
        fig.savefig(os.path.join(c.out_dir, 'population', 'lca_population_{}_lfs.png'.format(income)), bbox_inches = 'tight', pad_inches = 0.1)

## store mean and variances into one csv file
rows = [
    {'Descriptives': 'LCA Census',
     'Population Mean': merged['Population'].mean(),
     'Income Mean': merged['income'].mean(),
     'Population Var.': merged['Population'].var(),
     'Income Var.': merged['income'].var()},
    {'Descriptives': 'LCA LFS',
     'Population Mean': merged['Population'].mean(),
     'Income Mean': merged['income_mid'].mean(),
     'Population Var.': merged['Population'].var(),
     'Income Var.': merged['income_mid'].var()}
]

fn = os.path.join(c.out_dir, 'metrics', 'lca_summary_stats.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Descriptives', 'Population Mean', 'Income Mean', 'Population Var.', 'Income Var.'])
    writer.writeheader()
    writer.writerows(rows)

