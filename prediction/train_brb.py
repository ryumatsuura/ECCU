## This script trains the prediction model based on
## Barbados data

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

## load Barbados data
bslc_hhid = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Barbados-Survey-of-Living-Conditions-2016', 'Data BSLC2016', 'RT001_Public.dta'))
bslc_income = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Barbados-Survey-of-Living-Conditions-2016', 'Data BSLC2016', 'RT002_Public.dta'))

## keep variables of interest
bslc_hhid = bslc_hhid[['hhid', 'par', 'psu']]
bslc_income = bslc_income[['hhid', 'id', 'weight', 'q10_02a', 'q10_04a', 'q10_06', 'q10_07', 'q10_08', 'q10_09', 'q10_10', 'q10_11', 'q10_12', 'q10_13', 'q10_14', 'q10_15',
                           'q10_16', 'q10_17', 'q10_18', 'q10_19', 'q10_20', 'q10_21']]

## compute income in last month/year
bslc_income.loc[:, 'income_last_month'] = bslc_income[['q10_02a', 'q10_04a', 'q10_06', 'q10_07', 'q10_08', 'q10_09', 'q10_10', 'q10_11', 'q10_12', 'q10_13', 'q10_14', 'q10_15']].sum(axis = 1)
bslc_income.loc[:, 'income_last_year']  = bslc_income[['q10_16', 'q10_17', 'q10_18', 'q10_19', 'q10_20', 'q10_21']].sum(axis = 1)

## compute annual income - since income will be standardized it doesn't matter if we choose annual vs monthly
##bslc_income.loc[:, 'income']  = (bslc_income['income_last_month'] * 12) + bslc_income['income_last_year']
bslc_income.loc[:, 'income'] = bslc_income[['q10_02a', 'q10_04a']].sum(axis = 1)

## aggregate income and weight to household level
bslc_hhincome = bslc_income[['hhid', 'income']].groupby('hhid').agg(sum)
bslc_hhweight = bslc_income[['hhid', 'weight']].groupby('hhid').agg(sum)

## merge in household identifier and weights
bslc_hh = bslc_hhincome.merge(bslc_hhweight, left_on = 'hhid', right_on = 'hhid').merge(bslc_hhid[['hhid', 'psu']], left_on = 'hhid', right_on = 'hhid')

## aggregate income to enumeration districts, average income, and standardize it
psu_tot_weight = bslc_hh[['psu', 'weight']].groupby('psu').agg(sum)
agg_income = bslc_hh.income.mul(bslc_hh.weight).groupby(bslc_hh['psu']).sum()
df = psu_tot_weight.merge(agg_income.rename('agg_income'), left_index = True, right_index = True)
df['avg_income']  = df['agg_income'] / df['weight']
df['income'] = StandardScaler().fit_transform(df[['avg_income']])

## load MOSAIKS data
mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_brb_ed_demeaned.csv'), sep = ',', index_col = 0)

## select enumeration districts that match with MOSAIKS data
Y = df.merge(mosaiks_feat[['psu']], left_index = True, right_on = 'psu')

###################
## B) train model 
###################

## B-1. Barbados data - doesn't work well

## convert data to numpy array
Y_np = np.array(Y['income'])
mosaiks_feat_np = mosaiks_feat.iloc[:, 1:4001].to_numpy()

## set the bounds
mins = Y_np.min(axis = 0)
maxs = Y_np.max(axis = 0)
solver_kwargs['clip_bounds'] = np.vstack((mins, maxs)).T

## split the data into training vs testing sets
X_train, X_test, Y_train, Y_test, idxs_train, idsx_test = parse.split_data_train_test(
    mosaiks_feat_np, Y_np, frac_test = c.ml_model['test_set_frac'], return_idxs = True
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
np.savetxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_ed_income.csv'), wts, delimiter = ',')

###############
## C) predict
###############

# wts = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_ed_income.csv'), delimiter = ',')

## C-1. parish level for ECCU countries

eccu_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_nat.csv'), index_col = 0)
eccu_subnat_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat.csv'), index_col = 0)
eccu_subnat_demean_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat_demeaned.csv'), index_col = 0)

## initialize dataframe
eccu_subnat_preds = pd.DataFrame([])

## add country name and subnational unit name
eccu_subnat_preds['Country'] = eccu_subnat_feat['Country'].values.tolist()
eccu_subnat_preds['Name'] = eccu_subnat_feat[['NAME_1']]

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

## load LCA data
lca_2010 = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Saint Lucia Census and Labor Survey', '2010 Census Dataset', 'person_house_merged.dta'), convert_categoricals = False)

## store max income from 2010 LCA data
max_income = np.max(lca_2010.hhincome)

## district key for census
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

## aggregate income to district level
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
    fig.savefig(os.path.join(c.out_dir, 'income', 'brb_{}_lca_dist.png'.format(col)), bbox_inches = 'tight', pad_inches = 0.1)

#########################
## E) descriptive stats 
#########################

## E-1. plot income against population density

## aggregate income to enumeration districts, average income, and standardize it
bslc_hh = bslc_hhincome.merge(bslc_hhweight, left_on = 'hhid', right_on = 'hhid').merge(bslc_hhid[['hhid', 'par']], left_on = 'hhid', right_on = 'hhid')
par_tot_weight = bslc_hh[['par', 'weight']].groupby('par').agg(sum)
agg_income = bslc_hh.income.mul(bslc_hh.weight).groupby(bslc_hh['par']).sum()
df = par_tot_weight.merge(agg_income.rename('agg_income'), left_index = True, right_index = True)
df['avg_income']  = df['agg_income'] / df['weight']
df['income'] = StandardScaler().fit_transform(df[['avg_income']])

## merge with population density
brb_pop_density = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_brb_subnat.csv'), index_col = 0)
brb_pop_density = brb_pop_density.replace({'Name': 'Saint'}, {'Name': 'St.'}, regex = True)
df = df.merge(brb_pop_density[['Name', 'Population']], left_index = True, right_on = 'Name')

## plot income against population density
plt.clf()
fig, ax = plt.subplots()
ax.scatter(np.array(df['Population']), np.array(df['income']))
xmin = np.min(np.array(df['Population']))
p1, p0 = np.polyfit(np.array(df['Population']), np.array(df['income']), deg = 1)
newp0 = p0 + xmin * p1
ax.axline(xy1 = (xmin, newp0), slope = p1, color = 'r', lw = 2)
ax.set_xlabel('Log Population Density')
ax.set_ylabel('Standardized Income per Capita')
stat = (f"$r$ = {np.corrcoef(df['Population'], df['income'])[0][1]:.2f}")
bbox = dict(boxstyle = 'round', fc = 'blanchedalmond', alpha = 0.5)
ax.text(0.95, 0.07, stat, fontsize = 12, bbox = bbox, transform = ax.transAxes, horizontalalignment = 'right')
fig.savefig(os.path.join(c.out_dir, 'population', 'brb_population_income.png'), bbox_inches = 'tight', pad_inches = 0.1)

## store mean and variances into one csv file
rows = [
    {'Descriptives': 'BRB LCS',
     'Population Mean': df['Population'].mean(),
     'Income Mean': df['income'].mean(),
     'Population Var.': df['Population'].var(),
     'Income Var.': df['income'].var()}
]

fn = os.path.join(c.out_dir, 'metrics', 'brb_summary_stats.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Descriptives', 'Population Mean', 'Income Mean', 'Population Var.', 'Income Var.'])
    writer.writeheader()
    writer.writerows(rows)

