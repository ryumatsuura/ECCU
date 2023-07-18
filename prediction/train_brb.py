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
bslc_hhid = bslc_hhid[['hhid', 'psu']]
bslc_income = bslc_income[['hhid', 'id', 'weight', 'q10_02a', 'q10_04a', 'q10_06', 'q10_07', 'q10_08', 'q10_09', 'q10_10', 'q10_11', 'q10_12', 'q10_13', 'q10_14', 'q10_15',
                           'q10_16', 'q10_17', 'q10_18', 'q10_19', 'q10_20', 'q10_21']]

## compute income in last month/year
bslc_income.loc[:, 'income_last_month'] = bslc_income[['q10_02a', 'q10_04a', 'q10_06', 'q10_07', 'q10_08', 'q10_09', 'q10_10', 'q10_11', 'q10_12', 'q10_13', 'q10_14', 'q10_15']].sum(axis = 1)
bslc_income.loc[:, 'income_last_year']  = bslc_income[['q10_16', 'q10_17', 'q10_18', 'q10_19', 'q10_20', 'q10_21']].sum(axis = 1)

## compute annual income - since income will be standardized it doesn't matter if we choose annual vs monthly
bslc_income.loc[:, 'income']  = (bslc_income['income_last_month'] * 12) + bslc_income['income_last_year']
##bslc_income.loc[:, 'income'] = bslc_income[['q10_02a', 'q10_04a']].sum(axis = 1)

## aggregate income and weight to household level
bslc_hhincome = bslc_income[['hhid', 'income']].groupby('hhid').agg(sum)
bslc_hhweight = bslc_income[['hhid', 'weight']].groupby('hhid').agg(sum)

## merge in household identifier and weights
bslc_hh = bslc_hhincome.merge(bslc_hhweight, left_on = 'hhid', right_on = 'hhid').merge(bslc_hhid, left_on = 'hhid', right_on = 'hhid')

## aggregate income to enumeration districts, average income, and standardize it
psu_tot_weight = bslc_hh[['psu', 'weight']].groupby('psu').agg(sum)
agg_income = bslc_hh.income.mul(bslc_hh.weight).groupby(bslc_hh['psu']).sum()
df = psu_tot_weight.merge(agg_income.rename('agg_income'), left_index = True, right_index = True)
df['avg_income']  = df['agg_income'] / df['weight']
df['income'] = StandardScaler().fit_transform(df[['avg_income']])

##brb_pop_density = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_brb_ed.csv'), index_col = 0)
##df = df.merge(brb_pop_density[['psu', 'Population']], left_index = True, right_on = 'psu')
##df['income'].corr(df['Population'])
##df['agg_income'].corr(df['Population'])

## load MOSAIKS data
##mosaiks_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_brb_ed.csv'), sep = ',', index_col = 0)
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

# wts = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'lca_settle_income.csv'), delimiter = ',')

## C-1. parish level for ECCU countries

eccu_subnat_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat.csv'), index_col = 0)

## initialize dataframe
eccu_subnat_preds = pd.DataFrame([])

## add country name and subnational unit name
eccu_subnat_preds['Country'] = eccu_subnat_feat['Country'].values.tolist()
eccu_subnat_preds['Name'] = eccu_subnat_feat[['NAME_1']]

## predict 
ypreds = np.dot(eccu_subnat_feat.iloc[:, 2:4002], wts)
    
## bound the prediction
ypreds[ypreds < mins] = mins
ypreds[ypreds > maxs] = maxs

## store predicted values
eccu_subnat_preds['income_preds'] = ypreds.tolist()

## C-2. district level for St Lucia

lca_settle_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_lca_settle.csv'), index_col = 0)

## initialize dataframe
lca_settle_preds = pd.DataFrame([])

## add country name and subnational unit name
lca_settle_preds['ADM1_Code'] = lca_settle_feat['ADM1_Code'].values.tolist()
lca_settle_preds['Settle_Code'] = lca_settle_feat[['Settle_Code']]

## predict 
ypreds = np.dot(lca_settle_feat.iloc[:, 2:4002], wts)
    
## bound the prediction
ypreds[ypreds < mins] = mins
ypreds[ypreds > maxs] = maxs

## store predicted values
lca_settle_preds['income_preds'] = ypreds.tolist()

##############################
## D) clean LCA labor survey
##############################

## D-1. clean ground truth data

## district key for census
dist_key = {'LC04': 'Anse-la-Raye', 'LC05': 'Canaries', 'LC06': 'Soufrière', 'LC07': 'Choiseul', 'LC08': 'Laborie', 'LC09': 'Vieux Fort', 
            'LC10': 'Micoud', 'LC11': 'Dennery', 'LC12': 'Gros Islet', 'LC13': 'Castries'}

## load LCA data
lca_2010 = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Saint Lucia Census and Labor Survey', '2010 Census Dataset', 'person_house_merged.dta'), convert_categoricals = False)

## keep variables of interest
lca_2010 = lca_2010[['district', 'ed', 'hh', 'p34_per_num', 'hhincome', 'settlement']]

## I can separate ANSE-LA-RAYE and CANARIES districts based on ed id's
## ed = 5200-6205 in Anse La Raye district / 6301-6504 in Canaries district

## drop duplicates on district-enumeration block-household-person identifiers
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

## match predicted value and ground-truth at settlement level
merged_settle = lca_settle_preds.merge(lca_2010_settle[['adm1code', 'settlement', 'income']], left_on = ['ADM1_Code', 'Settle_Code'], right_on = ['adm1code', 'settlement'])

## aggregate household income to district level and standardize it
lca_2010_dist = lca_2010_hh.groupby(['adm1code'])['hhincome'].sum()
lca_2010_dist = lca_2010_dist.to_frame('agg_income').reset_index().merge(lca_2010_hh.groupby(['adm1code']).size().to_frame('hh_num').reset_index(), left_on = ['adm1code'], right_on = ['adm1code'])
lca_2010_dist['avg_income'] = lca_2010_dist['agg_income'] / lca_2010_dist['hh_num']
lca_2010_dist['income'] = StandardScaler().fit_transform(lca_2010_dist[['avg_income']])
lca_2010_dist = lca_2010_dist.merge(pd.DataFrame.from_dict(dist_key, orient = 'index', columns = ['Name']), left_on = 'adm1code', right_index = True)
lca_2010_dist['Country'] = 'LCA'

## match predicted value and ground-truth at district level
merged_dist = eccu_subnat_preds.merge(lca_2010_dist[['Country', 'Name', 'income']], left_on = ['Country', 'Name'], right_on = ['Country', 'Name'])

## D-2. comapre ground-truth against predicted values

for df in (merged_settle, merged_dist):
    
    ## plot prediction against ground truth
    plt.clf()
    tot_min = np.min([np.min(np.array(df['income_preds'])), np.min(np.array(df['income']))])
    tot_max = np.max([np.max(np.array(df['income_preds'])), np.max(np.array(df['income']))])
    fig, ax = plt.subplots()
    ax.scatter(np.array(df['income']), np.array(df['income_preds']))
    
    ## add 45 degree line and country names
    plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
    
    ## add axis title
    ax.set_xlabel('True Demeaned Income')
    ax.set_ylabel('Predicted Demeaned Income')
    
    ## output the graph
    if any(df.equals(y) for y in [merged_settle]):
        fig.savefig(os.path.join(c.out_dir, 'income', 'lca_income_settle.png'), bbox_inches = 'tight', pad_inches = 0.1)
    elif any(df.equals(y) for y in [merged_dist]):
        fig.savefig(os.path.join(c.out_dir, 'income', 'lca_income_dist.png'), bbox_inches = 'tight', pad_inches = 0.1)

