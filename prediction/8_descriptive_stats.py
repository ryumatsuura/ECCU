## This script implements the prediction exercises

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from mosaiks import transforms
from mosaiks.utils.imports import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression

#########################################
## A) summary stats for validation data 
#########################################

## A-1. summarize population and income distribution

for country in ['brb', 'lca']:
    
    ## define unit
    if country == 'brb':
        units = ['ed', 'mosaiks']
    elif country == 'lca':
        units = ['settle', 'mosaiks']
    
    for unit in units:
        
        for task in ['population', 'income']:
            
            ## skip mosaiks-population combination since it's done in another shapefile
            Y = pd.read_pickle(os.path.join(c.data_dir, 'int', task, '{}_{}_{}.pkl'.format(country, unit, task)))
            if task == 'population':
                globals()[f'{country}_{unit}_{task}_mean'] = Y['ln_pop_density'].mean()
                globals()[f'{country}_{unit}_{task}_var'] = Y['ln_pop_density'].var()        
            elif task == 'income':
                globals()[f'{country}_{unit}_{task}_mean'] = Y['ln_income'].mean()
                globals()[f'{country}_{unit}_{task}_var'] = Y['ln_income'].var() 

rows = [
    {'Descriptives': 'BRB EB',
     'Population Mean': brb_ed_population_mean,
     'Income Mean': brb_ed_income_mean,
     'Population Variance': brb_ed_population_var,
     'Income Variance': brb_ed_income_var},
    {'Descriptives': 'BRB MOSAIKS',
     'Population Mean': brb_mosaiks_population_mean,
     'Income Mean': brb_mosaiks_income_mean,
     'Population Variance': brb_mosaiks_population_var,
     'Income Variance': brb_mosaiks_income_var},
    {'Descriptives': 'LCA Settlement',
     'Population Mean': lca_settle_population_mean,
     'Income Mean': lca_settle_income_mean,
     'Population Variance': lca_settle_population_var,
     'Income Variance': lca_settle_income_var},
    {'Descriptives': 'LCA MOSAIKS',
     'Population Mean': lca_mosaiks_population_mean,
     'Income Mean': lca_mosaiks_income_mean,
     'Population Variance': lca_mosaiks_population_var,
     'Income Variance': lca_mosaiks_income_var}
]

fn = os.path.join(c.out_dir, 'metrics', 'brb_lca_population_income_sumstats.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Descriptives', 'Population Mean', 'Income Mean', 'Population Variance', 'Income Variance'])
    writer.writeheader()
    writer.writerows(rows)

## A-2. correlate population and income

brb_income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_parish_income.pkl'))
lca_2010_income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_district_income.pkl'))
lca_2016_income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_district_median_income.pkl'))
brb_pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'brb_subnat_population.pkl'))
lca_pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'lca_subnat_population.pkl'))

## merge income and population data
brb_merge = pd.merge(brb_income, brb_pop, left_index = True, right_index = True)
lca_2010_merge = pd.merge(lca_2010_income, lca_pop, left_index = True, right_index = True)
lca_2016_merge = pd.merge(lca_2016_income, lca_pop, left_index = True, right_index = True)

## plot income against population
for df in (brb_merge, lca_2010_merge, lca_2016_merge):
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(np.array(df['ln_pop_density']), np.array(df['ln_income']))
    xmin = np.min(np.array(df['ln_pop_density']))
    p1, p0 = np.polyfit(np.array(df['ln_pop_density']), np.array(df['ln_income']), deg = 1)
    newp0 = p0 + xmin * p1
    ax.axline(xy1 = (xmin, newp0), slope = p1, color = 'r', lw = 2)
    ax.set_xlabel('Log Population Density')
    ax.set_ylabel('Log Income per Capita')
    stat = (f"$r$ = {np.corrcoef(df['ln_pop_density'], df['ln_income'])[0][1]:.2f}")
    bbox = dict(boxstyle = 'round', fc = 'blanchedalmond', alpha = 0.5)
    ax.text(0.95, 0.07, stat, fontsize = 12, bbox = bbox, transform = ax.transAxes, horizontalalignment = 'right')
    if any(df.equals(y) for y in [brb_merge]):
        fig.savefig(os.path.join(c.out_dir, 'income', 'brb_income_population_correlates.png'), bbox_inches = 'tight', pad_inches = 0.1)
    elif any(df.equals(y) for y in [lca_2010_merge]):
        fig.savefig(os.path.join(c.out_dir, 'income', 'lca_income_population_correlates.png'), bbox_inches = 'tight', pad_inches = 0.1)
    elif any(df.equals(y) for y in [lca_2016_merge]):
        fig.savefig(os.path.join(c.out_dir, 'income', 'lca_median_income_population_correlates.png'), bbox_inches = 'tight', pad_inches = 0.1)

## A-3. correlate income and predicted HDI values

eccu_hdi = pd.read_pickle(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_hdi_preds.pkl'))

## merge income and predicted HDI data
brb_merge = pd.merge(brb_income, eccu_hdi, left_index = True, right_index = True)
lca_2010_merge = pd.merge(lca_2010_income, eccu_hdi, left_index = True, right_index = True)
lca_2016_merge = pd.merge(lca_2016_income, eccu_hdi, left_index = True, right_index = True)

tasks = ['hdi', 'gni', 'health', 'income', 'ed', 'iwi']

## plot predicted HDI values against income
for task in tasks:
    for df in (brb_merge, lca_2010_merge, lca_2016_merge):  
        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(np.array(df['ln_income']), np.array(df['{}_preds_subnat'.format(task)]))
        xmin = np.min(np.array(df['ln_income']))
        p1, p0 = np.polyfit(np.array(df['ln_income']), np.array(df['{}_preds_subnat'.format(task)]), deg = 1)
        newp0 = p0 + xmin * p1
        ax.axline(xy1 = (xmin, newp0), slope = p1, color = 'r', lw = 2)
        ax.set_xlabel('Log Income per Capita')
        if task == 'hdi' or task == 'gni' or task == 'iwi':
            ax.set_ylabel('Predicted {}'.format(task.upper()))
        elif task == 'health' or task == 'income':
            ax.set_ylabel('Predicted {} Index'.format(task.capitalize()))
        elif task == 'ed':
            ax.set_ylabel('Predicted Education Index')
        stat = (f"$r$ = {np.corrcoef(df['ln_income'], df['{}_preds_subnat'.format(task)])[0][1]:.2f}")
        bbox = dict(boxstyle = 'round', fc = 'blanchedalmond', alpha = 0.5)
        ax.text(0.95, 0.07, stat, fontsize = 12, bbox = bbox, transform = ax.transAxes, horizontalalignment = 'right')
        if any(df.equals(y) for y in [brb_merge]):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'brb_{}_income_correlates.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
        elif any(df.equals(y) for y in [lca_2010_merge]):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'lca_{}_income_correlates.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
        elif any(df.equals(y) for y in [lca_2016_merge]):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'lca_{}_median_income_correlates.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)

    
## B-2. Barbados
"""
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
bslc_income.loc[:, 'income'] = bslc_income[['q10_02a', 'q10_04a']].sum(axis = 1)

## aggregate income and weight to household level
bslc_hhincome = bslc_income[['hhid', 'income']].groupby('hhid').agg(sum)
bslc_hhweight = bslc_income[['hhid', 'weight']].groupby('hhid').agg(sum)

## merge in household identifier and weights
bslc_hh = bslc_hhincome.merge(bslc_hhweight, left_on = 'hhid', right_on = 'hhid').merge(bslc_hhid[['hhid', 'par']], left_on = 'hhid', right_on = 'hhid')

## aggregate income to parish, average income, and standardize it
par_tot_weight = bslc_hh[['par', 'weight']].groupby('par').agg(sum)
agg_income = bslc_hh.income.mul(bslc_hh.weight).groupby(bslc_hh['par']).sum()
df = par_tot_weight.merge(agg_income.rename('agg_income'), left_index = True, right_index = True)
df['avg_income']  = df['agg_income'] / df['weight']
df['income'] = StandardScaler().fit_transform(df[['avg_income']])
"""

## B-3. St. Lucia

## district key for census 
dist_key = pd.DataFrame.from_dict({'LC04': 'ANSE-LA-RAYE', 'LC05': 'CANARIES', 'LC06': 'SOUFRIERE', 'LC07': 'CHOISEUL', 'LC08': 'LABORIE', 'LC09': 'VIEUX-FORT', 
                                   'LC10': 'MICOUD', 'LC11': 'DENNERY', 'LC12': 'GROS-ISLET', 'LC13': 'CASTRIES'}, orient = 'index', columns = ['NAME'])

## load LCA data
lca_2010 = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Saint Lucia Census and Labor Survey', '2010 Census Dataset', 'person_house_merged.dta'), convert_categoricals = False)
lca_2010 = lca_2010[['district', 'ed', 'hh', 'p34_per_num', 'hhincome', 'settlement']]
lca_2010 = lca_2010.drop_duplicates(subset = ['district', 'ed', 'hh', 'p34_per_num'], keep = False)

## fill in missing household income and settlement code - only one non-missing value for each household
lca_2010['hhincome'] = lca_2010['hhincome'].fillna(lca_2010.groupby(['district', 'ed', 'hh'])['hhincome'].transform('mean')).astype(int)
lca_2010['settlement'] = lca_2010['settlement'].fillna(lca_2010.groupby(['district', 'ed', 'hh'])['settlement'].transform('mean')).astype(int)

## collapse down to household level
lca_2010 = lca_2010.merge(lca_2010.groupby(['district', 'ed', 'hh']).size().to_frame('pop').reset_index(), left_on = ['district', 'ed', 'hh'], right_on = ['district', 'ed', 'hh'])
lca_2010_hh = lca_2010.drop(columns = ['p34_per_num']).drop_duplicates()

## create admin code - aggregate castries to district 13
lca_2010_hh['adm2code'] = lca_2010_hh['settlement'].apply(lambda x: '{0:0>9}'.format(x))
lca_2010_hh['adm1code'] = 'LC' + lca_2010_hh['adm2code'].str[0:2]
lca_2010_hh.loc[lca_2010_hh['adm1code'].isin(['LC01', 'LC02', 'LC03']), ['adm1code']] = 'LC13'

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

## merge HDI prediction with census data
cols = ['Country', 'Name', 'hdi_preds_subnat', 'hdi_preds_subnat_demeaned', 'gni_preds_subnat', 'gni_preds_subnat_demeaned', 'health_preds_subnat', 'health_preds_subnat_demeaned',
        'income_preds_subnat', 'income_preds_subnat_demeaned', 'ed_preds_subnat', 'ed_preds_subnat_demeaned']
merged = pd.merge(eccu_subnat_preds[cols], lca_2010_dist[['Country', 'District', 'income']], left_on = ['Country', 'Name'], right_on = ['Country', 'District'])

## plot HDI/gni against income 
for task in ['hdi_preds_subnat', 'hdi_preds_subnat_demeaned', 'gni_preds_subnat', 'gni_preds_subnat_demeaned']:
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(np.array(merged['income']), np.array(merged[task]))
    xmin = np.min(np.array(merged['income']))
    p1, p0 = np.polyfit(np.array(merged['income']), np.array(merged[task]), deg = 1)
    newp0 = p0 + xmin * p1
    ax.axline(xy1 = (xmin, newp0), slope = p1, color = 'r', lw = 2)
    ax.set_xlabel('Standardized Income per Capita from Census')
    if task == 'hdi_preds_subnat':
        ax.set_ylabel('Predicted HDI')
    elif task == 'hdi_preds_subnat_demeaned':
        ax.set_ylabel('Predicted demeaned HDI')
    elif task == 'gni_preds_subnat':
        ax.set_ylabel('Predicted GNI')
    elif task == 'gni_preds_subnat_demeaned':
        ax.set_ylabel('Predicted demeaned GNI')
    stat = (f"$r$ = {np.corrcoef(merged['income'], merged[task])[0][1]:.2f}")
    bbox = dict(boxstyle = 'round', fc = 'blanchedalmond', alpha = 0.5)
    ax.text(0.95, 0.07, stat, fontsize = 12, bbox = bbox, transform = ax.transAxes, horizontalalignment = 'right')
    fig.savefig(os.path.join(c.out_dir, 'income', 'lca_income_{}_pred.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)


