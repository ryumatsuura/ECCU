## This script cleans surveys data

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from mosaiks import transforms
from mosaiks.utils.imports import *
from sklearn.preprocessing import StandardScaler

## create folder if not exists
if not os.path.exists(os.path.join(c.data_dir, 'int', 'income')):
    os.makedirs(os.path.join(c.data_dir, 'int', 'income'))

#############################
## A) clean Barbados survey
#############################

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
bslc_income.loc[:, 'income']  = (bslc_income['income_last_month'] * 12) + bslc_income['income_last_year']
##bslc_income.loc[:, 'income'] = bslc_income[['q10_02a', 'q10_04a']].sum(axis = 1)

## aggregate income and weight to household level
bslc_hhincome = bslc_income[['hhid', 'income']].groupby('hhid').agg(sum)
bslc_hhweight = bslc_income[['hhid', 'weight']].groupby('hhid').agg(sum)

## merge in household identifier and weights
bslc_hh = bslc_hhincome.merge(bslc_hhweight, left_on = 'hhid', right_on = 'hhid').merge(bslc_hhid[['hhid', 'psu']], left_on = 'hhid', right_on = 'hhid')

## aggregate income to enumeration districts, average income, and standardize it
psu_tot_weight = bslc_hh[['psu', 'weight']].groupby('psu').agg(sum)
agg_income = bslc_hh.income.mul(bslc_hh.weight).groupby(bslc_hh['psu']).sum()
income = psu_tot_weight.merge(agg_income.rename('agg_income'), left_index = True, right_index = True)
income['avg_income']  = income['agg_income'] / income['weight']
income['income'] = StandardScaler().fit_transform(income[['avg_income']])

## save income for future use
income.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_ed_income.pkl'))

##############################
## B) clean St. Lucia census
##############################

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
lca_2010_settle = lca_2010_hh.groupby(['adm1code', 'settlement'])[['hhincome', 'pop']].sum()
lca_2010_settle = lca_2010_settle.reset_index().merge(lca_2010_hh.groupby(['adm1code', 'settlement']).size().to_frame('hh_num').reset_index(), left_on = ['adm1code', 'settlement'], right_on = ['adm1code', 'settlement'])
lca_2010_settle['avg_income'] = lca_2010_settle['hhincome'] / lca_2010_settle['hh_num']
lca_2010_settle['income'] = StandardScaler().fit_transform(lca_2010_settle[['avg_income']])
lca_2010_settle = lca_2010_settle.set_index(lca_2010_settle['adm1code'] + lca_2010_settle['settlement'].apply(lambda x: '{0:0>9}'.format(x))).drop(columns = ['adm1code', 'settlement'])

## save income for future use
lca_2010_settle.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_settle_income.pkl'))

