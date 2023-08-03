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

## define function for weighted average
def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

## specify outcome variables
tasks = ['hdi', 'gni', 'health', 'income', 'ed']

## load HDI measures
hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_adm0_clean.p'))
hdi_subnat = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_clean.p'))

## rename columns 
hdi = hdi.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})
hdi_subnat = hdi_subnat.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})

for task in tasks:
    avg = hdi_subnat.groupby('ISO_Code').apply(w_avg, task, 'Population size in millions')
    hdi_subnat = hdi_subnat.reset_index().merge(avg.rename('avg'), left_on = 'ISO_Code', right_on = 'ISO_Code').set_index('GDLCODE')
    hdi_subnat['demeaned_{}'.format(task)] = hdi_subnat[task] - hdi_subnat['avg']
    hdi_subnat = hdi_subnat.drop(columns = ['avg'])

## create folder if not exists
if not os.path.exists(os.path.join(c.out_dir, 'hdi')):
    os.makedirs(os.path.join(c.out_dir, 'hdi'))

###############
## A) predict
###############

## A-1. load aggregated MOSAIKS features

eccu_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_nat_mosaiks_features.pkl'))
eccu_subnat_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_subnat_mosaiks_features.pkl'))
eccu_subnat_demean_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_subnat_mosaiks_features_demeaned.pkl'))
eccu_mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_mosaiks_mosaiks_features.pkl'))
eccu_mosaiks_demean_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_mosaiks_mosaiks_features_demeaned.pkl'))

## A-2. load aggregated NL features

## store the countries to loop over
countries = eccu_feat.index.values

## load national and subnational-level data
eccu_nl = pd.DataFrame([])
eccu_subnat_nl = pd.DataFrame([])
eccu_subnat_demean_nl = pd.DataFrame([])
eccu_mosaiks_nl = pd.DataFrame([])
eccu_mosaiks_demean_nl = pd.DataFrame([])

for x in ['nat', 'subnat', 'mosaiks']:
    for country in countries:
        nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_{}_nl.pkl'.format(country.lower(), x)))
        if x == 'nat':
            eccu_nl = pd.concat([eccu_nl, nl])
        elif x == 'subnat':
            eccu_subnat_nl = pd.concat([eccu_subnat_nl, nl])
            nl_demean = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_{}_nl_demeaned.pkl'.format(country.lower(), x)))
            eccu_subnat_demean_nl = pd.concat([eccu_subnat_demean_nl, nl_demean])
        elif x == 'mosaiks':
            eccu_mosaiks_nl = pd.concat([eccu_mosaiks_nl, nl])
            nl_demean = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_{}_nl_demeaned.pkl'.format(country.lower(), x)))
            eccu_mosaiks_demean_nl = pd.concat([eccu_mosaiks_demean_nl, nl_demean])

## merge MOSAIKS and NL features
X = pd.merge(eccu_feat, eccu_nl, left_index = True, right_index = True)
X_subnat = pd.merge(eccu_subnat_feat, eccu_subnat_nl, left_index = True, right_index = True)
X_subnat_demean = pd.merge(eccu_subnat_demean_feat, eccu_subnat_demean_nl, left_index = True, right_index = True)
X_mosaiks = pd.merge(eccu_mosaiks_feat, eccu_mosaiks_nl, left_index = True, right_index = True)
X_mosaiks_demean = pd.merge(eccu_mosaiks_demean_feat, eccu_mosaiks_demean_nl, left_index = True, right_index = True)

## A-3. prediction!

for task in tasks:
    
    ## set upper and lower bounds
    hdi_np = np.array(hdi[task])
    hdi_subnat_np = np.array(hdi_subnat[task])
    hdi_subnat_demean_np = np.array(hdi_subnat['demeaned_{}'.format(task)])
    lb = hdi_np.min(axis = 0)
    ub = hdi_np.max(axis = 0)
    lb_subnat = hdi_subnat_np.min(axis = 0)
    ub_subnat = hdi_subnat_np.max(axis = 0)
    lb_subnat_demean = hdi_subnat_demean_np.min(axis = 0)
    ub_subnat_demean = hdi_subnat_demean_np.max(axis = 0)
    
    ## extract weights
    wts_nat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_nat_both_{}.csv'.format(task)), delimiter = ',')
    wts_subnat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_both_{}.csv'.format(task)), delimiter = ',')
    wts_subnat_demean = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_both_{}_demeaned.csv'.format(task)), delimiter = ',')
    
    ## loop over national and subnational level predictions
    for w in (wts_nat, wts_subnat, wts_subnat_demean):
        for df in (X, X_subnat, X_subnat_demean, X_mosaiks, X_mosaiks_demean):
            
            ## skip national-level feature and subnational weights (upscaling exercise)
            if any(df.equals(y) for y in [X]) and (np.array_equiv(w, wts_subnat) or np.array_equiv(w, wts_subnat_demean)):
                continue
            
            ## restrict demean-demain pair
            if (any(df.equals(y) for y in [X_subnat]) or any(df.equals(y) for y in [X_mosaiks])) and (np.array_equiv(w, wts_subnat_demean)):
                continue
            elif (any(df.equals(y) for y in [X_subnat_demean]) or any(df.equals(y) for y in [X_mosaiks_demean])) and (np.array_equiv(w, wts_nat) or np.array_equiv(w, wts_subnat)):
                continue
            
            ## predict using national-level weights vector
            ypreds = np.dot(df, w)
            
            ## bound the prediction
            if np.array_equiv(w, wts_nat):
                ypreds[ypreds < lb] = lb
                ypreds[ypreds > ub] = ub
            elif np.array_equiv(w, wts_subnat):
                ypreds[ypreds < lb_subnat] = lb_subnat
                ypreds[ypreds > ub_subnat] = ub_subnat
            elif np.array_equiv(w, wts_subnat_demean):
                ypreds[ypreds < lb_subnat_demean] = lb_subnat_demean
                ypreds[ypreds > ub_subnat_demean] = ub_subnat_demean
            
            ## store predicted values
            if any(df.equals(y) for y in [X]):
                if 'eccu_preds' not in locals():
                    eccu_preds = pd.DataFrame([], index = df.index)
                eccu_preds['{}_preds'.format(task)] = ypreds.tolist()
            elif (any(df.equals(y) for y in [X_subnat])) or (any(df.equals(y) for y in [X_subnat_demean])):
                if 'eccu_subnat_preds' not in locals():
                    eccu_subnat_preds = pd.DataFrame([], index = df.index)
                if np.array_equiv(w, wts_nat):
                    eccu_subnat_preds['{}_preds'.format(task)] = ypreds.tolist()
                elif np.array_equiv(w, wts_subnat):
                    eccu_subnat_preds['{}_preds_subnat'.format(task)] = ypreds.tolist()
                elif np.array_equiv(w, wts_subnat_demean):
                    eccu_subnat_preds['{}_preds_subnat_demean'.format(task)] = ypreds.tolist()
            elif (any(df.equals(y) for y in [X_mosaiks])) or (any(df.equals(y) for y in [X_mosaiks_demean])):
                if 'eccu_mosaiks_preds' not in locals():
                    eccu_mosaiks_preds = pd.DataFrame([], index = df.index)
                if np.array_equiv(w, wts_nat):
                    eccu_mosaiks_preds['{}_preds'.format(task)] = ypreds.tolist()
                elif np.array_equiv(w, wts_subnat):
                    eccu_mosaiks_preds['{}_preds_subnat'.format(task)] = ypreds.tolist()
                elif np.array_equiv(w, wts_subnat_demean):
                    eccu_mosaiks_preds['{}_preds_subnat_demean'.format(task)] = ypreds.tolist()

###############################
## B) clean ground truth data 
###############################

## B-1. national level

## merge national-level HDI
merged = pd.merge(eccu_preds, hdi[tasks], left_index = True, right_index = True)
merged = merged.loc[(merged.index.values != 'BRB') & (merged.index.values != 'GLP') & (merged.index.values != 'MTQ')]

for task in tasks:
    
    ## plot prediction against 
    plt.clf()
    tot_min = np.min([np.min(np.array(merged['{}_preds'.format(task)])), np.min(np.array(merged[task]))])
    tot_max = np.max([np.max(np.array(merged['{}_preds'.format(task)])), np.max(np.array(merged[task]))])
    fig, ax = plt.subplots()
    ax.scatter(np.array(merged[task]), np.array(merged['{}_preds'.format(task)]))
    
    ## add 45 degree line and country names
    plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
    for i, txt in enumerate(np.array(merged.index)):
        ax.annotate(txt, (np.array(merged[task])[i], np.array(merged['{}_preds'.format(task)])[i]))
    
    ## add axis title
    if (task == 'hdi') or (task == 'gni'):
        ax.set_xlabel('True {}'.format(task.upper()))
        ax.set_ylabel('Predicted {}'.format(task.upper()))
    else:
        ax.set_xlabel('True {} Index'.format(task.capitalize()))
        ax.set_ylabel('Predicted {} Index'.format(task.capitalize()))
    
    ## output the graph
    fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_nat_{}_nat.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
    
    ## compute R-square from linear regression model
    model = LinearRegression().fit(merged[[task]], merged[['{}_preds'.format(task)]])
    globals()[f'r2_score_{task}'] = model.score(merged[[task]], merged[['{}_preds'.format(task)]])

## store MSE, MAE, R2
rows = [
    {'Metrics': 'HDI',
     'MSE': mean_squared_error(merged['hdi'], merged['hdi_preds']),
     'MAE': mean_absolute_error(merged['hdi'], merged['hdi_preds']),
     'R-square': r2_score_hdi},
    {'Metrics': 'GNI',
     'MSE': mean_squared_error(merged['gni'], merged['gni_preds']),
     'MAE': mean_absolute_error(merged['gni'], merged['gni_preds']),
     'R-square': r2_score_gni},
    {'Metrics': 'Health Index',
     'MSE': mean_squared_error(merged['health'], merged['health_preds']),
     'MAE': mean_absolute_error(merged['health'], merged['health_preds']),
     'R-square': r2_score_health},
    {'Metrics': 'Income Index',
     'MSE': mean_squared_error(merged['income'], merged['income_preds']),
     'MAE': mean_absolute_error(merged['income'], merged['income_preds']),
     'R-square': r2_score_income},
    {'Metrics': 'Education Index',
     'MSE': mean_squared_error(merged['ed'], merged['ed_preds']),
     'MAE': mean_absolute_error(merged['ed'], merged['ed_preds']),
     'R-square': r2_score_ed}
]

fn = os.path.join(c.out_dir, 'metrics', 'eccu_nat_hdi_metrics.csv')
with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MSE', 'MAE', 'R-square'])
    writer.writeheader()
    writer.writerows(rows)

## B-2. subnational maps

## initialize geodataframe
eccu_shp = gpd.GeoDataFrame([])

## loop over ECCU countries shapefiles
for dirs in os.listdir(os.path.join(c.data_dir, 'raw', 'shp')):
    if (dirs.startswith('gadm41_')) & (dirs.endswith('_shp')):
        
        ## extract country code
        ISO = dirs.replace('gadm41_', '', 1).replace('_shp', '', 1)
        
        ## load subnational shapefile
        shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', dirs, 'gadm41_{}_1.shp'.format(ISO)))
        
        ## append shapefiles
        eccu_shp = pd.concat([eccu_shp, shp])

## reindex 
eccu_shp = eccu_shp.set_index('GID_1')

## merge shapefile with predicted values
merged_subnat = gpd.GeoDataFrame(pd.merge(eccu_subnat_preds, eccu_shp[['GID_0', 'geometry']], left_index = True, right_index = True))

## visualization
for task in tasks:
    for x in ['{}_preds'.format(task), '{}_preds_subnat'.format(task), '{}_preds_subnat_demean'.format(task)]:
        plt.clf()
        fig, ax = plt.subplots()
        eccu_shp.to_crs(epsg = 4326).plot(ax = ax, color = 'lightgrey')
        merged_subnat.plot(column = x, ax = ax, cmap = 'RdYlGn', legend = True)
        fig.set_size_inches(30, 15, forward = True)
        if x == '{}_preds'.format(task):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_{}_nat.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
        elif x == '{}_preds_subnat'.format(task):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_{}_subnat.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
        elif x == '{}_preds_subnat_demean'.format(task):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_{}_demeaned.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)












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

