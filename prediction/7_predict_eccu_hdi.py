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

## specify outcome variables
tasks = ['hdi', 'gni', 'health', 'income', 'ed']

## load HDI measures
hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_adm0_clean.p'))
subnat_hdi = pd.read_pickle(os.path.join(c.data_dir, 'int', 'hdi', 'HDI_indicators_and_indices_clean.p'))

## rename columns 
hdi = hdi.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})
subnat_hdi = subnat_hdi.rename(columns = {'Sub-national HDI': 'hdi', 'GNI per capita in thousands of US$ (2011 PPP)': 'gni', 'Health index': 'health', 'Income index': 'income', 'Educational index ': 'ed'})

## demean subnational HDI
for task in tasks:
    merged = subnat_hdi[['ISO_Code', task]].merge(hdi[[task]], left_on = 'ISO_Code', right_index = True)
    subnat_hdi['demeaned_{}'.format(task)] = merged['{}_x'.format(task)] - merged['{}_y'.format(task)]
    del merged

## load IWI data
subnat_iwi = pd.read_csv(os.path.join(c.data_dir, 'raw', 'GDL', 'GDL-Mean-International-Wealth-Index-(IWI)-score-of-region-data.csv')).set_index('GDLCODE').rename(columns = {'2018': 'iwi'})
iwi = subnat_iwi.loc[subnat_iwi['Level'] == 'National', ['ISO_Code', 'iwi']].set_index('ISO_Code')
subnat_iwi = subnat_iwi.loc[subnat_iwi['Level'] == 'Subnat', ['ISO_Code', 'iwi']]

## demean subnational IWI
merged = subnat_iwi.merge(iwi, left_on = 'ISO_Code', right_index = True)
subnat_iwi['demeaned_iwi'] = merged['iwi_x'] - merged['iwi_y']
del merged

## merge in IWI into HDI dataset
hdi = hdi[['hdi', 'gni', 'health', 'income', 'ed']].merge(iwi, left_index = True, right_index = True, how = 'left')
subnat_hdi = subnat_hdi[['hdi', 'gni', 'health', 'income', 'ed', 'demeaned_hdi', 'demeaned_gni', 'demeaned_health', 'demeaned_income', 'demeaned_ed']].merge(subnat_iwi[['iwi', 'demeaned_iwi']], left_index = True, right_index = True, how = 'left')

## update task - add IWI
tasks.append('iwi')

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
    hdi_np = np.array(hdi[task].dropna())
    hdi_subnat_np = np.array(subnat_hdi[task].dropna())
    hdi_subnat_demean_np = np.array(subnat_hdi['demeaned_{}'.format(task)].dropna())
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
                        
            ## restrict demean-demain pair for demean weights
            if (any(df.equals(y) for y in [X]) or any(df.equals(y) for y in [X_subnat]) or any(df.equals(y) for y in [X_mosaiks])) and (np.array_equiv(w, wts_subnat_demean)):
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
                if np.array_equiv(w, wts_nat):
                    eccu_preds['{}_preds'.format(task)] = ypreds.tolist()
                elif np.array_equiv(w, wts_subnat):
                    eccu_preds['{}_preds_subnat'.format(task)] = ypreds.tolist()
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

eccu_preds.to_pickle(os.path.join(c.out_dir, 'hdi', 'eccu_nat_hdi_preds.pkl'))
eccu_subnat_preds.to_pickle(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_hdi_preds.pkl'))
eccu_mosaiks_preds.to_pickle(os.path.join(c.out_dir, 'hdi', 'eccu_mosaiks_hdi_preds.pkl'))

###############################
## B) clean ground truth data 
###############################

## B-1. national level

## merge national-level HDI
merged = pd.merge(eccu_preds, hdi[tasks], left_index = True, right_index = True)
merged = merged.loc[(merged.index.values != 'BRB') & (merged.index.values != 'GLP') & (merged.index.values != 'MTQ')]

for task in tasks:
    for x in ['{}_preds'.format(task), '{}_preds_subnat'.format(task)]:
        
        ## skip IWI - no ground-truth data
        if task == 'iwi':
            continue
        
        ## plot prediction against 
        plt.clf()
        tot_min = np.min([np.min(np.array(merged[x])), np.min(np.array(merged[task]))])
        tot_max = np.max([np.max(np.array(merged[x])), np.max(np.array(merged[task]))])
        fig, ax = plt.subplots()
        ax.scatter(np.array(merged[task]), np.array(merged[x]))
        
        ## add 45 degree line and country names
        plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
        for i, txt in enumerate(np.array(merged.index)):
            ax.annotate(txt, (np.array(merged[task])[i], np.array(merged[x])[i]))
        
        ## add axis title
        if (task == 'hdi') or (task == 'gni'):
            ax.set_xlabel('True {}'.format(task.upper()))
            ax.set_ylabel('Predicted {}'.format(task.upper()))
        else:
            ax.set_xlabel('True {} Index'.format(task.capitalize()))
            ax.set_ylabel('Predicted {} Index'.format(task.capitalize()))
        
        ## output the graph
        if x == '{}_preds'.format(task):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_nat_{}_nat.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
        elif x == '{}_preds_subnat'.format(task):
            fig.savefig(os.path.join(c.out_dir, 'hdi', 'eccu_nat_{}_subnat.png'.format(task)), bbox_inches = 'tight', pad_inches = 0.1)
        
        ## compute R-square from linear regression model
        model = LinearRegression().fit(merged[[task]], merged[[x]])
        if x == '{}_preds'.format(task):
            globals()[f'r2_score_{task}'] = model.score(merged[[task]], merged[[x]])
        elif x == '{}_preds_subnat'.format(task):
            globals()[f'r2_score_{task}_subnat'] = model.score(merged[[task]], merged[[x]])

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
     'R-square': r2_score_ed},
    {'Metrics': 'Subnat HDI',
     'MSE': mean_squared_error(merged['hdi'], merged['hdi_preds_subnat']),
     'MAE': mean_absolute_error(merged['hdi'], merged['hdi_preds_subnat']),
     'R-square': r2_score_hdi_subnat},
    {'Metrics': 'Subnat GNI',
     'MSE': mean_squared_error(merged['gni'], merged['gni_preds_subnat']),
     'MAE': mean_absolute_error(merged['gni'], merged['gni_preds_subnat']),
     'R-square': r2_score_gni_subnat},
    {'Metrics': 'Subnat Health Index',
     'MSE': mean_squared_error(merged['health'], merged['health_preds_subnat']),
     'MAE': mean_absolute_error(merged['health'], merged['health_preds_subnat']),
     'R-square': r2_score_health_subnat},
    {'Metrics': 'Subnat Income Index',
     'MSE': mean_squared_error(merged['income'], merged['income_preds_subnat']),
     'MAE': mean_absolute_error(merged['income'], merged['income_preds_subnat']),
     'R-square': r2_score_income_subnat},
    {'Metrics': 'Subnat Education Index',
     'MSE': mean_squared_error(merged['ed'], merged['ed_preds_subnat']),
     'MAE': mean_absolute_error(merged['ed'], merged['ed_preds_subnat']),
     'R-square': r2_score_ed_subnat}
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


