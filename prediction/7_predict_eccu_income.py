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

## create folder if not exists
if not os.path.exists(os.path.join(c.out_dir, 'income')):
    os.makedirs(os.path.join(c.out_dir, 'income'))

###############
## A) predict
###############

## A-1. extract weights vectors

wts_brb = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_mosaiks_both_income.csv'), delimiter = ',')
wts_lca = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'lca_mosaiks_both_income.csv'), delimiter = ',')
wts_brb_ed = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_ed_both_income.csv'), delimiter = ',')
wts_lca_settle = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'lca_settle_both_income.csv'), delimiter = ',')

## A-2. load aggregated MOSAIKS features

## load national and subnatinal-level data
eccu_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_nat_mosaiks_features.pkl'))
eccu_subnat_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_subnat_mosaiks_features.pkl'))
eccu_mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_mosaiks_mosaiks_features.pkl'))

## A-3. load aggregated NL features

## store the countries to loop over
countries = eccu_feat.index.values

## load national and subnational-level data
eccu_nl = pd.DataFrame([])
eccu_subnat_nl = pd.DataFrame([])
eccu_mosaiks_nl = pd.DataFrame([])

for x in ['nat', 'subnat', 'mosaiks']:
    for country in countries:
        nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_{}_nl.pkl'.format(country.lower(), x)))
        if x == 'nat':
            eccu_nl = pd.concat([eccu_nl, nl])
        elif x == 'subnat':
            eccu_subnat_nl = pd.concat([eccu_subnat_nl, nl])
        elif x == 'mosaiks':
            eccu_mosaiks_nl = pd.concat([eccu_mosaiks_nl, nl])

## merge MOSAIKS and NL features
X = pd.merge(eccu_feat, eccu_nl, left_index = True, right_index = True)
X_subnat = pd.merge(eccu_subnat_feat, eccu_subnat_nl, left_index = True, right_index = True)
X_mosaiks = pd.merge(eccu_mosaiks_feat, eccu_mosaiks_nl, left_index = True, right_index = True)

## A-4. prediction!

## loop over national level and subnational level predictions and then weights
for df in (X, X_subnat, X_mosaiks):
    for w in (wts_brb, wts_lca, wts_brb_ed, wts_lca_settle):
        
        ## extract bounds for density
        if np.array_equiv(w, wts_brb):
            income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_mosaiks_income.pkl'))
        elif np.array_equiv(w, wts_lca):
            income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_mosaiks_income.pkl'))
        elif np.array_equiv(w, wts_brb_ed):
            income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_ed_income.pkl'))
        elif np.array_equiv(w, wts_lca_settle):
            income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_settle_income.pkl'))
        lb = np.array(income['ln_income']).min(axis = 0)
        ub = np.array(income['ln_income']).max(axis = 0)
        
        ## store weights name
        name = next(x for x in globals() if globals()[x] is w)
        
        ## predict and bound the prediction
        ypreds = np.dot(df, w)
        ypreds[ypreds < lb] = lb
        ypreds[ypreds > ub] = ub
                
        ## store predicted values
        if any(df.equals(y) for y in [X]):
            if 'eccu_preds' not in locals():
                eccu_preds = pd.DataFrame([], index = df.index)
            eccu_preds['y_preds_{}'.format(name.replace('wts_', ''))] = ypreds.tolist()
        elif any(df.equals(y) for y in [X_subnat]):
            if 'eccu_subnat_preds' not in locals():
                eccu_subnat_preds = pd.DataFrame([], index = df.index)
            eccu_subnat_preds['y_preds_{}'.format(name.replace('wts_', ''))] = ypreds.tolist()
        elif any(df.equals(y) for y in [X_mosaiks]):
            if 'eccu_mosaiks_preds' not in locals():
                eccu_mosaiks_preds = pd.DataFrame([], index = df.index)
            eccu_mosaiks_preds['y_preds_{}'.format(name.replace('wts_', ''))] = ypreds.tolist()

eccu_preds.to_pickle(os.path.join(c.out_dir, 'income', 'eccu_nat_income_preds.pkl'))
eccu_subnat_preds.to_pickle(os.path.join(c.out_dir, 'income', 'eccu_subnat_income_preds.pkl'))
eccu_mosaiks_preds.to_pickle(os.path.join(c.out_dir, 'income', 'eccu_mosaiks_income_preds.pkl'))

###############################
## B) clean ground truth data 
###############################

## subnational level

brb_income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_parish_income.pkl'))
lca_income = pd.read_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_district_income.pkl'))

for col in eccu_subnat_preds.columns:
    clean_col = col.replace('y_preds_', '')
    for df in (brb_income, lca_income):
        
        ## merge prediction with income data
        merged = pd.merge(eccu_subnat_preds[col], df['ln_income'], left_index = True, right_index = True)
        
        ## plot prediction against true data
        plt.clf()
        tot_min = np.min([np.min(np.array(merged[col])), np.min(np.array(merged['ln_income']))])
        tot_max = np.max([np.max(np.array(merged[col])), np.max(np.array(merged['ln_income']))])
        fig, ax = plt.subplots()
        ax.scatter(np.array(merged['ln_income']), np.array(merged[col]))
        
        xmin = np.min(np.array(merged['ln_income']))
        p1, p0 = np.polyfit(np.array(merged['ln_income']), np.array(merged[col]), deg = 1)
        newp0 = p0 + xmin * p1
        ax.axline(xy1 = (xmin, newp0), slope = p1, color = 'r', lw = 2)
        stat = (f"$r$ = {np.corrcoef(merged['ln_income'], merged[col])[0][1]:.2f}")
        bbox = dict(boxstyle = 'round', fc = 'blanchedalmond', alpha = 0.5)
        ax.text(0.95, 0.07, stat, fontsize = 12, bbox = bbox, transform = ax.transAxes, horizontalalignment = 'right')
        
        ## add title
        ax.set_xlabel('True Log Income')
        if col == 'y_preds_brb' or col == 'y_preds_brb_ed':
            ax.set_ylabel('Predicted Log Income based on Barbados')
        elif col == 'y_preds_lca' or col == 'y_preds_lca_settle':
            ax.set_ylabel('Predicted Log Income based on St. Lucia')
        
        ## output the graph
        if any(df.equals(y) for y in [brb_income]):
            fig.savefig(os.path.join(c.out_dir, 'income', 'brb_subnat_income_{}.png'.format(clean_col)), bbox_inches = 'tight', pad_inches = 0.1)
        elif any(df.equals(y) for y in [lca_income]):
            fig.savefig(os.path.join(c.out_dir, 'income', 'lca_subnat_income_{}.png'.format(clean_col)), bbox_inches = 'tight', pad_inches = 0.1)

