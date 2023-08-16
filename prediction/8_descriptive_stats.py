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

