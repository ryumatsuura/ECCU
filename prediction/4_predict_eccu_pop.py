## This script implements the prediction exercises

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from mosaiks import transforms
from mosaiks.utils.imports import *
from sklearn.metrics import *

###############
## A) predict
###############

## extract bounds for density
pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'outcome_sampled_population_global.csv'), index_col = 0)
pop = pop.loc[pop['population'].isnull() == False]
lb = np.array(pop['population']).min(axis = 0)
ub = np.array(pop['population']).max(axis = 0)

## A-1. extract weights vectors

wts_global = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_population.csv'), delimiter = ',')
wts_cont = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_population.csv'), delimiter = ',')
wts_cont_fixed = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_fixed_population.csv'), delimiter = ',')
wts_brb = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_population.csv'), delimiter = ',')
wts_glp = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'glp_population.csv'), delimiter = ',')
wts_mtq = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'mtq_population.csv'), delimiter = ',')
wts_nbr = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'nbr_population.csv'), delimiter = ',')
wts_nat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_nat_population.csv'), delimiter = ',')
wts_subnat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_population.csv'), delimiter = ',')

## A-2. load aggregated MOSAIKS features

eccu_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_nat.csv'), index_col = 0)
eccu_subnat_feat = pd.read_csv(os.path.join(c.features_dir, 'aggregate_mosaiks_features_subnat.csv'), index_col = 0)

## add sample columns - Grenada and Trinidad and Tobago are in sample 3 
for df in (eccu_feat, eccu_subnat_feat):
    df['sample'] = 0
    df.loc[df['Country'] == 'GRD', ['sample']] = 3 
    df.loc[df['Country'] == 'TTO', ['sample']] = 3 
    df['sample_fixed'] = 3

## A-3. prediction!

## loop over national level and subnational level predictions and then weights
for df in (eccu_feat, eccu_subnat_feat):
    for w in (wts_global, wts_cont, wts_cont_fixed, wts_brb, wts_glp, wts_mtq, wts_nbr, wts_nat, wts_subnat):
        
        ## store weights name
        name = next(x for x in globals() if globals()[x] is w)
        
        ## predict using global-scale weights vector
        if np.array_equiv(w, wts_global) or np.array_equiv(w, wts_brb) or np.array_equiv(w, wts_glp) or np.array_equiv(w, wts_mtq) or np.array_equiv(w, wts_nbr) or np.array_equiv(w, wts_nat) or np.array_equiv(w, wts_subnat):
            
            if any(df.equals(y) for y in [eccu_feat]):
                ypreds = np.dot(df.iloc[:, 1:4001], w)
            elif any(df.equals(y) for y in [eccu_subnat_feat]):
                ypreds = np.dot(df.iloc[:, 2:4002], w)
            
            ## bound the prediction
            ypreds[ypreds < lb] = lb
            ypreds[ypreds > ub] = ub
        
        elif np.array_equiv(w, wts_cont) or np.array_equiv(w, wts_cont_fixed):
            
            ## predict using continent-based weights vector
            for i in range(df.shape[0]):
                
                ## extract weight for each sample 
                mywts = w[df.loc[i, ['sample']].values[0]]
                
                ## predict
                if any(df.equals(y) for y in [eccu_feat]):
                    ypreds[i] = np.dot(df.iloc[i, 1:4001], mywts)
                elif any(df.equals(y) for y in [eccu_subnat_feat]):
                    ypreds[i] = np.dot(df.iloc[i, 2:4002], mywts)
                
                ## bound the prediction
                if ypreds[i] < lb:
                    ypreds[i] = lb
                if ypreds[i] > ub:
                    ypreds[i] = ub
        
        ## store predicted values
        if any(df.equals(y) for y in [eccu_feat]):
            if np.array_equiv(w, wts_global):
                eccu_preds = df[['Country']]
            eccu_preds['y_preds_{}'.format(name.replace('wts_', ''))] = ypreds.tolist()
        elif any(df.equals(y) for y in [eccu_subnat_feat]):
            if np.array_equiv(w, wts_global):
                eccu_subnat_preds = df[['Country']]
                eccu_subnat_preds['Name'] = df[['NAME_1']]
            eccu_subnat_preds['y_preds_{}'.format(name.replace('wts_', ''))] = ypreds.tolist()

###############################
## B) clean ground truth data 
###############################

eccu_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_eccu.csv'), index_col = 0)
eccu_subnat_pop = pd.read_csv(os.path.join(c.data_dir, 'int', 'applications', 'population', 'population_eccu_subnat.csv'), index_col = 0)

## merge two dataframes
merged = pd.merge(eccu_preds, eccu_pop)
merged_subnat = pd.merge(eccu_subnat_preds, eccu_subnat_pop)

## loop over level and weights
for df in (merged, merged_subnat):
    for x in ['global', 'cont', 'cont_fixed', 'brb', 'glp', 'mtq', 'nbr', 'nat', 'subnat']:
        
        ## plot prediction against ground truth
        plt.clf()
        tot_min = np.min([np.min(np.array(df['y_preds_{}'.format(x)])), np.min(np.array(df['Population']))])
        tot_max = np.max([np.max(np.array(df['y_preds_{}'.format(x)])), np.max(np.array(df['Population']))])
        fig, ax = plt.subplots()
        if any(df.equals(y) for y in [merged]):
            ax.scatter(np.array(df['Population']), np.array(df['y_preds_{}'.format(x)]))
        elif any(df.equals(y) for y in [merged_subnat]):
            c_labels, c_indices = np.unique(df['Country'], return_inverse = True)
            sc = ax.scatter(np.array(df['Population']), np.array(df['y_preds_{}'.format(x)]), c = c_indices)
            ax.legend(sc.legend_elements()[0], c_labels)
        
        ## add 45 degree line and country names
        plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
        if any(df.equals(y) for y in [merged]):
            for i, txt in enumerate(np.array(df['Country'])):
                ax.annotate(txt, (np.array(df['Population'])[i], np.array(df['y_preds_{}'.format(x)])[i]))
        
        ## add axis title
        ax.set_xlabel('True Population Density')
        ax.set_ylabel('Predicted Population Density')
        
        ## output the graph
        if any(df.equals(y) for y in [merged]):
            fig.savefig(os.path.join(c.out_dir, 'population', 'eccu_population_{}.png'.format(x)), bbox_inches = 'tight', pad_inches = 0.1)
        elif any(df.equals(y) for y in [merged_subnat]):
            fig.savefig(os.path.join(c.out_dir, 'population', 'eccu_subnat_population_{}.png'.format(x)), bbox_inches = 'tight', pad_inches = 0.1)
    
    ## store MSE, MAE, R2
    rows = [
        {'Metrics': 'Global-scale',
         'MSE': mean_squared_error(df['Population'], df['y_preds_global']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_global']),
         'R-square': r2_score(df['Population'], df['y_preds_global'])},
        {'Metrics': 'By-continent',
         'MSE': mean_squared_error(df['Population'], df['y_preds_cont']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_cont']),
         'R-square': r2_score(df['Population'], df['y_preds_cont'])},
        {'Metrics': 'By-continent fixed',
         'MSE': mean_squared_error(df['Population'], df['y_preds_cont_fixed']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_cont_fixed']),
         'R-square': r2_score(df['Population'], df['y_preds_cont_fixed'])},
        {'Metrics': 'Barbados-based',
         'MSE': mean_squared_error(df['Population'], df['y_preds_brb']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_brb']),
         'R-square': r2_score(df['Population'], df['y_preds_brb'])},
        {'Metrics': 'Guadeloupe-based',
         'MSE': mean_squared_error(df['Population'], df['y_preds_glp']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_glp']),
         'R-square': r2_score(df['Population'], df['y_preds_glp'])},
        {'Metrics': 'Martinique-based',
         'MSE': mean_squared_error(df['Population'], df['y_preds_mtq']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_mtq']),
         'R-square': r2_score(df['Population'], df['y_preds_mtq'])},
        {'Metrics': 'Neighbors-based',
         'MSE': mean_squared_error(df['Population'], df['y_preds_nbr']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_nbr']),
         'R-square': r2_score(df['Population'], df['y_preds_nbr'])},
        {'Metrics': 'National-level',
         'MSE': mean_squared_error(df['Population'], df['y_preds_nat']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_nat']),
         'R-square': r2_score(df['Population'], df['y_preds_nat'])},
        {'Metrics': 'Subnational-level',
         'MSE': mean_squared_error(df['Population'], df['y_preds_subnat']),
         'MAE': mean_absolute_error(df['Population'], df['y_preds_subnat']),
         'R-square': r2_score(df['Population'], df['y_preds_subnat'])}        
    ]
    
    ## set file name 
    if any(df.equals(y) for y in [merged]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_population_metrics.csv')
    elif any(df.equals(y) for y in [merged_subnat]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_subnat_population_metrics.csv')
    
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MSE', 'MAE', 'R-square'])
        writer.writeheader()
        writer.writerows(rows)
    
    ## store mean and variances into one csv file
    rows = [
        {'Descriptives': 'Ground-truth',
         'Mean': df['Population'].mean(),
         'Variance': df['Population'].var()},
        {'Descriptives': 'Global-scale',
         'Mean': df['y_preds_global'].mean(),
         'Variance': df['y_preds_global'].var()},
        {'Descriptives': 'By-continent',
         'Mean': df['y_preds_cont'].mean(),
         'Variance': df['y_preds_cont'].var()},
        {'Descriptives': 'By-continent fixed',
         'Mean': df['y_preds_cont_fixed'].mean(),
         'Variance': df['y_preds_cont_fixed'].var()},
        {'Descriptives': 'Barbados-based',
         'Mean': df['y_preds_brb'].mean(),
         'Variance': df['y_preds_brb'].var()},
        {'Descriptives': 'Guadeloupe-based',
         'Mean': df['y_preds_glp'].mean(),
         'Variance': df['y_preds_glp'].var()},
        {'Descriptives': 'Martinique-based',
         'Mean': df['y_preds_mtq'].mean(),
         'Variance': df['y_preds_mtq'].var()},
        {'Descriptives': 'Neighbors-based',
         'Mean': df['y_preds_nbr'].mean(),
         'Variance': df['y_preds_nbr'].var()},
        {'Descriptives': 'National-level',
         'Mean': df['y_preds_nat'].mean(),
         'Variance': df['y_preds_nat'].var()},
        {'Descriptives': 'Subnational-level',
         'Mean': df['y_preds_subnat'].mean(),
         'Variance': df['y_preds_subnat'].var()}
    ]
    
    ## set file name 
    if any(df.equals(y) for y in [merged]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_population_summary_stats.csv')
    elif any(df.equals(y) for y in [merged_subnat]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_subnat_population_summary_stats.csv')
    
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Descriptives', 'Mean', 'Variance'])
        writer.writeheader()
        writer.writerows(rows)


