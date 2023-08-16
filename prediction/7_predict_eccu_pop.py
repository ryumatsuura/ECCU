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
if not os.path.exists(os.path.join(c.out_dir, 'population')):
    os.makedirs(os.path.join(c.out_dir, 'population'))

###############
## A) predict
###############

## A-1. extract weights vectors

wts_global = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_both_population.csv'), delimiter = ',')
wts_cont = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_both_population.csv'), delimiter = ',')
wts_cont_fixed = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_continent_fixed_both_population.csv'), delimiter = ',')
wts_brb = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_both_population.csv'), delimiter = ',')
wts_glp = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'glp_both_population.csv'), delimiter = ',')
wts_mtq = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'mtq_both_population.csv'), delimiter = ',')
wts_lca = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'lca_both_population.csv'), delimiter = ',')
wts_nbr = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'nbr_both_population.csv'), delimiter = ',')
wts_brb_ed = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'brb_ed_both_population.csv'), delimiter = ',')
wts_lca_settle = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'lca_settle_both_population.csv'), delimiter = ',')
wts_nat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_nat_both_population.csv'), delimiter = ',')
wts_subnat = np.genfromtxt(os.path.join(c.data_dir, 'int', 'weights', 'global_subnat_both_population.csv'), delimiter = ',')

## A-2. load aggregated MOSAIKS features

## load national and subnatinal-level data
eccu_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_nat_mosaiks_features.pkl'))
eccu_subnat_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_subnat_mosaiks_features.pkl'))
eccu_mosaiks_feat = pd.read_pickle(os.path.join(c.features_dir, 'eccu_mosaiks_mosaiks_features.pkl'))

## add sample columns - Grenada and Trinidad and Tobago are in sample 3 
for df in (eccu_feat, eccu_subnat_feat):
    df['sample'] = 0
    df.loc[df.index.str.contains('GRD'), ['sample']] = 3 
    df.loc[df.index.str.contains('TTO'), ['sample']] = 3 
    df['sample_fixed'] = 3

## add lat/lon to mosaiks dataframe
eccu_mosaiks_feat['lat'] = pd.DataFrame(eccu_mosaiks_feat.index.values, index = eccu_mosaiks_feat.index, columns = ['coords'])['coords'].str.split(':').str[0].astype(float)
eccu_mosaiks_feat['lon'] = pd.DataFrame(eccu_mosaiks_feat.index.values, index = eccu_mosaiks_feat.index, columns = ['coords'])['coords'].str.split(':').str[1].astype(float)

## add sample columns to MOSAIKS features
eccu_mosaiks_feat = parse.split_world_sample(eccu_mosaiks_feat).rename(columns = {'samp': 'sample'})
eccu_mosaiks_feat['sample_fixed'] = 3
eccu_mosaiks_feat = eccu_mosaiks_feat.drop(columns = ['lat', 'lon'])

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

## change the order of columns
X = pd.merge(X.drop(columns = {'sample', 'sample_fixed'}), X[['sample', 'sample_fixed']], left_index = True, right_index = True)
X_subnat = pd.merge(X_subnat.drop(columns = {'sample', 'sample_fixed'}), X_subnat[['sample', 'sample_fixed']], left_index = True, right_index = True)
X_mosaiks = pd.merge(X_mosaiks.drop(columns = {'sample', 'sample_fixed'}), X_mosaiks[['sample', 'sample_fixed']], left_index = True, right_index = True)

## A-4. prediction!

## loop over national level and subnational level predictions and then weights
for df in (X, X_subnat, X_mosaiks):
    for w in (wts_global, wts_cont, wts_cont_fixed, wts_brb, wts_glp, wts_mtq, wts_lca, wts_nbr, wts_brb_ed, wts_lca_settle, wts_nat, wts_subnat):
        
        ## extract bounds for density
        if np.array_equiv(w, wts_global) or np.array_equiv(w, wts_cont) or np.array_equiv(w, wts_cont_fixed):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'global_population.pkl'))
        elif np.array_equiv(w, wts_brb):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'brb_mosaiks_population.pkl'))
        elif np.array_equiv(w, wts_glp):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'glp_mosaiks_population.pkl'))
        elif np.array_equiv(w, wts_mtq):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'mtq_mosaiks_population.pkl'))
        elif np.array_equiv(w, wts_lca):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'lca_mosaiks_population.pkl'))
        elif np.array_equiv(w, wts_nbr):
            pop_brb = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'brb_mosaiks_population.pkl'))
            pop_glp = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'glp_mosaiks_population.pkl'))
            pop_mtq = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'mtq_mosaiks_population.pkl'))
            pop = pd.concat([pop_brb, pop_glp, pop_mtq])
        elif np.array_equiv(w, wts_brb_ed):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'brb_ed_population.pkl'))
        elif np.array_equiv(w, wts_lca_settle):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'lca_settle_population.pkl'))
        elif np.array_equiv(w, wts_nat):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'global_nat_population.pkl'))
        elif np.array_equiv(w, wts_subnat):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', 'global_subnat_population.pkl'))    
        lb = np.array(pop['ln_pop_density']).min(axis = 0)
        ub = np.array(pop['ln_pop_density']).max(axis = 0)
        
        ## store weights name
        name = next(x for x in globals() if globals()[x] is w)
        
        ## predict using global-scale weights vector
        if np.array_equiv(w, wts_global) or np.array_equiv(w, wts_brb) or np.array_equiv(w, wts_glp) or np.array_equiv(w, wts_mtq) or np.array_equiv(w, wts_lca) or np.array_equiv(w, wts_nbr) or np.array_equiv(w, wts_brb_ed) or np.array_equiv(w, wts_lca_settle) or np.array_equiv(w, wts_nat) or np.array_equiv(w, wts_subnat):
            
            ## predict and bound the prediction
            ypreds = np.dot(df.iloc[:, 0:4020], w)
            ypreds[ypreds < lb] = lb
            ypreds[ypreds > ub] = ub
        
        elif np.array_equiv(w, wts_cont) or np.array_equiv(w, wts_cont_fixed):
            
            ## predict using continent-based weights vector
            for i, ind in enumerate(df.index):
                
                ## extract weight for each sample 
                if np.array_equiv(w, wts_cont):
                    mywts = w[int(df.loc[ind, ['sample']].values[0])]
                elif np.array_equiv(w, wts_cont_fixed):
                    mywts = w[int(df.loc[ind, ['sample_fixed']].values[0])]
                
                ## predict and bound the prediction
                ypreds[i] = np.dot(df.iloc[i, 0:4020], mywts)
                if ypreds[i] < lb:
                    ypreds[i] = lb
                if ypreds[i] > ub:
                    ypreds[i] = ub
        
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

eccu_preds.to_pickle(os.path.join(c.out_dir, 'population', 'eccu_nat_population_preds.pkl'))
eccu_subnat_preds.to_pickle(os.path.join(c.out_dir, 'population', 'eccu_subnat_population_preds.pkl'))
eccu_mosaiks_preds.to_pickle(os.path.join(c.out_dir, 'population', 'eccu_mosaiks_population_preds.pkl'))

###############################
## B) clean ground truth data 
###############################

## load MOSAIKS feature and append
for x in ['nat', 'subnat', 'mosaiks']:
    globals()[f'eccu_{x}_pop'] = pd.DataFrame([])
    for files in os.listdir(os.path.join(c.data_dir, 'int', 'population')):
        if files.endswith('_{}_population.pkl'.format(x)) and not files.startswith('global_'):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', files))
            ISO = files.replace('_{}_population.pkl'.format(x), '').upper()
            pop['Country'] = ISO
            globals()[f'eccu_{x}_pop'] = pd.concat([globals()[f'eccu_{x}_pop'], pop])

## merge two dataframes
merged = pd.merge(eccu_preds, eccu_nat_pop, left_index = True, right_index = True)
merged_subnat = pd.merge(eccu_subnat_preds, eccu_subnat_pop, left_index = True, right_index = True)
merged_mosaiks = pd.merge(eccu_mosaiks_preds, eccu_mosaiks_pop, left_index = True, right_index = True)

## remove AIA, BRB, GLP, MSR, and MTQ from analysis 
merged = merged.loc[(merged['Country'] != 'AIA') & (merged['Country'] != 'BRB') & (merged['Country'] != 'GLP') & (merged['Country'] != 'MSR') & (merged['Country'] != 'MTQ')]
merged_subnat = merged_subnat.loc[(merged_subnat['Country'] != 'AIA') & (merged_subnat['Country'] != 'BRB') & (merged_subnat['Country'] != 'GLP') & (merged_subnat['Country'] != 'MSR') & (merged_subnat['Country'] != 'MTQ')]
merged_mosaiks = merged_mosaiks.loc[(merged_mosaiks['Country'] != 'AIA') & (merged_mosaiks['Country'] != 'BRB') & (merged_mosaiks['Country'] != 'GLP') & (merged_mosaiks['Country'] != 'MSR') & (merged_mosaiks['Country'] != 'MTQ')]

## loop over level and weights
for df in (merged, merged_subnat, merged_mosaiks):
    for col in eccu_preds.columns.values:
        clean_col = col.replace('y_preds_', '')
                
        ## plot prediction against ground truth
        plt.clf()
        tot_min = np.min([np.min(np.array(df[col])), np.min(np.array(df['ln_pop_density']))])
        tot_max = np.max([np.max(np.array(df[col])), np.max(np.array(df['ln_pop_density']))])
        fig, ax = plt.subplots()
        if any(df.equals(y) for y in [merged]):
            ax.scatter(np.array(df['ln_pop_density']), np.array(df[col]))
        else:
            c_labels, c_indices = np.unique(df['Country'], return_inverse = True)
            sc = ax.scatter(np.array(df['ln_pop_density']), np.array(df[col]), c = c_indices)
            ax.legend(sc.legend_elements()[0], c_labels)
        
        ## add 45 degree line and country names
        plt.plot([tot_min, tot_max], [tot_min, tot_max], color = 'black', linewidth = 2)
        if any(df.equals(y) for y in [merged]):
            for i, txt in enumerate(np.array(df['Country'])):
                ax.annotate(txt, (np.array(df['ln_pop_density'])[i], np.array(df[col])[i]))
        
        ## add axis title
        ax.set_xlabel('True Population Density')
        ax.set_ylabel('Predicted Population Density')
        
        ## output the graph
        if any(df.equals(y) for y in [merged]):
            fig.savefig(os.path.join(c.out_dir, 'population', 'eccu_nat_population_{}.png'.format(clean_col)), bbox_inches = 'tight', pad_inches = 0.1)
        elif any(df.equals(y) for y in [merged_subnat]):
            fig.savefig(os.path.join(c.out_dir, 'population', 'eccu_subnat_population_{}.png'.format(clean_col)), bbox_inches = 'tight', pad_inches = 0.1)
        elif any(df.equals(y) for y in [merged_mosaiks]):
            fig.savefig(os.path.join(c.out_dir, 'population', 'eccu_mosaiks_population_{}.png'.format(clean_col)), bbox_inches = 'tight', pad_inches = 0.1)
        
        ## compute R-square from linear regression model
        model = LinearRegression().fit(df[['ln_pop_density']], df[[col]])
        globals()[f'r2_score_{clean_col}'] = model.score(df[['ln_pop_density']], df[[col]])
    
    ## store MSE, MAE, R2
    rows = [
        {'Metrics': 'Global-scale',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_global']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_global']),
         'R-square': r2_score_global},
        {'Metrics': 'By-continent',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_cont']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_cont']),
         'R-square': r2_score_cont},
        {'Metrics': 'By-continent fixed',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_cont_fixed']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_cont_fixed']),
         'R-square': r2_score_cont_fixed},
        {'Metrics': 'Barbados-based',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_brb']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_brb']),
         'R-square': r2_score_brb},
        {'Metrics': 'Guadeloupe-based',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_glp']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_glp']),
         'R-square': r2_score_glp},
        {'Metrics': 'Martinique-based',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_mtq']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_mtq']),
         'R-square': r2_score_mtq},
        {'Metrics': 'St. Lucia-based',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_lca']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_lca']),
         'R-square': r2_score_lca},
        {'Metrics': 'Neighbors-based',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_nbr']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_nbr']),
         'R-square': r2_score_nbr},
        {'Metrics': 'National-level',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_nat']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_nat']),
         'R-square': r2_score_nat},
        {'Metrics': 'Subnational-level',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_subnat']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_subnat']),
         'R-square': r2_score_subnat},    
        {'Metrics': 'Barbados EB',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_brb_ed']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_brb_ed']),
         'R-square': r2_score_brb_ed},
        {'Metrics': 'St. Lucia settlement',
         'MSE': mean_squared_error(df['ln_pop_density'], df['y_preds_lca_settle']),
         'MAE': mean_absolute_error(df['ln_pop_density'], df['y_preds_lca_settle']),
         'R-square': r2_score_lca_settle}
    ]
    
    ## set file name 
    if any(df.equals(y) for y in [merged]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_nat_population_metrics.csv')
    elif any(df.equals(y) for y in [merged_subnat]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_subnat_population_metrics.csv')
    elif any(df.equals(y) for y in [merged_mosaiks]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_mosaiks_population_metrics.csv')
    
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Metrics', 'MSE', 'MAE', 'R-square'])
        writer.writeheader()
        writer.writerows(rows)
    
    ## store mean and variances into one csv file
    rows = [
        {'Descriptives': 'Ground-truth',
         'Mean': df['ln_pop_density'].mean(),
         'Variance': df['ln_pop_density'].var()},
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
        {'Descriptives': 'St. Lucia-based',
         'Mean': df['y_preds_lca'].mean(),
         'Variance': df['y_preds_lca'].var()},
        {'Descriptives': 'Neighbors-based',
         'Mean': df['y_preds_nbr'].mean(),
         'Variance': df['y_preds_nbr'].var()},
        {'Descriptives': 'National-level',
         'Mean': df['y_preds_nat'].mean(),
         'Variance': df['y_preds_nat'].var()},
        {'Descriptives': 'Subnational-level',
         'Mean': df['y_preds_subnat'].mean(),
         'Variance': df['y_preds_subnat'].var()},
        {'Descriptives': 'Barbados EB',
         'Mean': df['y_preds_brb_ed'].mean(),
         'Variance': df['y_preds_brb_ed'].var()},
        {'Descriptives': 'St. Lucia settlement',
         'Mean': df['y_preds_lca_settle'].mean(),
         'Variance': df['y_preds_lca_settle'].var()}
    ]
    
    ## set file name 
    if any(df.equals(y) for y in [merged]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_nat_population_summary_stats.csv')
    elif any(df.equals(y) for y in [merged_subnat]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_subnat_population_summary_stats.csv')
    elif any(df.equals(y) for y in [merged_mosaiks]):
        fn = os.path.join(c.out_dir, 'metrics', 'eccu_mosaiks_population_summary_stats.csv')
    
    with open(fn, 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['Descriptives', 'Mean', 'Variance'])
        writer.writeheader()
        writer.writerows(rows)

