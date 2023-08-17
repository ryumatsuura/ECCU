## This script implements the policy simulation exercises

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from mosaiks import transforms
from mosaiks.utils.imports import *
from sklearn.metrics import *
from scipy.stats import truncnorm

## set bootstrap samples
num_bootstrap = 25

## specify outcome variables - use three different measures for income/wealth
tasks = ['hdi', 'gni', 'income']

def generate_truncated_normal(lb, ub, mean, std, size):
    samples_log = truncnorm.rvs((np.log(lb)- np.log(mean)) / std, (np.log(ub) - np.log(mean)) / std, loc = np.log(mean), scale = std, size = size)
    samples = np.exp(samples_log)
    return samples

def calc_percentile(group, val_col, p):
    return np.percentile(group[val_col], p)

def compute_std(group, val_col, weight_col):
    weighted_mean = np.sum(group[val_col] * group[weight_col]) / np.sum(group[weight_col])
    return np.sqrt(np.sum(group[weight_col] * (group[val_col] - weighted_mean) ** 2) / np.sum(group[weight_col]))

def calc_counts(group, true_col, model_col):
    tp = ((group[true_col] == True) & (group[model_col] == True)).sum()
    tn = ((group[true_col] == False) & (group[model_col] == False)).sum()
    fp = ((group[true_col] == False) & (group[model_col] == True)).sum()
    fn = ((group[true_col] == True) & (group[model_col] == False)).sum()
    return pd.Series({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})

# create folder if not exists
if not os.path.exists(os.path.join(c.out_dir, 'metrics')):
    os.makedirs(os.path.join(c.out_dir, 'metrics'))

#############################
## A) clean prediction data
#############################

## load subnational and MOSAIKS level prediction values
eccu_nat_preds = pd.read_pickle(os.path.join(c.out_dir, 'hdi', 'eccu_nat_hdi_preds.pkl'))
eccu_subnat_preds = pd.read_pickle(os.path.join(c.out_dir, 'hdi', 'eccu_subnat_hdi_preds.pkl'))
eccu_mosaiks_preds = pd.read_pickle(os.path.join(c.out_dir, 'hdi', 'eccu_mosaiks_hdi_preds.pkl'))
eccu_subnat_key = pd.read_pickle(os.path.join(c.data_dir, 'int', 'keys', 'eccu_subnat_mosaiks_key.pkl'))

## load population data and append
for x in ['nat', 'subnat', 'mosaiks']:
    globals()[f'eccu_{x}_pop'] = pd.DataFrame([])
    for files in os.listdir(os.path.join(c.data_dir, 'int', 'population')):
        if files.endswith('_{}_population.pkl'.format(x)) and not files.startswith('global_'):
            pop = pd.read_pickle(os.path.join(c.data_dir, 'int', 'population', files))
            ISO = files.replace('_{}_population.pkl'.format(x), '').upper()
            pop['Country'] = ISO
            globals()[f'eccu_{x}_pop'] = pd.concat([globals()[f'eccu_{x}_pop'], pop])

## merge two dataframes
merged_nat = pd.merge(eccu_nat_preds, eccu_nat_pop, left_index = True, right_index = True)
merged_subnat = pd.merge(eccu_subnat_preds, eccu_subnat_pop, left_index = True, right_index = True)
merged_mosaiks = pd.merge(eccu_mosaiks_preds, eccu_mosaiks_pop, left_index = True, right_index = True)

## remove AIA, BRB, GLP, MSR, and MTQ from analysis 
merged_nat = merged_nat.loc[(merged_nat['Country'] != 'AIA') & (merged_nat['Country'] != 'BRB') & (merged_nat['Country'] != 'GLP') & (merged_nat['Country'] != 'MSR') & (merged_nat['Country'] != 'MTQ')]
merged_subnat = merged_subnat.loc[(merged_subnat['Country'] != 'AIA') & (merged_subnat['Country'] != 'BRB') & (merged_subnat['Country'] != 'GLP') & (merged_subnat['Country'] != 'MSR') & (merged_subnat['Country'] != 'MTQ')]
merged_mosaiks = merged_mosaiks.loc[(merged_mosaiks['Country'] != 'AIA') & (merged_mosaiks['Country'] != 'BRB') & (merged_mosaiks['Country'] != 'GLP') & (merged_mosaiks['Country'] != 'MSR') & (merged_mosaiks['Country'] != 'MTQ')]

## merge in subnational ID
merged_mosaiks = merged_mosaiks.merge(eccu_subnat_key, left_index = True, right_index = True)

## compute the number of localities
nat_counts = merged_mosaiks.groupby('Country')['total_pop'].count().to_frame()
merged_nat = merged_nat.merge(nat_counts.rename(columns = {'total_pop': 'counts'}), left_index = True, right_index = True)
subnat_counts = merged_mosaiks.groupby('GID_1')['total_pop'].count().to_frame()
merged_subnat = merged_subnat.merge(subnat_counts.rename(columns = {'total_pop': 'counts'}), left_index = True, right_index = True)

## compute standard deviation for national and subnational-level
for task in tasks:
    
    ## compute the national-level standard deviation based on the subnational-level prediction data
    nat_std = merged_subnat.groupby('Country').apply(compute_std, val_col = '{}_preds_subnat'.format(task), weight_col = 'total_pop').to_frame()
    nat_std.columns = ['{}_preds_subnat_std'.format(task)]
    merged_nat = merged_nat.merge(nat_std, left_index = True, right_index = True)
    
    ## add min/max
    nat_stats = merged_subnat.groupby('Country')['{}_preds_subnat'.format(task)].describe()
    merged_nat = merged_nat.merge(nat_stats[['min', 'max']].rename(columns = {'min': '{}_preds_subnat_min'.format(task), 'max': '{}_preds_subnat_max'.format(task)}), left_index = True, right_index = True)
    
    ## compute the subnational-level standard deviation based on the MOSAIKS-level prediction data
    subnat_std = merged_mosaiks.groupby('GID_1').apply(compute_std, val_col = '{}_preds_subnat'.format(task), weight_col = 'total_pop').to_frame()
    subnat_std.columns = ['{}_preds_subnat_std'.format(task)]
    merged_subnat = merged_subnat.merge(subnat_std, left_index = True, right_index = True)
    
    ## add min/max
    subnat_stats = merged_mosaiks.groupby('GID_1')['{}_preds_subnat'.format(task)].describe()
    merged_subnat = merged_subnat.merge(subnat_stats[['min', 'max']].rename(columns = {'min': '{}_preds_subnat_min'.format(task), 'max': '{}_preds_subnat_max'.format(task)}), left_index = True, right_index = True)

## expand national-level data to subnational-level
merged_nat_exp = pd.merge(merged_subnat[['Country', 'counts']], merged_nat.drop(columns = ['Country', 'counts']), left_on = 'Country', right_index = True)

#############################################
## B) implement policy simulation exercises
#############################################

np.random.seed(1)

## B-1. Use percentile

## set different threshold
ps = np.linspace(0, 100, 1000)

## Model 1: 
### Program eligiblity: Population in the lowest p percentile are eligible for the program
### Model assumption: Income distribution for the each parish is log-normal and we model based on national-level summary statistics
### Those in the lowest p percentile in each parish are eligible

merged_mosaiks.sort_values(by = ['GID_1'], inplace = True)

## initialize a new dataframe
model1_errors_b = pd.DataFrame([])

for b in range(num_bootstrap):
    for task in tasks:
        
        ## create log-normal distribution for each parish
        samples = merged_nat_exp.apply(lambda x: generate_truncated_normal(x['{}_preds_subnat_min'.format(task)], x['{}_preds_subnat_max'.format(task)], x['{}_preds_subnat'.format(task)], x['{}_preds_subnat_std'.format(task)], x['counts']), axis = 1).to_frame().reset_index()
        samples.columns = ['GID_1', 'lognormal_dist']
        
        ## convert parish-level data to MOSAIKS-level data
        samples_df = pd.DataFrame([])
        _, idx = np.unique(merged_mosaiks.GID_1, return_index = True)
        for i, parish in enumerate(merged_mosaiks.GID_1[np.sort(idx)]):
            samples_df = pd.concat([samples_df, pd.DataFrame({'Parish': np.full(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0].shape[0], parish), 'lognormal_dist': np.hstack(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0])})])
        
        ## horizontally concatenate MOSAIKS-level data
        samples_df = samples_df.set_index(merged_mosaiks.index)
        merged_mosaiks = pd.concat([merged_mosaiks, samples_df], axis = 1)
        assert (merged_mosaiks['GID_1'] == merged_mosaiks['Parish']).all()
        
        for p in ps:
            
            ## define ground-truth eligibilty
            perc_nat = merged_mosaiks.groupby('Country').apply(calc_percentile, val_col = '{}_preds_subnat'.format(task), p = p).to_frame()
            perc_nat.columns = ['percentile']
            merged_mosaiks = merged_mosaiks.merge(perc_nat, left_on = 'Country', right_index = True)
            merged_mosaiks['prog_eligible'] = merged_mosaiks['{}_preds_subnat'.format(task)] <= merged_mosaiks['percentile']
            
            ## define model eligibility
            model_perc_subnat = merged_mosaiks.groupby('GID_1').apply(calc_percentile, val_col = 'lognormal_dist', p = p).to_frame()
            model_perc_subnat.columns = ['model_percentile']
            merged_mosaiks = merged_mosaiks.merge(model_perc_subnat, left_on = 'GID_1', right_index = True)
            merged_mosaiks['model_eligible'] = merged_mosaiks['lognormal_dist'] <= merged_mosaiks['model_percentile']
            
            ## compute inclusion and exclusion errors 
            counts = merged_mosaiks.groupby('Country').apply(calc_counts, true_col = 'prog_eligible', model_col = 'model_eligible')
            
            ## compute inclusion error
            counts['inc_err'] = counts['FP'] / (counts['TN'] + counts['FP'])
            counts['inc_err'] = counts['inc_err'].fillna(1)
            
            ## compute exclusion error
            counts['exc_err'] = counts['FN'] / (counts['TP'] + counts['FN'])
            counts['exc_err'] = counts['exc_err'].fillna(0)
            
            ## add task and p-th threshold 
            counts['task'] = task
            counts['p_threshold'] = p
            counts['bootstrap'] = b
            
            ## concatenate to dataframe
            model1_errors_b = pd.concat([model1_errors_b, counts[['inc_err', 'exc_err', 'task', 'p_threshold', 'bootstrap']]])
            
            ## remove recurring columns
            merged_mosaiks = merged_mosaiks.drop(columns = ['percentile', 'prog_eligible', 'model_percentile', 'model_eligible'])
        
        ## remove recurring columns
        merged_mosaiks = merged_mosaiks.drop(columns = ['Parish', 'lognormal_dist'])
    
    if (b + 1) % 5 == 0:
        print(f'{b + 1} iterations completed')

model1_errors_b = model1_errors_b.reset_index()

## compute mean and confidence interval
inc_err_means = model1_errors_b.groupby(['Country', 'task', 'p_threshold'])['inc_err'].mean().to_frame()
inc_err_cis = model1_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'inc_err', p = [2.5, 97.5]).to_frame()
inc_err_cis.columns = ['inc_err_ci']
exc_err_means = model1_errors_b.groupby(['Country', 'task', 'p_threshold'])['exc_err'].mean().to_frame()
exc_err_cis = model1_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'exc_err', p = [2.5, 97.5]).to_frame()
exc_err_cis.columns = ['exc_err_ci']

## merge 
model1_errors = inc_err_means.merge(inc_err_cis, left_index = True, right_index = True).merge(exc_err_means, left_index = True, right_index = True).merge(exc_err_cis, left_index = True, right_index = True)
model1_errors = model1_errors.reset_index().set_index('Country')

## Model 2: 
### Program eligibility: Population in the lowest p percentile are eligible for the program
### Model assumption: Income distribution in each parish is log-normal and we model based on parish-level summary statistics
### Those in the lowest p percentile in each parish are eligible 

merged_mosaiks.sort_values(by = ['GID_1'], inplace = True)

## initialize a new dataframe
model2_errors_b = pd.DataFrame([])

for b in range(num_bootstrap):
    for task in tasks:
        
        ## create log-normal distribution for each parish
        samples = merged_subnat.apply(lambda x: generate_truncated_normal(x['{}_preds_subnat_min'.format(task)], x['{}_preds_subnat_max'.format(task)], x['{}_preds_subnat'.format(task)], x['{}_preds_subnat_std'.format(task)], x['counts']), axis = 1).to_frame().reset_index()
        samples.columns = ['GID_1', 'lognormal_dist']
        
        ## convert parish-level data to MOSAIKS-level data
        samples_df = pd.DataFrame([])
        _, idx = np.unique(merged_mosaiks.GID_1, return_index = True)
        for i, parish in enumerate(merged_mosaiks.GID_1[np.sort(idx)]):
            samples_df = pd.concat([samples_df, pd.DataFrame({'Parish': np.full(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0].shape[0], parish), 'lognormal_dist': np.hstack(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0])})])
        
        ## horizontally concatenate MOSAIKS-level data
        samples_df = samples_df.set_index(merged_mosaiks.index)
        merged_mosaiks = pd.concat([merged_mosaiks, samples_df], axis = 1)
        assert (merged_mosaiks['GID_1'] == merged_mosaiks['Parish']).all()
        
        for p in ps:
            
            ## define ground-truth eligibilty
            perc_nat = merged_mosaiks.groupby('Country').apply(calc_percentile, val_col = '{}_preds_subnat'.format(task), p = p).to_frame()
            perc_nat.columns = ['percentile']
            merged_mosaiks = merged_mosaiks.merge(perc_nat, left_on = 'Country', right_index = True)
            merged_mosaiks['prog_eligible'] = merged_mosaiks['{}_preds_subnat'.format(task)] <= merged_mosaiks['percentile']
            
            ## define model eligibility
            model_perc_subnat = merged_mosaiks.groupby('GID_1').apply(calc_percentile, val_col = 'lognormal_dist', p = p).to_frame()
            model_perc_subnat.columns = ['model_percentile']
            merged_mosaiks = merged_mosaiks.merge(model_perc_subnat, left_on = 'GID_1', right_index = True)
            merged_mosaiks['model_eligible'] = merged_mosaiks['lognormal_dist'] <= merged_mosaiks['model_percentile']
            
            ## compute inclusion and exclusion errors 
            counts = merged_mosaiks.groupby('Country').apply(calc_counts, true_col = 'prog_eligible', model_col = 'model_eligible')
            
            ## compute inclusion error
            counts['inc_err'] = counts['FP'] / (counts['TN'] + counts['FP'])
            counts['inc_err'] = counts['inc_err'].fillna(1)
            
            ## compute exclusion error
            counts['exc_err'] = counts['FN'] / (counts['TP'] + counts['FN'])
            counts['exc_err'] = counts['exc_err'].fillna(0)
            
            ## add task and p-th threshold 
            counts['task'] = task
            counts['p_threshold'] = p
            counts['bootstrap'] = b
            
            ## concatenate to dataframe
            model2_errors_b = pd.concat([model2_errors_b, counts[['inc_err', 'exc_err', 'task', 'p_threshold', 'bootstrap']]])
            
            ## remove recurring columns
            merged_mosaiks = merged_mosaiks.drop(columns = ['percentile', 'prog_eligible', 'model_percentile', 'model_eligible'])
        
        ## remove recurring columns
        merged_mosaiks = merged_mosaiks.drop(columns = ['Parish', 'lognormal_dist'])
    
    if (b + 1) % 5 == 0:
        print(f'{b + 1} iterations completed')

model2_errors_b = model2_errors_b.reset_index()

## compute mean and confidence interval
inc_err_means = model2_errors_b.groupby(['Country', 'task', 'p_threshold'])['inc_err'].mean().to_frame()
inc_err_cis = model2_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'inc_err', p = [2.5, 97.5]).to_frame()
inc_err_cis.columns = ['inc_err_ci']
exc_err_means = model2_errors_b.groupby(['Country', 'task', 'p_threshold'])['exc_err'].mean().to_frame()
exc_err_cis = model2_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'exc_err', p = [2.5, 97.5]).to_frame()
exc_err_cis.columns = ['exc_err_ci']

## merge 
model2_errors = inc_err_means.merge(inc_err_cis, left_index = True, right_index = True).merge(exc_err_means, left_index = True, right_index = True).merge(exc_err_cis, left_index = True, right_index = True)
model2_errors = model2_errors.reset_index().set_index('Country')

## Model 3:
### Program eligiblity: Population in the lowest p percentile are eligible for the program
### Model assumption: Income distribution in each parish is log-normal and those in the lowest p_p percentile in each parish are eligible
### where p_p is determined so that the aggregated eligible population from all parishes add up to the population in the lowest p percentile in the country

merged_mosaiks.sort_values(by = ['GID_1'], inplace = True)

## initialize a new dataframe
model3_errors_b = pd.DataFrame([])

for b in range(num_bootstrap):
    for task in tasks:
        
        ## create log-normal distribution for each parish
        samples = merged_subnat.apply(lambda x: generate_truncated_normal(x['{}_preds_subnat_min'.format(task)], x['{}_preds_subnat_max'.format(task)], x['{}_preds_subnat'.format(task)], x['{}_preds_subnat_std'.format(task)], x['counts']), axis = 1).to_frame().reset_index()
        samples.columns = ['GID_1', 'lognormal_dist']
        
        ## convert parish-level data to MOSAIKS-level data
        samples_df = pd.DataFrame([])
        _, idx = np.unique(merged_mosaiks.GID_1, return_index = True)
        for i, parish in enumerate(merged_mosaiks.GID_1[np.sort(idx)]):
            samples_df = pd.concat([samples_df, pd.DataFrame({'Parish': np.full(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0].shape[0], parish), 'lognormal_dist': np.hstack(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0])})])
        
        ## horizontally concatenate MOSAIKS-level data
        samples_df = samples_df.set_index(merged_mosaiks.index)
        merged_mosaiks = pd.concat([merged_mosaiks, samples_df], axis = 1)
        assert (merged_mosaiks['GID_1'] == merged_mosaiks['Parish']).all()
        
        for p in ps:
            
            ## define ground-truth eligibilty
            perc_nat = merged_mosaiks.groupby('Country').apply(calc_percentile, val_col = '{}_preds_subnat'.format(task), p = p).to_frame()
            perc_nat.columns = ['percentile']
            merged_mosaiks = merged_mosaiks.merge(perc_nat, left_on = 'Country', right_index = True)
            merged_mosaiks['prog_eligible'] = merged_mosaiks['{}_preds_subnat'.format(task)] <= merged_mosaiks['percentile']
            
            ## define model eligibility
            model_perc_nat = merged_mosaiks.groupby('Country').apply(calc_percentile, val_col = 'lognormal_dist', p = p).to_frame()
            model_perc_nat.columns = ['model_percentile']
            merged_mosaiks = merged_mosaiks.merge(model_perc_nat, left_on = 'Country', right_index = True)
            merged_mosaiks['model_eligible'] = merged_mosaiks['lognormal_dist'] <= merged_mosaiks['model_percentile']
            
            ## compute inclusion and exclusion errors 
            counts = merged_mosaiks.groupby('Country').apply(calc_counts, true_col = 'prog_eligible', model_col = 'model_eligible')
            
            ## compute inclusion error
            counts['inc_err'] = counts['FP'] / (counts['TN'] + counts['FP'])
            counts['inc_err'] = counts['inc_err'].fillna(1)
            
            ## compute exclusion error
            counts['exc_err'] = counts['FN'] / (counts['TP'] + counts['FN'])
            counts['exc_err'] = counts['exc_err'].fillna(0)
            
            ## add task and p-th threshold 
            counts['task'] = task
            counts['p_threshold'] = p
            counts['bootstrap'] = b
            
            ## concatenate to dataframe
            model3_errors_b = pd.concat([model3_errors_b, counts[['inc_err', 'exc_err', 'task', 'p_threshold', 'bootstrap']]])
            
            ## remove recurring columns
            merged_mosaiks = merged_mosaiks.drop(columns = ['percentile', 'prog_eligible', 'model_percentile', 'model_eligible'])
        
        ## remove recurring columns
        merged_mosaiks = merged_mosaiks.drop(columns = ['Parish', 'lognormal_dist'])
    
    if (b + 1) % 5 == 0:
        print(f'{b + 1} iterations completed')

model3_errors_b = model3_errors_b.reset_index()

## compute mean and confidence interval
inc_err_means = model3_errors_b.groupby(['Country', 'task', 'p_threshold'])['inc_err'].mean().to_frame()
inc_err_cis = model3_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'inc_err', p = [2.5, 97.5]).to_frame()
inc_err_cis.columns = ['inc_err_ci']
exc_err_means = model3_errors_b.groupby(['Country', 'task', 'p_threshold'])['exc_err'].mean().to_frame()
exc_err_cis = model3_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'exc_err', p = [2.5, 97.5]).to_frame()
exc_err_cis.columns = ['exc_err_ci']

## merge 
model3_errors = inc_err_means.merge(inc_err_cis, left_index = True, right_index = True).merge(exc_err_means, left_index = True, right_index = True).merge(exc_err_cis, left_index = True, right_index = True)
model3_errors = model3_errors.reset_index().set_index('Country')

## Draw Models1-3 

assert sorted(model1_errors.index.unique()) == sorted(model2_errors.index.unique()) == sorted(model3_errors.index.unique())

for task in tasks:
    for country in np.unique(model1_errors.index):
        
        plt.close()
        
        ## draw inclusino and exclusion errors from models 
        plt.plot(model1_errors[(model1_errors.index == country) & (model1_errors.task == task)].inc_err, model1_errors[(model1_errors.index == country) & (model1_errors.task == task)].exc_err, label = 'Model 1')
        plt.plot(model2_errors[(model2_errors.index == country) & (model2_errors.task == task)].inc_err, model2_errors[(model2_errors.index == country) & (model2_errors.task == task)].exc_err, label = 'Model 2')
        plt.plot(model3_errors[(model3_errors.index == country) & (model3_errors.task == task)].inc_err, model3_errors[(model3_errors.index == country) & (model3_errors.task == task)].exc_err, label = 'Model 3')
        
        ## label the graph
        plt.xlabel('Inclusion error')
        plt.ylabel('Exclusion error')
        plt.title(country)
        plt.legend()
        plt.savefig(os.path.join(c.out_dir, 'metrics', '{}_{}_perc_eic.png'.format(country.lower(), task)), bbox_inches = 'tight', pad_inches = 0.1)

## B-2. Use absolute value

## Model 1: 
### Program eligiblity: Population whose HDI/GNI/income index are below p are eligible for the program
### Model assumption: Income distribution for the each parish is log-normal and we model based on national-level summary statistics
### Those with HDI/GNI/income index below p are eligible

merged_mosaiks.sort_values(by = ['GID_1'], inplace = True)

## initialize a new dataframe
model1_errors_b = pd.DataFrame([])

for b in range(num_bootstrap):
    for task in tasks:
        
        ## create log-normal distribution for each parish
        samples = merged_nat_exp.apply(lambda x: generate_truncated_normal(x['{}_preds_subnat_min'.format(task)], x['{}_preds_subnat_max'.format(task)], x['{}_preds_subnat'.format(task)], x['{}_preds_subnat_std'.format(task)], x['counts']), axis = 1).to_frame().reset_index()
        samples.columns = ['GID_1', 'lognormal_dist']
        
        ## convert parish-level data to MOSAIKS-level data
        samples_df = pd.DataFrame([])
        _, idx = np.unique(merged_mosaiks.GID_1, return_index = True)
        for i, parish in enumerate(merged_mosaiks.GID_1[np.sort(idx)]):
            samples_df = pd.concat([samples_df, pd.DataFrame({'Parish': np.full(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0].shape[0], parish), 'lognormal_dist': np.hstack(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0])})])
        
        ## horizontally concatenate MOSAIKS-level data
        samples_df = samples_df.set_index(merged_mosaiks.index)
        merged_mosaiks = pd.concat([merged_mosaiks, samples_df], axis = 1)
        assert (merged_mosaiks['GID_1'] == merged_mosaiks['Parish']).all()
        
        ## determine range of values to explore
        ps = np.linspace(merged_mosaiks['lognormal_dist'].min(), merged_mosaiks['lognormal_dist'].max(), 1000)
        
        for i, p in enumerate(ps):
            
            ## define ground-truth eligibility
            merged_mosaiks['prog_eligible'] = merged_mosaiks['{}_preds_subnat'.format(task)] <= p
            
            ## define model eligibility
            merged_mosaiks['model_eligible'] = merged_mosaiks['lognormal_dist'] <= p
            
            ## compute inclusion and exclusion errors
            counts = merged_mosaiks.groupby('Country').apply(calc_counts, true_col = 'prog_eligible', model_col = 'model_eligible')
            
            ## compute inclusion error
            counts['inc_err'] = counts['FP'] / (counts['TN'] + counts['FP'])
            counts['inc_err'] = counts['inc_err'].fillna(1)
            
            ## compute exclusion error
            counts['exc_err'] = counts['FN'] / (counts['TP'] + counts['FN'])
            counts['exc_err'] = counts['exc_err'].fillna(0)
            
            ## add task and p-th threshold 
            counts['task'] = task
            counts['p_threshold'] = i
            counts['bootstrap'] = b
            
            ## concatenate to dataframe
            model1_errors_b = pd.concat([model1_errors_b, counts[['inc_err', 'exc_err', 'task', 'p_threshold', 'bootstrap']]])
            
            ## remove recurring columns
            merged_mosaiks = merged_mosaiks.drop(columns = ['prog_eligible', 'model_eligible'])
        
        ## remove recurring columns
        merged_mosaiks = merged_mosaiks.drop(columns = ['Parish', 'lognormal_dist'])
    
    if (b + 1) % 5 == 0:
        print(f'{b + 1} iterations completed')

model1_errors_b = model1_errors_b.reset_index()

## compute mean and confidence interval
inc_err_means = model1_errors_b.groupby(['Country', 'task', 'p_threshold'])['inc_err'].mean().to_frame()
inc_err_cis = model1_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'inc_err', p = [2.5, 97.5]).to_frame()
inc_err_cis.columns = ['inc_err_ci']
exc_err_means = model1_errors_b.groupby(['Country', 'task', 'p_threshold'])['exc_err'].mean().to_frame()
exc_err_cis = model1_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'exc_err', p = [2.5, 97.5]).to_frame()
exc_err_cis.columns = ['exc_err_ci']

## merge 
model1_errors = inc_err_means.merge(inc_err_cis, left_index = True, right_index = True).merge(exc_err_means, left_index = True, right_index = True).merge(exc_err_cis, left_index = True, right_index = True)
model1_errors = model1_errors.reset_index().set_index('Country')

## Model 2: 
### Program eligibility: Population whose HDI/GNI/income index are below p are eligible for the program
### Model assumption: Income distribution in each parish is log-normal and we model based on parish-level summary statistics
### Those with HDI/GNI/income index below p are eligible 

merged_mosaiks.sort_values(by = ['GID_1'], inplace = True)

## initialize a new dataframe
model2_errors_b = pd.DataFrame([])

for b in range(num_bootstrap):
    for task in tasks:
        
        ## create log-normal distribution for each parish
        samples = merged_subnat.apply(lambda x: generate_truncated_normal(x['{}_preds_subnat_min'.format(task)], x['{}_preds_subnat_max'.format(task)], x['{}_preds_subnat'.format(task)], x['{}_preds_subnat_std'.format(task)], x['counts']), axis = 1).to_frame().reset_index()
        samples.columns = ['GID_1', 'lognormal_dist']
        
        ## convert parish-level data to MOSAIKS-level data
        samples_df = pd.DataFrame([])
        _, idx = np.unique(merged_mosaiks.GID_1, return_index = True)
        for i, parish in enumerate(merged_mosaiks.GID_1[np.sort(idx)]):
            samples_df = pd.concat([samples_df, pd.DataFrame({'Parish': np.full(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0].shape[0], parish), 'lognormal_dist': np.hstack(samples[samples['GID_1'] == parish]['lognormal_dist'].values[0])})])
        
        ## horizontally concatenate MOSAIKS-level data
        samples_df = samples_df.set_index(merged_mosaiks.index)
        merged_mosaiks = pd.concat([merged_mosaiks, samples_df], axis = 1)
        assert (merged_mosaiks['GID_1'] == merged_mosaiks['Parish']).all()
        
        ## determine range of values to explore
        ps = np.linspace(merged_mosaiks['lognormal_dist'].min(), merged_mosaiks['lognormal_dist'].max(), 1000)
        
        for i, p in enumerate(ps):
            
            ## define ground-truth eligibility
            merged_mosaiks['prog_eligible'] = merged_mosaiks['{}_preds_subnat'.format(task)] <= p
            
            ## define model eligibility
            merged_mosaiks['model_eligible'] = merged_mosaiks['lognormal_dist'] <= p
            
            ## compute inclusion and exclusion errors
            counts = merged_mosaiks.groupby('Country').apply(calc_counts, true_col = 'prog_eligible', model_col = 'model_eligible')
            
            ## compute inclusion error
            counts['inc_err'] = counts['FP'] / (counts['TN'] + counts['FP'])
            counts['inc_err'] = counts['inc_err'].fillna(1)
            
            ## compute exclusion error
            counts['exc_err'] = counts['FN'] / (counts['TP'] + counts['FN'])
            counts['exc_err'] = counts['exc_err'].fillna(0)
            
            ## add task and p-th threshold 
            counts['task'] = task
            counts['p_threshold'] = i
            counts['bootstrap'] = b
            
            ## concatenate to dataframe
            model2_errors_b = pd.concat([model2_errors_b, counts[['inc_err', 'exc_err', 'task', 'p_threshold', 'bootstrap']]])
            
            ## remove recurring columns
            merged_mosaiks = merged_mosaiks.drop(columns = ['prog_eligible', 'model_eligible'])
        
        ## remove recurring columns
        merged_mosaiks = merged_mosaiks.drop(columns = ['Parish', 'lognormal_dist'])
    
    if (b + 1) % 5 == 0:
        print(f'{b + 1} iterations completed')

model2_errors_b = model2_errors_b.reset_index()

## compute mean and confidence interval
inc_err_means = model2_errors_b.groupby(['Country', 'task', 'p_threshold'])['inc_err'].mean().to_frame()
inc_err_cis = model2_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'inc_err', p = [2.5, 97.5]).to_frame()
inc_err_cis.columns = ['inc_err_ci']
exc_err_means = model2_errors_b.groupby(['Country', 'task', 'p_threshold'])['exc_err'].mean().to_frame()
exc_err_cis = model2_errors_b.groupby(['Country', 'task', 'p_threshold']).apply(calc_percentile, val_col = 'exc_err', p = [2.5, 97.5]).to_frame()
exc_err_cis.columns = ['exc_err_ci']

## merge 
model2_errors = inc_err_means.merge(inc_err_cis, left_index = True, right_index = True).merge(exc_err_means, left_index = True, right_index = True).merge(exc_err_cis, left_index = True, right_index = True)
model2_errors = model2_errors.reset_index().set_index('Country')

## Draw Models1-2

assert sorted(model1_errors.index.unique()) == sorted(model2_errors.index.unique())

for task in tasks:
    for country in np.unique(model1_errors.index):
        
        plt.close()
        
        ## draw inclusino and exclusion errors from models 
        plt.plot(model1_errors[(model1_errors.index == country) & (model1_errors.task == task)].inc_err, model1_errors[(model1_errors.index == country) & (model1_errors.task == task)].exc_err, label = 'Model 1')
        plt.plot(model2_errors[(model2_errors.index == country) & (model2_errors.task == task)].inc_err, model2_errors[(model2_errors.index == country) & (model2_errors.task == task)].exc_err, label = 'Model 2')
        
        ## label the graph
        plt.xlabel('Inclusion error')
        plt.ylabel('Exclusion error')
        plt.title(country)
        plt.legend()
        plt.savefig(os.path.join(c.out_dir, 'metrics', '{}_{}_abs_eic.png'.format(country.lower(), task)), bbox_inches = 'tight', pad_inches = 0.1)
