## This script cleans surveys data

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv, warnings
from mosaiks import transforms
from mosaiks.utils.imports import *
from shapely.geometry import Point

def match_income_poly(shp, income_gdf, col):
    
    ## spatially match MOSAIKS features with polygons
    sjoin = gpd.sjoin(income_gdf, shp, how = 'left', op ='within')
    sjoin = sjoin[sjoin[col].notna()]
    
    ## initialize dataframes
    income = pd.DataFrame([])
    
    ## loop over each index
    u, indices = np.unique(sjoin[col], return_inverse = True)
    for x in np.unique(indices):
        
        ## compute average for each polygon
        sjoin_group = sjoin.loc[indices == x]
        income_means = np.average(sjoin_group['income'], weights = sjoin_group['pop'], axis = 0)
        pop_means = np.average(sjoin_group['pop'], axis = 0)
        
        ## convert results to dataframe
        df = pd.DataFrame([income_means, pop_means]).transpose()
        df.columns = ['income', 'pop']
        df.insert(0, col, [u[x]])
        
        ## append 
        income = pd.concat([income, df])
    
    ## reindex 
    income = income.set_index(col)
    return income

## create folder if not exists
if not os.path.exists(os.path.join(c.data_dir, 'int', 'income')):
    os.makedirs(os.path.join(c.data_dir, 'int', 'income'))

#############################
## A) clean Barbados survey
#############################

## A-1. aggregate Barbados income to enumeration block

data = [['BRB.1_1', 'Christ Church'], ['BRB.2_1', 'Saint Andrew'], ['BRB.3_1', 'Saint George'], ['BRB.4_1', 'Saint James'], ['BRB.5_1', 'Saint John'], ['BRB.6_1', 'Saint Joseph'],
        ['BRB.7_1', 'Saint Lucy'], ['BRB.8_1', 'Saint Michael'], ['BRB.9_1', 'Saint Peter'], ['BRB.10_1', 'Saint Philip'], ['BRB.11_1', 'Saint Thomas']]
par_key = pd.DataFrame(data, columns = ['GID_1', 'NAME'])

## load Barbados data
bslc_hhid = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Barbados-Survey-of-Living-Conditions-2016', 'Data BSLC2016', 'RT001_Public.dta'))
bslc_income = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Barbados-Survey-of-Living-Conditions-2016', 'Data BSLC2016', 'RT002_Public.dta'))

## keep variables of interest
bslc_hhid = bslc_hhid[['hhid', 'par', 'psu', 'lat_cen', 'long_cen']]
bslc_hhid['par'] = bslc_hhid['par'].str.replace('St.', 'Saint')
bslc_hhid = bslc_hhid.merge(par_key, left_on = 'par', right_on = 'NAME')
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
bslc_hh = bslc_hhincome.merge(bslc_hhweight, left_on = 'hhid', right_on = 'hhid').merge(bslc_hhid[['hhid', 'GID_1', 'psu']], left_on = 'hhid', right_on = 'hhid')

## aggregate income to enumeration districts and compute average individual income
psu_tot_weight = bslc_hh[['psu', 'weight']].groupby('psu').agg(sum)
agg_income = bslc_hh.income.mul(bslc_hh.weight).groupby(bslc_hh['psu']).sum()
income = psu_tot_weight.merge(agg_income.rename('agg_income'), left_index = True, right_index = True)
income['avg_income'] = income['agg_income'] / income['weight']
income['ln_income'] = np.log(income['avg_income'] + 1)

## save income for future use
income.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_ed_income.pkl'))

## aggregate income to parish and compute average individual income
par_tot_weight = bslc_hh[['GID_1', 'weight']].groupby('GID_1').agg(sum)
agg_income = bslc_hh.income.mul(bslc_hh.weight).groupby(bslc_hh['GID_1']).sum()
income = par_tot_weight.merge(agg_income.rename('agg_income'), left_index = True, right_index = True)
income['avg_income'] = income['agg_income'] / income['weight']
income['ln_income'] = np.log(income['avg_income'] + 1)

## save income for future use
income.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_parish_income.pkl'))

## A-2. match income with MOSAIKS-level polygons

shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'brb_mosaiks_voronoi_poly.shp'))

## merge in household identifier and weights
bslc_hh_coords = bslc_hhincome.merge(bslc_hhweight, left_on = 'hhid', right_on = 'hhid').merge(bslc_hhid[['hhid', 'lat_cen', 'long_cen']], left_on = 'hhid', right_on = 'hhid')
bslc_hh_coords['weighted_income'] = bslc_hh_coords['income'] * bslc_hh_coords['weight']

## collapse down for each coordinate
bslc_coords = bslc_hh_coords[['weight', 'weighted_income', 'lat_cen', 'long_cen']].groupby(['lat_cen', 'long_cen']).agg(sum).reset_index()
bslc_coords = bslc_coords.rename(columns = {'weight': 'pop', 'weighted_income': 'income'})

## convert to geopandas framework
geometry = [Point(xy) for xy in zip(bslc_coords.long_cen, bslc_coords.lat_cen)]
bslc_gdf = gpd.GeoDataFrame(bslc_coords.drop(['lat_cen', 'long_cen'], axis = 1), crs = shp.crs, geometry = geometry)

## spatially match MOSAIKS polygons with income data
mosaiks_income = match_income_poly(shp, bslc_gdf, 'id')

## extract MOSAIKS polygons that are not matched with MOSAIKS features
unmatched_shp = shp.loc[~shp.id.isin(mosaiks_income.index)]

## create buffer around the centroid
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    buffers = gpd.GeoDataFrame(geometry = unmatched_shp.buffer(0.00833333333333))
    buffers = buffers.merge(unmatched_shp['id'], left_index = True, right_index = True)

## spatially match again
buffer_mosaiks_income = match_income_poly(buffers[['id', 'geometry']], bslc_gdf, 'id')

## concatenate and set index
mosaiks_all_income = pd.concat([mosaiks_income, buffer_mosaiks_income])

## compute average individual income 
mosaiks_all_income['avg_income'] = mosaiks_all_income['income'] / mosaiks_all_income['pop']
mosaiks_all_income['ln_income'] = np.log(mosaiks_all_income['avg_income'] + 1)

## merge in lat/lon from mosaiks 
clean_income = pd.merge(mosaiks_all_income, shp[['id', 'lat', 'lon']], left_index = True, right_on = 'id')
clean_income = clean_income.set_index(clean_income['lat'].astype(str) + ':' + clean_income['lon'].astype(str)).drop(columns = ['id', 'lat', 'lon'])
clean_income.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'brb_mosaiks_income.pkl'))

##############################
## B) clean St. Lucia census
##############################

## B-1. aggregate St. Lucia income to settlement block
data = [['LCA.1_1', 'LC04', 'ANSE-LA-RAYE'], ['LCA.2_1', 'LC05', 'CANARIES'], ['LCA.9_1', 'LC06', 'SOUFRIERE'], ['LCA.4_1', 'LC07', 'CHOISEUL'], ['LCA.7_1', 'LC08', 'LABORIE'], ['LCA.10_1', 'LC09', 'VIEUX-FORT'], ['LCA.8_1', 'LC10', 'MICOUD'], 
        ['LCA.5_1', 'LC11', 'DENNERY'], ['LCA.6_1', 'LC12', 'GROS-ISLET'], ['LCA.3_1', 'LC13', 'CASTRIES']]
dist_key = pd.DataFrame(data, columns = ['GID_1', 'adm1code', 'NAME'])

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

## aggregate household income to settlement level and compute average individual income
lca_2010_settle = lca_2010_hh.groupby(['adm1code', 'settlement'])[['hhincome', 'pop']].sum()
lca_2010_settle = lca_2010_settle.reset_index()
lca_2010_settle['avg_income'] = lca_2010_settle['hhincome'] / lca_2010_settle['pop']
lca_2010_settle['ln_income'] = np.log(lca_2010_settle['avg_income'] + 1)
lca_2010_settle = lca_2010_settle.set_index(lca_2010_settle['adm1code'] + lca_2010_settle['settlement'].apply(lambda x: '{0:0>9}'.format(x))).drop(columns = ['adm1code', 'settlement'])

## save income for future use
lca_2010_settle.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_settle_income.pkl'))

## aggregate household income to district level and compute average individual income
lca_2010_district = lca_2010_hh.groupby(['adm1code'])[['hhincome', 'pop']].sum()
lca_2010_district['avg_income'] = lca_2010_district['hhincome'] / lca_2010_district['pop']
lca_2010_district['ln_income'] = np.log(lca_2010_district['avg_income'] + 1)
lca_2010_district = lca_2010_district.merge(dist_key[['GID_1', 'adm1code']], left_index = True, right_on = 'adm1code').set_index('GID_1')

## save
lca_2010_district.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_district_income.pkl'))

## B-2. match income with MOSAIKS-level polygons

shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'lca_mosaiks_voronoi_poly.shp'))

## load settlement shapefile
settle_shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'lca_admbnda_gov_2019_shp', 'lca_admbnda_adm2_gov_2019.shp'))
settle_shp = settle_shp.dissolve(['ADM1_PCODE', 'SETTLECODE']).reset_index()
settle_shp['lon'] = settle_shp.centroid.x
settle_shp['lat'] = settle_shp.centroid.y

## merge in household identifier and weights
lca_2010_coords = pd.merge(lca_2010_hh, settle_shp[['ADM1_PCODE', 'SETTLECODE', 'lat', 'lon']], left_on = ['adm1code', 'settlement'], right_on = ['ADM1_PCODE', 'SETTLECODE'])

## collapse down for each coordiante
lca_coords = lca_2010_coords[['pop', 'hhincome', 'lat', 'lon']].groupby(['lat', 'lon']).agg(sum).reset_index()
lca_coords = lca_coords.rename(columns = {'hhincome': 'income'})

## convert to geopandas framework
geometry = [Point(xy) for xy in zip(lca_coords.lon, lca_coords.lat)]
lca_gdf = gpd.GeoDataFrame(lca_coords.drop(['lat', 'lon'], axis = 1), crs = shp.crs, geometry = geometry)

## spatially match MOSAIKS polygons with income data
mosaiks_income = match_income_poly(shp, lca_gdf, 'id')

## extract MOSAIKS polygons that are not matched with MOSAIKS features
unmatched_shp = shp.loc[~shp.id.isin(mosaiks_income.index)]

## create buffer around the centroid
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    buffers = gpd.GeoDataFrame(geometry = unmatched_shp.buffer(0.00833333333333))
    buffers = buffers.merge(unmatched_shp['id'], left_index = True, right_index = True)

## spatially match again
buffer_mosaiks_income = match_income_poly(buffers[['id', 'geometry']], lca_gdf, 'id')

## concatenate and set index
mosaiks_all_income = pd.concat([mosaiks_income, buffer_mosaiks_income])

## compute average individual income
mosaiks_all_income['avg_income'] = mosaiks_all_income['income'] / mosaiks_all_income['pop']
mosaiks_all_income['ln_income'] = np.log(mosaiks_all_income['avg_income'] + 1)

## merge in lat/lon from mosaiks 
clean_income = pd.merge(mosaiks_all_income, shp[['id', 'lat', 'lon']], left_index = True, right_on = 'id')
clean_income = clean_income.set_index(clean_income['lat'].astype(str) + ':' + clean_income['lon'].astype(str)).drop(columns = ['id', 'lat', 'lon'])
clean_income.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_mosaiks_income.pkl'))

## B-3. clean LCA labor survey

## load LCA data
lfs_2016 = pd.read_stata(os.path.join(c.data_dir, 'raw', 'surveys', 'Saint Lucia Census and Labor Survey', 'LFS', 'LCA_2016.dta'))
lfs_2016 = lfs_2016[['district', 'ed', 'wt', 'income']]

## recreate district variable
lfs_2016['NAME'] = lfs_2016.district.to_list()

## I can separate ANSE-LA-RAYE and CANARIES districts based on ed id's
## ed = 5200-6205 in Anse La Raye district / 6301-6504 in Canaries district
lfs_2016.loc[(lfs_2016['ed'] >= 5200) & (lfs_2016['ed'] <= 6205), 'NAME'] = 'ANSE-LA-RAYE'
lfs_2016.loc[(lfs_2016['ed'] >= 6301) & (lfs_2016['ed'] <= 6504), 'NAME'] = 'CANARIES'
lfs_2016.loc[(lfs_2016['district'] == 'CASTRIES CITY') | (lfs_2016['district'] == 'CASTRIES RURAL'), 'NAME'] = 'CASTRIES'
lfs_2016 = lfs_2016.drop(columns = ['district'])

## merge in district code
lfs_2016 = lfs_2016.merge(dist_key[['GID_1', 'NAME']], left_on = 'NAME', right_on = 'NAME')

## initialize the dataset at district level
district_income = pd.DataFrame([], index = np.unique(lfs_2016.GID_1))

## compute median individual income for each district
for index in district_income.index:
    
    ## extract income brackets and frequency for each bin
    income = np.unique(lfs_2016.loc[lfs_2016['GID_1'] == index, ['income']])
    freq = lfs_2016.loc[lfs_2016['GID_1'] == index, ['income', 'wt']].groupby(['income']).agg(sum).values
    freq = np.array([val for sublist in freq for val in sublist])
    
    ## compute total and median counts and initialize the bin index
    total_count = sum(freq)
    mid_count = total_count / 2
    cum_count = 0
    median_bin_idx = -1
    
    for i, f in enumerate(freq):
        cum_count += f
        if cum_count >= mid_count:
            median_bin_idx = i
            break
    
    ## adjust the bin index i fthe median is in the last bin
    if median_bin_idx == -1:
        median_bin_idx = len(income) - 1
    
    ## compute the median using linear interpolation
    left_edge = income[median_bin_idx]
    right_edge = income[median_bin_idx + 1]
    left_freq = freq[median_bin_idx]
    median = left_edge + ((mid_count - cum_count + left_freq) / left_freq) * (right_edge - left_edge)
    district_income.loc[district_income.index == index, ['income']] = median

## save
district_income['ln_income'] = np.log(district_income['income'])
district_income.to_pickle(os.path.join(c.data_dir, 'int', 'income', 'lca_district_median_income.pkl'))

