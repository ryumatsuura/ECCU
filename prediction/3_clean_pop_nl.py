## This script creates population-weighted 
## NL features and population data

import os, shapely, fiona, warnings
import geopandas as gpd
from affine import Affine
import rasterio as rio
import rasterio.mask
from rasterio import warp
from mosaiks.utils.imports import *

## supress runtimewarning 
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = FutureWarning)

def apply_polygon_mask_and_return_flat_array(polygon, raster_file):
    out_image, out_transform = rasterio.mask.mask(raster_file, [polygon], crop = True, nodata = np.nan, pad = False, all_touched = False)
    array = out_image[0].flatten()
    array = array[~np.isnan(array)]
    try:
        return np.hstack(array)
    except:
        return np.array([0])

def create_nl_binned_dataframe(shp_file, raster_file, bins, weight_raster = None):
    
    ## create binned dataframe
    c = 1
    for index in shp_file.index:
        if c % 10000 == 0:
            print('{} out of {} layers are done'.format(c, shp_file.shape[0]))
        a = apply_polygon_mask_and_return_flat_array(shp_file['geometry'].at[index], raster_file = raster_file)
        w = None
        if weight_raster:
            w = apply_polygon_mask_and_return_flat_array(shp_file['geometry'].at[index], raster_file = weight_raster)
            assert a.shape == w.shape
        d = np.histogram(a, bins = bins, density = False, weights = w)
        if weight_raster:
            perc_in_each_bin = d[0] / w.sum()
        else:
            perc_in_each_bin = d[0] / len(a)
        if 'stacked' in locals():
            stacked = np.vstack([stacked, perc_in_each_bin])
        else:
            stacked = perc_in_each_bin
        c = c + 1
    
    ## clean the output
    if len(stacked.shape) == 2:
        out = pd.DataFrame(stacked, index = shp_file.index)
    elif len(stacked.shape) == 1:
        out = pd.DataFrame([stacked], index = shp_file.index)
    out.columns = ['perc_pixels_in_bin' + str(i) for i in range(out.shape[1])]
    return out 

def correct_nl_df_creation(out, shp_file, raster_file, bins):
    null_idxs = out[out.iloc[:, 0].isnull()].index
    num_missing = len(null_idxs)
    if num_missing == 0:
        return out
    
    ## create buffer around the centroid
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        buffers = shp_file.loc[null_idxs]['geometry'].centroid.buffer(0.00833333333333/2)
    
    ## run the correction
    for index in buffers.index:
        a = apply_polygon_mask_and_return_flat_array(buffers.at[index], raster_file = raster_file)
        assert len(a) <= 1
        if len(a) == 0:
            a = np.array([0])
        d = np.histogram(a, bins = bins, density = False, weights = None)
        perc_in_each_bin = d[0]
        if 'stacked' in locals():
            stacked = np.vstack([stacked, perc_in_each_bin])
        else:
            stacked = perc_in_each_bin
    
    ## clean the output
    if len(stacked.shape) == 2:
        fixed_out = pd.DataFrame(stacked, index = null_idxs)
    elif len(stacked.shape) == 1:
        fixed_out = pd.DataFrame([stacked], index = null_idxs)
    fixed_out.columns = ['perc_pixels_in_bin' + str(i) for i in range(out.shape[1])]
    out_dropped = out.drop(null_idxs)
    return pd.concat([fixed_out, out_dropped])

## define function for population computation
def compute_pop(shp, raster_file, cols):
    
    ## copy shapefile
    df = shp.copy()
    
    sums = []
    c = 1
    for index in shp.index:
        if c % 10000 == 0:
            print('{} out of {} layers are done'.format(c, shp.shape[0]))
        sums.append(apply_polygon_mask_and_return_flat_array(shp['geometry'].at[index], raster_file = src_tot).sum())
        c = c + 1
    
    ## add population and area
    df['total_pop'] = np.round(np.array(sums)).astype(int)
    df['area_sq_km'] = df.to_crs({'init': 'epsg:6933'})['geometry'].area / 1e6
    df['pop_density'] = df['total_pop'] / df['area_sq_km']
    df['ln_pop_density'] = np.log(df['pop_density'] + 1)
    all_cols = cols + ['total_pop', 'area_sq_km', 'pop_density', 'ln_pop_density']
    return df[all_cols]

## create folders if not exist
if not os.path.exists(os.path.join(c.data_dir, 'int', 'population')):
    os.makedirs(os.path.join(c.data_dir, 'int', 'population'))

if not os.path.exists(os.path.join(c.data_dir, 'int', 'nightlights')):
    os.makedirs(os.path.join(c.data_dir, 'int', 'nightlights'))  

###########################################
## A) match population data with polygons
###########################################

## load population density raster data
src_tot = rio.open(os.path.join(c.data_dir, 'raw', 'population', 'gpw_v4_population_count_rev11_2015_30_sec.tif'))
 
## A-1. global grid data

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'global_recs.shp'))

## compute population
df = compute_pop(shp, src_tot, ['lat', 'lon'])
df = df.set_index(df['lat'].astype(str) + ':' + df['lon'].astype(str)).drop(columns = ['lat', 'lon'])
df.to_pickle(os.path.join(c.data_dir, 'int', 'population', 'global_population.pkl'))

## A-2. ECCU countries at national and subnational level

## loop over ECCU countries shapefiles
for dirs in os.listdir(os.path.join(c.data_dir, 'raw', 'shp')):
    if (dirs.startswith('gadm41_')) & (dirs.endswith('_shp')):
        
        for files in os.listdir(os.path.join(c.data_dir, 'raw', 'shp', dirs)):
            
            ## national level
            if files.endswith('_0.shp'):
                
                ## load shapefile
                ISO = files.replace('gadm41_', '', 1).replace('_0.shp', '', 1)
                shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', dirs, files))
                
                ## compute population and append
                df = compute_pop(shp, src_tot, ['GID_0'])
                df = df.set_index('GID_0')
                df.to_pickle(os.path.join(c.data_dir, 'int', 'population', '{}_nat_population.pkl'.format(ISO.lower())))
            
            ## subnational level
            if files.endswith('_1.shp'):
                
                ## load shapefile
                ISO = files.replace('gadm41_', '', 1).replace('_1.shp', '', 1)
                shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', dirs, files))
                
                ## compute population and append
                df = compute_pop(shp, src_tot, ['GID_1'])
                df = df.set_index('GID_1')
                df.to_pickle(os.path.join(c.data_dir, 'int', 'population', '{}_subnat_population.pkl'.format(ISO.lower())))

## A-3. MOSAIKS polygons

## loop over all the MOSAIKS polygons files
for files in os.listdir(os.path.join(c.data_dir, 'int', 'shp')):
    if files.endswith('_mosaiks_voronoi_poly.shp'):
        
        ## load shapefile
        ISO = files.replace('_mosaiks_voronoi_poly.shp', '')
        shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', files))
        
        ## compute population
        df = compute_pop(shp, src_tot, ['lat', 'lon'])
        df = df.set_index(df['lat'].astype(str) + ':' + df['lon'].astype(str)).drop(columns = ['lat', 'lon'])
        df.to_pickle(os.path.join(c.data_dir, 'int', 'population', '{}_mosaiks_population.pkl'.format(ISO)))

## A-4. Barbados enumeration districts

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'brb_ed_voronoi_poly.shp'))

## compute population
df = compute_pop(shp, src_tot, ['psu'])
df = df.set_index('psu')
df.to_pickle(os.path.join(c.data_dir, 'int', 'population', 'brb_ed_population.pkl'))

## A-5. St. Lucia settlements

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'lca_admbnda_gov_2019_shp', 'lca_admbnda_adm2_gov_2019.shp'))
shp = shp.dissolve(['ADM1_PCODE', 'SETTLECODE']).reset_index()

## compute population
df = compute_pop(shp, src_tot, ['ADM1_PCODE', 'SETTLECODE'])
df['settlement'] = df['ADM1_PCODE'] + df['SETTLECODE'].apply(lambda x: '{0:0>9}'.format(x))
df = df.drop(columns = ['ADM1_PCODE', 'SETTLECODE']).set_index('settlement')
df.to_pickle(os.path.join(c.data_dir, 'int', 'population', 'lca_settle_population.pkl'))

##########################
## B) clean raster files 
##########################

## load nl raster data
src = rio.open(os.path.join(c.data_dir, 'raw', 'nightlights', 'F182013.v4c_web.stable_lights.avg_vis.tif'))
arr = src.read(1)
arr = arr.astype(np.float64)

## load population density raster data
pop_raster = rio.open(os.path.join(c.data_dir, 'raw', 'population', 'gpw_v4_population_density_rev10_2015_30_sec.tif'))
pop_arr = pop_raster.read(1)

## assume nin population density
min_data_value_of_pop_raster = pop_arr[pop_arr > 0].flatten().min()
pop_arr[pop_arr < 0] = min_data_value_of_pop_raster

## create new bounding box before we make adjustment
pop_raster_new_bounds = list(pop_raster.bounds)
src_new_bounds = list(src.bounds)

## adjust the bounding box so that rasters have the same height 
min_lon_of_pop_raster = pop_raster.bounds.bottom
new_max_lon = src.bounds.top
num_pixels_to_drop_from_top = (pop_raster.bounds.top - new_max_lon) / pop_raster.meta['transform'][0]
num_pixels_to_drop_from_top = int(np.floor(num_pixels_to_drop_from_top))

## keep track of the new upper bound on the raster
pop_raster_new_bounds[3] -= pop_raster.meta['transform'][0] * num_pixels_to_drop_from_top

pop_arr_crop = pop_arr[num_pixels_to_drop_from_top:, :]
min_lon_of_nl_raster = src.bounds.bottom
num_of_zero_pixels_to_add = int(np.floor((min_lon_of_pop_raster - min_lon_of_nl_raster)/pop_raster.meta['transform'][0]))

## assume minimum population density for all locations missing the value
zeros_to_stack_below = np.full((num_of_zero_pixels_to_add, pop_raster.meta['width']), min_data_value_of_pop_raster)
pop_arr_filled = np.vstack([pop_arr_crop, zeros_to_stack_below])

## keep track of the new lower bounds on the raster
pop_raster_new_bounds[1] += - num_of_zero_pixels_to_add * pop_raster.meta['transform'][0]

## clean longitudinal shift
arr_crop = arr[:,1:]
src_new_bounds[0] -=  src.transform[4]

pop_arr_filled = pop_arr_filled.astype(np.float64)
out_meta = pop_raster.meta.copy()
out_transform = Affine(out_meta['transform'][0], 0.0, src_new_bounds[0], 0.0, 
                       out_meta['transform'][4], src_new_bounds[3])

out_meta['transform'] = out_transform
out_meta['height'] = pop_arr_filled.shape[0]
out_meta['dtype'] = np.float64
_ = out_meta.pop('nodata')

## write out nl and population density raster data
dmsp_adj_to_pop_outpath = os.path.join(c.data_dir, 'int', 'nightlights', 'nl_shifted_to_match_pop_raster.tif')
with rio.open(dmsp_adj_to_pop_outpath , 'w', **out_meta) as dest:
    dest.write(np.array([arr_crop]))

pop_adj_to_nl_outpath = os.path.join(c.data_dir, 'int', 'population', 'pop_shifted_to_match_nl_raster.tif')
with rio.open(pop_adj_to_nl_outpath, 'w', **out_meta) as dest:
    dest.write(np.array([pop_arr_filled]))

###################################
## C) match NL data with polygons
###################################

## load adjusted raster files
nl_adj = rio.open(dmsp_adj_to_pop_outpath)
pop_adj = rio.open(pop_adj_to_nl_outpath)

## create bins for 20 features
bins = np.hstack([0, np.linspace(0.0, 63, 20)])

## C-1. global grid data

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'global_recs.shp'))

## create NL binned dataframe 
out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
new_out = correct_nl_df_creation(out, shp, nl_adj, bins)

## save the nighttime light data
nl = new_out.sort_index().merge(shp[['lat', 'lon']], left_index = True, right_index = True)
nl = nl.set_index(nl['lat'].astype(str) + ':' + nl['lon'].astype(str)).drop(columns = ['lat', 'lon'])
nl.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'global_nl.pkl'))

## C-2. ECCU countires at national and subnational level

## loop over ECCU countries shapefiles
for dirs in os.listdir(os.path.join(c.data_dir, 'raw', 'shp')):
    if (dirs.startswith('gadm41_')) & (dirs.endswith('_shp')):
        
        ## extract country code
        ISO = dirs.replace('gadm41_', '', 1).replace('_shp', '', 1).lower()
        
        ## national shapefile
        shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', dirs, 'gadm41_{}_0.shp'.format(ISO.upper())))
        
        ## create NL binned dataframe at national level
        out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
        new_out = correct_nl_df_creation(out, shp, nl_adj, bins)
        
        ## subnational shapefiles
        subnat_shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', dirs, 'gadm41_{}_1.shp'.format(ISO.upper())))
        
        ## create NL binned dataframe
        subnat_out = create_nl_binned_dataframe(subnat_shp, nl_adj, bins, pop_adj)
        subnat_new_out = correct_nl_df_creation(subnat_out, subnat_shp, nl_adj, bins)
        
        ## save the nighttime light data
        nl = new_out.sort_index().merge(shp['GID_0'], left_index = True, right_index = True)
        nl = nl.set_index('GID_0')
        nl.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_nat_nl.pkl'.format(ISO)))
        subnat_nl = subnat_new_out.sort_index().merge(subnat_shp['GID_1'], left_index = True, right_index = True)
        subnat_nl = subnat_nl.set_index('GID_1')
        subnat_nl.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_subnat_nl.pkl'.format(ISO)))
        
        ## clean national data
        feat_means_matrix = pd.DataFrame(np.resize(nl.values, (subnat_nl.shape[0], 20)), index = subnat_nl.index)
        feat_means_matrix.columns = ['perc_pixels_in_bin' + str(i) for i in range(nl.shape[1])]
        
        ## demean NL features
        nl_demean = subnat_nl - feat_means_matrix
        nl_demean.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_subnat_nl_demeaned.pkl'.format(ISO)))

## C-3. MOSAIKS polygons 

## loop over all the MOSAIKS polygons files
for files in os.listdir(os.path.join(c.data_dir, 'int', 'shp')):
    if files.endswith('_mosaiks_voronoi_poly.shp'):
        
        ## load shapefile
        ISO = files.replace('_mosaiks_voronoi_poly.shp', '').lower()
        shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', files))
        
        ## create NL binned dataframe 
        out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
        new_out = correct_nl_df_creation(out, shp, nl_adj, bins)
        
        ## save the nighttime light data
        nl = new_out.sort_index().merge(shp[['lat', 'lon']], left_index = True, right_index = True)
        nl = nl.set_index(nl['lat'].astype(str) + ':' + nl['lon'].astype(str)).drop(columns = ['lat', 'lon'])
        nl.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_mosaiks_nl.pkl'.format(ISO)))
        
        ## load national data
        nat_nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_nat_nl.pkl'.format(ISO)))
        
        ## clean national data
        feat_means_matrix = pd.DataFrame(np.resize(nat_nl.values, (nl.shape[0], 20)), index = nl.index)
        feat_means_matrix.columns = ['perc_pixels_in_bin' + str(i) for i in range(nat_nl.shape[1])]
        
        ## demean NL features
        nl_demean = nl - feat_means_matrix
        nl_demean.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', '{}_mosaiks_nl_demeaned.pkl'.format(ISO)))

## C-4. Barbados enumeration districts

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'brb_ed_voronoi_poly.shp'))

## create NL binned dataframe and run corrections
out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
new_out = correct_nl_df_creation(out, shp, nl_adj, bins)

## save the nighttime light data
nl = new_out.sort_index().merge(shp[['psu']], left_index = True, right_index = True)
nl = nl.set_index('psu')
nl.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'brb_ed_nl.pkl'))

## load national data
nat_nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'brb_nat_nl.pkl'.format(ISO)))
        
## clean national data
feat_means_matrix = pd.DataFrame(np.resize(nat_nl.values, (nl.shape[0], 20)), index = nl.index)
feat_means_matrix.columns = ['perc_pixels_in_bin' + str(i) for i in range(nat_nl.shape[1])]

## demean NL features
nl_demean = nl - feat_means_matrix
nl_demean.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'brb_ed_nl_demeaned.pkl'))

## C-5. St. Lucia settlements

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'lca_admbnda_gov_2019_shp', 'lca_admbnda_adm2_gov_2019.shp'))
shp = shp.dissolve(['ADM1_PCODE', 'SETTLECODE']).reset_index()

## create NL binned dataframe and run corrections
out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
new_out = correct_nl_df_creation(out, shp, nl_adj, bins)

## save the nighttime light data
nl = new_out.sort_index().merge(shp[['ADM1_PCODE', 'SETTLECODE']], left_index = True, right_index = True)
nl['settlement'] = nl['ADM1_PCODE'] + nl['SETTLECODE'].apply(lambda x: '{0:0>9}'.format(x))
nl = nl.drop(columns = ['ADM1_PCODE', 'SETTLECODE']).set_index('settlement')
nl.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'lca_settle_nl.pkl'))

## load national data
nat_nl = pd.read_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'brb_nat_nl.pkl'.format(ISO)))
        
## clean national data
feat_means_matrix = pd.DataFrame(np.resize(nat_nl.values, (nl.shape[0], 20)), index = nl.index)
feat_means_matrix.columns = ['perc_pixels_in_bin' + str(i) for i in range(nat_nl.shape[1])]

## demean NL features
nl_demean = nl - feat_means_matrix
nl_demean.to_pickle(os.path.join(c.data_dir, 'int', 'nightlights', 'lca_settle_nl_demeaned.pkl'))

