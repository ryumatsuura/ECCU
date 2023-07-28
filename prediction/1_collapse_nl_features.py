## This script creates population-weighted 
## NL features

import os, shapely, fiona, warnings
import geopandas as gpd
from affine import Affine
import rasterio as rio
import rasterio.mask
from rasterio import warp
from mosaiks.utils.imports import *

## supress runtimewarning 
warnings.filterwarnings('ignore', category = RuntimeWarning)

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
    out = pd.DataFrame(stacked, index = shp_file.index)
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
    fixed_out = pd.DataFrame(stacked, index = null_idxs)
    fixed_out.columns = ['perc_pixels_in_bin' + str(i) for i in range(out.shape[1])]
    out_dropped = out.drop(null_idxs)
    return pd.concat([fixed_out, out_dropped])

#########################
## A) clean raste files 
#########################

## load nl raster data
src = rio.open(os.path.join(c.data_dir, 'raw', 'applications', 'nightlights', 'F182013.v4c_web.stable_lights.avg_vis.tif'))
arr = src.read(1)
arr = arr.astype(np.float64)

## load population density raster data
pop_raster = rio.open(os.path.join(c.data_dir, 'raw', 'applications', 'population', 'gpw_v4_population_density_rev10_2015_30_sec.tif'))
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
dmsp_adj_to_pop_outpath = os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'nl_shifted_to_match_pop_raster.tif')
with rio.open(dmsp_adj_to_pop_outpath , 'w', **out_meta) as dest:
    dest.write(np.array([arr_crop]))

pop_adj_to_nl_outpath = os.path.join(c.data_dir, 'int', 'applications', 'population', 'pop_shifted_to_match_nl_raster.tif')
with rio.open(pop_adj_to_nl_outpath, 'w', **out_meta) as dest:
    dest.write(np.array([pop_arr_filled]))


########################
## B) collapse NL data
########################

## load adjusted raster files
nl_adj = rio.open(dmsp_adj_to_pop_outpath)
pop_adj = rio.open(pop_adj_to_nl_outpath)

## create bins for 20 features
bins = np.hstack([0, np.linspace(0.0, 63, 20)])

## B-1. global grid data

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'recs_global.shp'))

## create NL binned dataframe 
out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
new_out = correct_nl_df_creation(out, shp, nl_adj, bins)

## save the nighttime light data
new_out.sort_index().merge(shp[['lat', 'lon']], left_index = True, right_index = True).to_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'global_nl_features_pop_weighted.pkl'))

## B-2. neighboring ocuntries

for country in ['brb', 'glp', 'mtq']:
    
    ## load shapefile
    shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', '{}_mosaiks_voronoi_poly.shp'.format(country.upper())))
    
    ## create NL binned dataframe 
    out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
    
    ## run corrections
    new_out = correct_nl_df_creation(out, shp, nl_adj, bins)
    
    ## save the nighttime light data
    new_out.sort_index().merge(shp[['lat', 'lon']], left_index = True, right_index = True).to_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', '{}_nl_features_pop_weighted.pkl'.format(country)))

## B-3. Barbados enumeration district

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'int', 'shp', 'BRB_voronoi_poly.shp'))

## create NL binned dataframe and run corrections
out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
new_out = correct_nl_df_creation(out, shp, nl_adj, bins)

## save the nighttime light data
new_out.sort_index().merge(shp[['psu']], left_index = True, right_index = True).to_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'brb_ed_nl_features_pop_weighted.pkl'))

## B-4. St. Lucia settlement

## load shapefile
shp = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'lca_admbnda_gov_2019_shp', 'lca_admbnda_adm2_gov_2019.shp'))

## create NL binned dataframe and run corrections
out = create_nl_binned_dataframe(shp, nl_adj, bins, pop_adj)
new_out = correct_nl_df_creation(out, shp, nl_adj, bins)

## save the nighttime light data
new_out.sort_index().merge(shp[['ADM1_PCODE', 'SETTLECODE']], left_index = True, right_index = True).to_pickle(os.path.join(c.data_dir, 'int', 'applications', 'nightlights', 'lca_settle_nl_features_pop_weighted.pkl'))

"""

gpdf = pd.read_pickle(os.path.join(c.data_dir, 'int', 'shp', 'HDI_ADM1_shapefile_clean.p'))

## create NL binned dataframe 
c = 1
for index in gpdf.index:
    if c % 10 == 0:
        print('{} out of {} layers are done'.format(c, gpdf.shape[0]))
    a = apply_polygon_mask_and_return_flat_array(gpdf['geometry'].at[index], raster_file = nl_adj)
    w = apply_polygon_mask_and_return_flat_array(gpdf['geometry'].at[index], raster_file = pop_adj)
    assert a.shape == w.shape
    d = np.histogram(a, bins = bins, density = False, weights = w)
    perc_in_each_bin = d[0] / w.sum()
    if 'stacked' in locals():
        stacked = np.vstack([stacked, perc_in_each_bin])
    else:
        stacked = perc_in_each_bin
    c = c + 1

## clean the output
out = pd.DataFrame(stacked, index = gpdf.index)
out.columns = ['NL_' + str(i) for i in range(out.shape[1])]




gpdf = gpd.read_file(os.path.join(c.data_dir, 'raw', 'shp', 'GDL Shapefiles V4', 'GDL Shapefiles V4.shp'))
gpdf.set_index('GDLcode', inplace=True)
gpdf.loc['BHRt', 'iso_code'] = 'BHR'
gpdf.loc['MLTt', 'iso_code'] = 'MLT' 
gpdf.reset_index(inplace=True)
gpdf.dropna(subset = ['GDLcode'], inplace=True)

gpdf_country = gpdf.dissolve('iso_code')
gpdf_country.to_pickle(os.path.join(c.data_dir, 'int', 'shp', 'HDI_ADM0_dissolved_shapefile.p'))

indicators = pd.read_csv(os.path.join(c.data_dir, 'raw', 'GDL', 'GDL-Indicators-(2018)-data.csv'))
indices =  pd.read_csv(os.path.join(c.data_dir, 'raw', 'GDL', 'GDL-Indices-(2018)-data.csv'))

data = pd.concat([indicators, indices.iloc[:,5:]], axis=1)
national_data_only_indices = data.groupby('ISO_Code').count()['Country']==1
national_data_only = data.groupby('ISO_Code').first()[national_data_only_indices].reset_index()
subnational_data_only = data[data['Region'] != 'Total']
df = pd.concat([national_data_only, subnational_data_only])

nats_dropped = national_data_only[~national_data_only['GDLCODE'].isin(gpdf['GDLcode'])]
subnats_dropped = subnational_data_only[~subnational_data_only.GDLCODE.isin(gpdf.GDLcode)]
n_dropped = len(nats_dropped) + len(subnats_dropped)

df.set_index('GDLCODE', inplace=True)
gpdf.set_index('GDLcode', inplace=True)

matching_locs = df.index[df.index.isin(gpdf.index)]

df = df.loc[matching_locs]
gpdf = gpdf.loc[matching_locs]

gpdf.to_pickle(os.path.join(c.data_dir, 'int', 'shp', 'HDI_ADM1_shapefile_clean.p'))
"""