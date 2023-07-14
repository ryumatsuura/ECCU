## This script matches the global-scale grid data with MOSAIKS features

## set preambles
subset_n = slice(None)
subset_feat = slice(None)

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from pathlib import Path
from mosaiks import transforms
from mosaiks.utils.imports import *
from shapely.geometry import Point
from scipy.spatial import distance

## set output path for data
out_dir = Path(c.out_dir) / 'world'
out_path = Path(out_dir) / 'world_results_with_5foldcrossval.npz'

lambdas = lambdas_single = c.ml_model['global_lambdas']
solver = solve.ridge_regression
solver_kwargs = {'return_preds':True, 'svd_solve':False}

####################################
## A) load global mosaiks features
####################################

## A-1. load mosaiks features

for files in os.listdir(c.features_dir):
    if files.endswith('.zip'):
        print('Loading {}...'.format(files))
        zf = zipfile.ZipFile(os.path.join(c.features_dir, files))
        df = pd.read_csv(zf.open('Mosaiks_features.csv'), sep = ',')
        try:
            coords = np.append(coords, df.iloc[:, 0:2].values, axis = 0)
        except NameError:
            coords = df.iloc[:, 0:2].values
        try:
            features = np.append(features, df.iloc[:, 2:len(df.columns)].values, axis = 0)
        except NameError:
            features = df.iloc[:, 2:len(df.columns)].values

## create dataframes
X = pd.DataFrame(features)
latlons = pd.DataFrame(coords, columns = ['lat', 'lon'])

## drop all duplicates features
X = X.drop_duplicates(subset = [0, 1], keep = False)
latlons = latlons.reindex(X.index)

## convert to geopandas dataframe
gdf_mosaiks = gpd.GeoDataFrame(latlons, geometry = gpd.points_from_xy(latlons.lon, latlons.lat), crs = 'EPSG:4326')

## A-2. load global grid file

## load global grid file
grid_path = os.path.join(c.grid_dir, 'WORLD_16_640_UAR_1000000_0.npz')
npz = np.load(grid_path)

## extract grid 
lats = npz['lat']
lons = npz['lon']
ids = npz['ID']

## convert to geopandas dataframe
df_grid = gpd.GeoDataFrame({'X':lons, 'Y':lats})
df_grid['coords'] = list(zip(df_grid['X'], df_grid['Y']))
df_grid['coords'] = df_grid['coords'].apply(Point)
gdf_grid = gpd.GeoDataFrame(df_grid, geometry = 'coords', crs = 'EPSG:4326').set_index(ids)

## A-3. find nearest mosaik feature for each grid point

## intialize empty geopandas dataframe
indices = np.array([])

## loop each 10k observations in grid
obs = 10000
for i in range(int(gdf_grid.shape[0] / obs)):
    
    ## display the progress
    print('{} out of {} in progress...'.format(str(i), str(int(gdf_grid.shape[0] / obs))))
    
    ## define the first and last obs
    first = i * obs
    last  = (i + 1) * obs
    
    ## compute distance matrix - compute for each 10k obs in grid
    dist = distance.cdist(np.dstack([gdf_grid['X'].values, gdf_grid['Y'].values])[0][first:last], np.dstack([gdf_mosaiks['lon'].values, gdf_mosaiks['lat'].values])[0])
    
    ## extract Mosaiks features for each grid point and add horizontally
    indices = np.hstack((indices, gdf_mosaiks.iloc[dist.argmin(axis = 1)].index))

## match grid index with mosaiks features
mosaiks_feat = X.loc[indices]
grid_feat = mosaiks_feat.set_index(gdf_grid.index)
merged = grid_feat.join(gdf_grid[['X', 'Y']])

## save the cleaned MOSAIKS features
merged.to_pickle(os.path.join(c.data_dir, 'int', 'feature_matrices', 'grid_features.pkl'))
