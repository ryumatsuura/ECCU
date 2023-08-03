## This script matches the global-scale grid data with MOSAIKS features

## packages
import io as b_io
import geopandas as gpd
import rasterio as rio
import os, dill, rtree, zipfile, csv
from mosaiks import transforms
from mosaiks.utils.imports import *
from shapely.geometry import Point
from scipy.spatial import distance

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
gdf_mosaiks.merge(X, left_index = True, right_index = True).to_pickle(os.path.join(c.features_dir, 'grid_features.pkl'))

