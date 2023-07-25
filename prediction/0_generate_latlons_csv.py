## This script generates csv file with lat/lon information from global-scale grid data

## package
import os 
import numpy as np
import pandas as pd
from mosaiks import config as c

## load global grid file
grid_path = os.path.join(c.grid_dir, 'WORLD_16_640_UAR_1000000_0.npz')
npz = np.load(grid_path)

## split 1m obs into 10 files 
## since only 100k obs are downloadable
for i in range(10):
    
    ## define the first and last obs
    first = i * 100000
    last = (i + 1) * 100000
    
    ## extract 100k observations in each loop
    lats = npz['lat'][first:last]
    lons = npz['lon'][first:last]
    latlons = np.dstack([lats, lons])
    
    ## convert to dataframe and export as csv
    df = pd.DataFrame(latlons[0], columns = ['Latitude', 'Longitude'])
    df.to_csv(os.path.join(c.features_dir, 'latlons{}.csv'.format(i)), index = False, sep = ',')

