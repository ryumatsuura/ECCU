## extract population data for each 

library(raster)
library(reticulate)
library(data.table)
library(here)
library(rgdal)
library(gdalUtilities)
library(deldir)
library(dismo)
library(rgeos)
library(sf)
library(exactextractr)

rm(list = ls())

## import config.R to set filepaths
mosaiks_code = Sys.getenv('MOSAIKS_CODE')
if (mosaiks_code == '') {
    mosaiks_code = here('codes')
}
source(file.path(mosaiks_code, 'mosaiks', 'config.R'))

## source the necessary helper files
source(file.path(utils_dir, 'R_utils.R'))

crs = CRS(as.character('+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0'))

##################################
## A) extract population density
##################################

## load raster files
pop = raster(file.path(data_dir, 'raw/applications/population/gpw_v4_population_density_rev10_2015_30_sec.tif'))

## load grid data matched with MOSAIKS features
pd = import('pandas')
pickle_data = pd$read_pickle(file.path(data_dir, 'int', 'feature_matrices', 'grid_features.pkl'))

## keep latitude and longitude
lats = as.numeric(unlist(c(pickle_data['lat'])))
lons = as.numeric(unlist(c(pickle_data['lon'])))

## create rectangle around each point
## load global grid file
np = import('numpy')
grid_path = file.path(data_dir, 'int', 'grids', 'WORLD_16_640_UAR_1000000_0.npz')
npz = np$load(grid_path)
zoom = npz$f[['zoom']]
pixels = npz$f[['pixels']]

## create rectangles 
recs = centroidsToTiles(lat = lats, lon = lons, zoom = zoom, numPix = pixels)
recs_df = as(recs, 'SpatialPolygonsDataFrame')
recs_df$lat = lats
recs_df$lon = lons

## save the rectangles shapefile
writeOGR(obj = recs_df[, (2:3)], dsn = file.path(data_dir, 'int/shp'), layer = 'recs_global', driver = 'ESRI Shapefile', overwrite_layer = TRUE)

## set the extent
e = extent(recs)
delta = 0.1
e@xmin = e@xmin - delta
e@xmax = e@xmax + delta
e@ymin = e@ymin - delta
e@ymax = e@ymax + delta

## crop the population raster by bounding box
ras_crop = crop(pop, e)

## extract mean population raster value in each rectangle
out = exact_extract(x = ras_crop, y = recs, fun = 'weighted_mean', weights = 'area', progress = TRUE)
ln = log(out + 1)

## convert to data table
df = cbind(lons, lats, ln)
colnames(df) = c('lon', 'lat', 'population')
dt = data.table(df)

## output global populatin data
fn = file.path(data_dir, paste0('int/applications/population/outcome_sampled_population_global.csv'))
write.csv(x = df, file = fn)

