## This script matches MOSAIKS features coordinates 
## and population data for aggregation exercises

library(raster)
library(foreach)
library(doParallel)
library(reticulate)
library(data.table)
library(here)
library(rgdal)
library(deldir)
library(dismo)
library(rgeos)
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

##############################################
## A) match MOSAIKS features with population
##############################################

## load raster
pop = raster(file.path(data_dir, 'raw/applications/population/gpw_v4_population_count_rev11_2015_30_sec.tif'))

## loop over MOSAIKS files
files = list.files(file.path(data_dir, 'int/feature_matrices'), pattern = glob2rx('Mosaiks_features_*.csv'))
for (f in files) {
    
    ## load MOSAIKS feature
    fn = file.path(data_dir, 'int/feature_matrices', f)
    mosaiks_feat = read.csv(fn)
    
    ## store country name
    ISO = toupper(gsub('.csv', '', gsub('Mosaiks_features_', '', f)))
    
    ## extract lat/lon information
    Lats = c(mosaiks_feat[['Lat']])
    Lons = c(mosaiks_feat[['Lon']])
    
    ## load shapefile of each country
    shp = shapefile(file.path(data_dir, 'raw/shp', paste0('gadm41_', ISO, '_shp'), paste0('gadm41_', ISO, '_0.shp')))
    
    ## thiessen polygons
    v = voronoi(cbind(Lons, Lats))
    v_crop = intersect(v, shp)
    
    ## select coordinates that remain even after conversion to voronoi polygons
    coords = as.data.frame(cbind(v$id, Lons, Lats))
    colnames(coords) = c('id', 'lon', 'lat')
    coords_crop = merge(coords, v_crop, by = 'id')
    
    ## crop the population raster to the voronoi polygons
    e = extent(v_crop)
    delta = 0.1
    e@xmin = e@xmin - delta
    e@xmax = e@xmax + delta
    e@ymin = e@ymin - delta
    e@ymax = e@ymax + delta
    pop_crop = crop(pop, e)
    
    ## extract raster value in each voronoi polygon
    out = exact_extract(x = pop_crop, y = v_crop, fun = 'sum', progress = FALSE)
    
    ## convert to dataframe
    df = as.data.frame(cbind(coords_crop$lon, coords_crop$lat, out))
    colnames(df) = c('Lon', 'Lat', 'population')
    
    ## save the MOSAIKS feature level population
    fn = file.path(data_dir, 'int/population', paste0('pop_Mosaiks_features_', tolower(ISO), '.csv'))
    write.csv(x = df, file = fn)
}
