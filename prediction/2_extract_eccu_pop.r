## This script extracts national and subnational average 
## of population density for ECCU countries
## and creates population density data in Barbados 

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

##################################
## A) extract population density 
##################################

## A-1. National-level

## load raster
pop = raster(file.path(data_dir, 'raw/applications/population/gpw_v4_population_density_rev10_2015_30_sec.tif'))

## loop over shapefile folders
folders = list.dirs(file.path(data_dir, 'raw', 'shp'))[-1]
for (f in folders) {
    
    ## load shapefile
    shp_file = list.files(f, pattern = glob2rx('*_0.shp'))
    if (identical(shp_file, character(0))) next
    shp = shapefile(file.path(f, shp_file))
    
    ## store the country ISO code
    ISO = gsub('_0.shp', '', gsub('gadm41_', '', shp_file))
    
    ## crop the population raster by bounding box
    e = extent(shp)
    delta = 0.1
    e@xmin = e@xmin - delta
    e@xmax = e@xmax + delta
    e@ymin = e@ymin - delta
    e@ymax = e@ymax + delta
    pop_crop = crop(pop, e)
    
    ## extract raster value in each country polygon
    out = exact_extract(x = pop_crop, y = shp, fun = 'weighted_mean', weights = 'area', progress = FALSE)
    ln = log(out + 1)
    
    ## save to csv
    if (exists('eccu')) {
        eccu = rbindlist(list(eccu, data.table('Country' = ISO, 'Population' = ln)))
    } else {
        eccu = data.table('Country' = ISO, 'Population' = ln)
    }
}

fn = file.path(data_dir, 'int/applications/population/population_density_eccu.csv')
write.csv(x = eccu, file = fn)

## A-2. Subnational-level

## loop over shapefile folders
for (f in folders) {
    
    ## load shapefile
    shp_file = list.files(f, pattern = glob2rx('*_1.shp'))
    if (identical(shp_file, character(0))) next
    shp = shapefile(file.path(f, shp_file))
    
    ## store the country ISO code
    ISO = gsub('_1.shp', '', gsub('gadm41_', '', shp_file))
    
    ## crop the population raster by bounding box
    e = extent(shp)
    delta = 0.1
    e@xmin = e@xmin - delta
    e@xmax = e@xmax + delta
    e@ymin = e@ymin - delta
    e@ymax = e@ymax + delta
    pop_crop = crop(pop, e)
    
    ## extract raster value in each country polygon
    out = exact_extract(x = pop_crop, y = shp, fun = 'weighted_mean', weights = 'area', progress = FALSE)
    ln = log(out + 1)
    
    ## stack to data table
    if (exists('eccu_subnat')) {
        eccu_subnat = rbindlist(list(eccu_subnat, data.table('Country' = ISO, 'Population' = ln, 'Name' = shp$NAME_1)))
    } else {
        eccu_subnat = data.table('Country' = ISO, 'Population' = ln, 'Name' = shp$NAME_1)
    }
}

fn = file.path(data_dir, 'int/applications/population/population_density_eccu_subnat.csv')
write.csv(x = eccu_subnat, file = fn)

###############################################
## B) create grid Barbados population density
###############################################

## B-1. load Barbados coordinates and create voronoi polygons

## load Barbados mosaiks feature
fn = file.path(data_dir, 'int/feature_matrices/Mosaiks_features_brb.csv')
brb = read.csv(fn)

## extract lat/lon information
Lats = c(brb[['Lat']])
Lons = c(brb[['Lon']])

## load Barbados shapefile
shp = shapefile(file.path(data_dir, 'raw/shp/gadm41_BRB_shp/gadm41_BRB_0.shp'))

## thiessen polygons
v = voronoi(cbind(Lons, Lats))
v_crop = intersect(v, shp)

## select coordinates that remain even after conversion to voronoi polygons
coords = as.data.frame(cbind(v$id, Lons, Lats))
colnames(coords) = c('id', 'lon', 'lat')
coords_crop = merge(coords, v_crop, by = 'id')

## B-2. extract population density for each polygon

## crop the population raster to the rectangle polygons
e = extent(v_crop)
delta = 0.1
e@xmin = e@xmin - delta
e@xmax = e@xmax + delta
e@ymin = e@ymin - delta
e@ymax = e@ymax + delta
pop_crop = crop(pop, e)

## extract raster value in each voronoi polygon
out = exact_extract(x = pop_crop, y = v_crop, fun = 'weighted_mean', weights = 'area', progress = FALSE)
ln = log(out + 1)

## convert to dataframe
df = as.data.frame(cbind(coords_crop$lon, coords_crop$lat, ln))
colnames(df) = c('lon', 'lat', 'population')

## save the label
fn = file.path(data_dir, 'int/applications/population/outcome_sampled_population_BRB.csv')
write.csv(x = df, file = fn)
