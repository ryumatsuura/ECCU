## This script creates voronoi polygons for Barbados enumeration districts

library(raster)
library(reticulate)
library(data.table)
library(here)
library(rgdal)
library(deldir)
library(dismo)
library(rgeos)
library(readstata13)
library(sf)

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

##################################################
## A) create voronoi polygons for ED in Barbados
##################################################

## read BRB labor condition survey to load coordinates of enumeration districts
bslc = read.dta13(file.path(data_dir, 'raw/surveys/Barbados-Survey-of-Living-Conditions-2016/Data BSLC2016/RT001_Public.dta'))
bslc = bslc[, c('psu', 'lat_cen', 'long_cen')]

## extract lat/lon for each enumeration districts
bslc_ed = bslc[!duplicated(bslc), ]

## extract lat/lon information
Lats = c(bslc_ed[['lat_cen']])
Lons = c(bslc_ed[['long_cen']])

## load shapefile of Barbados
shp = shapefile(file.path(data_dir, 'raw/shp/gadm41_BRB_shp/gadm41_BRB_0.shp'))

## thiessen polygons
v = voronoi(cbind(Lons, Lats))
v_crop = intersect(v, shp)

## merge in psu id into voronoi polygons
v_crop_sf = st_as_sf(v_crop['COUNTRY'])
bslc_sf = st_as_sf(bslc_ed, coords = c('long_cen', 'lat_cen'))
psu_poly = st_join(v_crop_sf, bslc_sf)

## export voronoi polygons
st_write(psu_poly, file.path(data_dir, 'int/shp'), 'BRB_voronoi_poly', driver = 'ESRI Shapefile', append = FALSE, delete_layer = TRUE)

