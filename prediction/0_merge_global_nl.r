## This script merge separate NL raster files into 
## single global NL raster file

library(raster)
library(reticulate)
library(data.table)
library(here)
library(rgdal)
library(gdalUtilities)
library(deldir)
library(dismo)
library(rgeos)
library(terra)

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

#############################
## A) merge NL raster files
#############################

## load raster files
ras_00N060E = rast(file.path(data_dir, 'raw/applications/nightlights/SVDNB_npp_20150101-20151231_00N060E_vcm-ntl_v10_c201701311200.avg_rade9.tif'))
ras_00N060W = rast(file.path(data_dir, 'raw/applications/nightlights/SVDNB_npp_20150101-20151231_00N060W_vcm-ntl_v10_c201701311200.avg_rade9.tif'))
ras_00N180W = rast(file.path(data_dir, 'raw/applications/nightlights/SVDNB_npp_20150101-20151231_00N180W_vcm-ntl_v10_c201701311200.avg_rade9.tif'))
ras_75N060E = rast(file.path(data_dir, 'raw/applications/nightlights/SVDNB_npp_20150101-20151231_75N060E_vcm-ntl_v10_c201701311200.avg_rade9.tif'))
ras_75N060W = rast(file.path(data_dir, 'raw/applications/nightlights/SVDNB_npp_20150101-20151231_75N060W_vcm-ntl_v10_c201701311200.avg_rade9.tif'))
ras_75N180W = rast(file.path(data_dir, 'raw/applications/nightlights/SVDNB_npp_20150101-20151231_75N180W_vcm-ntl_v10_c201701311200.avg_rade9.tif'))

## merge all raster files
values(ras_00N060E) = 1:ncell(ras_00N060E)
values(ras_00N060W) = 1:ncell(ras_00N060W)
values(ras_00N180W) = 1:ncell(ras_00N180W)
values(ras_75N060E) = 1:ncell(ras_75N060E)
values(ras_75N060W) = 1:ncell(ras_75N060W)
values(ras_75N180W) = 1:ncell(ras_75N180W)

## codes may crush so merge three files each 
r_list = list(ras_00N060E, ras_00N060W, ras_00N180W)
rsrc = sprc(r_list)
merged_raster_00N = terra::merge(rsrc)

r_list = list(ras_75N060E, ras_75N060W, ras_75N180W)
rsrc = sprc(r_list)
merged_raster_75N = terra::merge(rsrc)

## remove original raster files to free up storage
rm(ras_00N060E, ras_00N060W, ras_00N180W, ras_75N060E, ras_75N060W, ras_75N180W)

## merge last files
rsrc = sprc(list(merged_raster_00N, merged_raster_75N))
merged_raster = terra::merge(rsrc)

## save global raster file
fn = file.path(data_dir, 'int/applications/nightlights/global_nl.tif')
terra::writeRaster(merged_raster, file = fn, overwrite = TRUE)

