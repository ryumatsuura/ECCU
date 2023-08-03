## create shapefiles

library(reticulate)
library(data.table)
library(here)
library(rgdal)
library(raster)
library(rgeos)
library(dismo)
library(deldir)
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

crs = CRS(as.character('+proj=longlat +datum=WGS84 +no_defs'))

## create folder if not exists
if (!dir.exists(file.path(data_dir, 'int/shp'))) {
    dir.create(file.path(data_dir, 'int/shp'))
}

#######################################################
## A) create rectangles around global MOSAIKS feature
#######################################################

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
writeOGR(obj = recs_df[, (2:3)], dsn = file.path(data_dir, 'int/shp'), layer = 'global_recs', driver = 'ESRI Shapefile', overwrite_layer = TRUE)

##################################################
## B) create voronoi polygons for ED in Barbados
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
psu_poly = psu_poly[!is.na(psu_poly$psu), ]
st_crs(psu_poly) = crs

## export voronoi polygons
st_write(psu_poly, file.path(data_dir, 'int/shp'), 'brb_ed_voronoi_poly', driver = 'ESRI Shapefile', append = FALSE, delete_layer = TRUE)

####################################################
## C) create voronoi polygons for MOSAIKS features
####################################################

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
    v$lat = Lats
    v$lon = Lons
    v_crop = intersect(v, shp)
    proj4string(v_crop) = crs
    
    ## save the thiessen polygons
    writeOGR(obj = v_crop, dsn = file.path(data_dir, 'int/shp'), layer = paste0(tolower(ISO), '_mosaiks_voronoi_poly'), driver = 'ESRI Shapefile', overwrite_layer = TRUE)
}

