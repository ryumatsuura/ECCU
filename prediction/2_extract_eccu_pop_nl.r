## This script extracts national and subnational average of population 
## density and nighttime lights for ECCU countries and creates population 
## density and nighttime lights data in Barbados/Guadeloupe/Martinique

library(raster)
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

#############################
## A) extract population/nl
#############################

## load raster
pop = raster(file.path(data_dir, 'raw/applications/population/gpw_v4_population_density_rev10_2015_30_sec.tif'))
nl = raster(file.path(data_dir, 'int/applications/nightlights/global_nl.tif'))

## store folders with shapefiles
folders = list.dirs(file.path(data_dir, 'raw', 'shp'))[-1]

## loop over national/subnational, population/nightlight, and folders
for (l in c('nat', 'subnat')) {
    for (d in c('nl', 'pop')) {
        for (f in folders) {
            
            ## load shapefile
            if (l == 'nat') {
                shp_file = list.files(f, pattern = glob2rx('*_0.shp'))
            } else if (l == 'subnat') {
                shp_file = list.files(f, pattern = glob2rx('*_1.shp'))
            }    
            if (identical(shp_file, character(0)) || shp_file %in% c('gadm41_BRB_0.shp', 'gadm41_MTQ_0.shp', 'gadm41_GLP_0.shp', 'gadm41_BRB_1.shp', 'gadm41_MTQ_1.shp', 'gadm41_GLP_1.shp')) next
            shp = shapefile(file.path(f, shp_file))
            
            ## store the country ISO code
            if (l == 'nat') {
                ISO = gsub('_0.shp', '', gsub('gadm41_', '', shp_file))
            } else if (l == 'subnat') {
                ISO = gsub('_1.shp', '', gsub('gadm41_', '', shp_file))
            }
            
            ## crop the population/nl raster by bounding box
            e = extent(shp)
            delta = 0.1
            e@xmin = e@xmin - delta
            e@xmax = e@xmax + delta
            e@ymin = e@ymin - delta
            e@ymax = e@ymax + delta
            if (d == 'nl') {
                ras_crop = crop(nl, e)
            } else if (d == 'pop') {
                ras_crop = crop(pop, e)
            }
            
            ## extract raster value in each country polygon
            out = exact_extract(x = ras_crop, y = shp, fun = 'weighted_mean', weights = 'area', progress = FALSE)
            ln = log(out + 1)
            
            ## append data tables
            if (exists('eccu')) {
                if (d == 'nl') {
                    eccu = rbindlist(list(eccu, data.table('Country' = ISO, 'Nightlights' = ln)))
                } else if (d == 'pop') {
                    eccu = rbindlist(list(eccu, data.table('Country' = ISO, 'Population' = ln)))
                }
            } else {
                if (d == 'nl') {
                    eccu = data.table('Country' = ISO, 'Nightlights' = ln)
                } else if (d == 'pop') {
                    eccu = data.table('Country' = ISO, 'Population' = ln)
                }
            }
        }
        
        if (l == 'nat' && d == 'nl') {
            fn = file.path(data_dir, 'int/applications/nightlights/nightlights_eccu.csv')
        } else if (l == 'subnat' && d == 'nl') {
            fn = file.path(data_dir, 'int/applications/nightlights/nightlights_eccu_subnat.csv')
        } else if (l == 'nat' && d == 'pop') {
            fn = file.path(data_dir, 'int/applications/population/population_eccu.csv')
        } else if (l == 'subnat' && d == 'pop') {
            fn = file.path(data_dir, 'int/applications/population/population_eccu_subnat.csv')
        }
        write.csv(x = eccu, file = fn)
        rm(eccu)
    }
}

######################################################
## B) create population/nl for neighboring countries
######################################################

## loop over Barbados, Guadeloupe, and Martinique
for (c in c('brb', 'glp', 'mtq')) {
    
    ## load neighboring countries mosaiks feature
    fn = file.path(data_dir, paste0('int/feature_matrices/Mosaiks_features_', c, '.csv'))
    csv_file = read.csv(fn)
    
    ## extract lat/lon information
    Lats = c(csv_file[['Lat']])
    Lons = c(csv_file[['Lon']])
    
    ## load Barbados shapefile
    shp = shapefile(file.path(data_dir, paste0('raw/shp/gadm41_', toupper(c), '_shp/gadm41_', toupper(c), '_0.shp')))
    
    ## thiessen polygons
    v = voronoi(cbind(Lons, Lats))
    v_crop = intersect(v, shp)
    
    ## select coordinates that remain even after conversion to voronoi polygons
    coords = as.data.frame(cbind(v$id, Lons, Lats))
    colnames(coords) = c('id', 'lon', 'lat')
    coords_crop = merge(coords, v_crop, by = 'id')
    
    ## set the extent
    e = extent(v_crop)
    delta = 0.1
    e@xmin = e@xmin - delta
    e@xmax = e@xmax + delta
    e@ymin = e@ymin - delta
    e@ymax = e@ymax + delta
    
    ## loop over population/nl
    for (d in c('nl', 'pop')) {
        
        ## crop outcome variable
        if (d == 'nl') {
            ras_crop = crop(nl, e)
        } else if (d == 'pop') {
            ras_crop = crop(pop, e)
        }
        
        ## extract raster value in each voronoi polygon
        out = exact_extract(x = ras_crop, y = v_crop, fun = 'weighted_mean', weights = 'area', progress = FALSE)
        ln = log(out + 1)
        
        ## convert to dataframe
        df = as.data.frame(cbind(coords_crop$lon, coords_crop$lat, ln))
        if (d == 'nl') {
            colnames(df) = c('lon', 'lat', 'population')
        } else if (d == 'pop') {
            colnames(df) = c('lon', 'lat', 'nightlights')
        }
        
        ## save the label
        if (d == 'nl') {
            fn = file.path(data_dir, paste0('int/applications/nightlights/outcome_sampled_nightlights_', toupper(c), '.csv'))
        } else if (d == 'pop') {
            fn = file.path(data_dir, paste0('int/applications/population/outcome_sampled_population1_', toupper(c), '.csv'))
        }
        write.csv(x = df, file = fn)
    }
}

