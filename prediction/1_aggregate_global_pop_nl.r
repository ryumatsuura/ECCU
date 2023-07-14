## This script aggregates population and NL raster values
## to national and subnational levels at global scale

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

## load subnational shapefile
f = file.path(data_dir, 'raw/shp/GDL Shapefiles V4/GDL Shapefiles V4.shp')
shp = readOGR(f)

## store all the ISO codes to loop over
iso_codes = unique(shp$iso_code)

## loop over population density and nighttime light
for (d in c('nl', 'pop')) {
    c = 0
    for (i in iso_codes) {
        
        ## print out progress
        if (c %% 10 == 0) print(paste(c, 'out of', length(iso_codes), 'countries is done'))
        
        ## skip country with missing ISO
        if (is.na(i)) next
        
        ## extract each country
        country = subset(shp, iso_code == i)
        
        ## aggregate shapefile to national level
        agg = gUnaryUnion(country, country$iso_code)
        
        ## extract centroids
        point = gCentroid(agg, byid = TRUE)
        points = gCentroid(country, byid = TRUE)
        
        ## convert polygon to data frame
        spdf = SpatialPolygonsDataFrame(agg, data.frame(iso_code = i), match.ID = F)
        spdf$X = point$x
        spdf$Y = point$y
        
        ## crop the population/nl raster by bounding box
        e = extent(spdf)
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
        
        ## extract mean population/nl raster value in each country polygon
        out = exact_extract(x = ras_crop, y = spdf, fun = 'weighted_mean', weights = 'area', progress = FALSE)
        if (d == 'nl') {
            spdf$Nightlights = log(out + 1)
        } else if (d == 'pop') {
            spdf$Population = log(out + 1)
        }
        
        ## append spatial data frames
        if (exists('pop_ln_out')) {
            pop_ln_out = rbind(pop_ln_out, spdf)
        } else {
            pop_ln_out = spdf
        }
        
        ## extract mean population/nl raster value at subnational level
        subnat_out = exact_extract(x = ras_crop, y = country, fun = 'weighted_mean', weights = 'area', progress = FALSE)
        ln = log(subnat_out + 1)
        
        ## append spatial data frames
        if (exists('pop_ln_subnat_out')) {
            if (d == 'nl') {
                pop_ln_subnat_out = rbind(pop_ln_subnat_out, data.table('GDLCODE' = country$GDLcode, 'Country' = country$iso_code, 'X' = points$x, 'Y' = points$y, 'Nightlights' = ln))
            } else if (d == 'pop') {
                pop_ln_subnat_out = rbind(pop_ln_subnat_out, data.table('GDLCODE' = country$GDLcode, 'Country' = country$iso_code, 'X' = points$x, 'Y' = points$y, 'Population' = ln))
            }
        } else {
            if (d == 'nl') {
                pop_ln_subnat_out = data.table('GDLCODE' = country$GDLcode, 'Country' = country$iso_code, 'X' = points$x, 'Y' = points$y, 'Nightlights' = ln)
            } else if (d == 'pop') {
                pop_ln_subnat_out = data.table('GDLCODE' = country$GDLcode, 'Country' = country$iso_code, 'X' = points$x, 'Y' = points$y, 'Population' = ln)
            }
        }
        
        ## increment the counter
        c = c + 1
    }
    
    ## save population/nl into csv
    if (d == 'nl') {
        fn = file.path(data_dir, 'int/applications/nightlights/nightlights_global.csv')
    } else if (d == 'pop') {
        fn = file.path(data_dir, 'int/applications/population/population_global.csv')
    }
    write.csv(x = pop_ln_out, file = fn)
    
    ## save population/nl into csv
    if (d == 'nl') {
        fn = file.path(data_dir, 'int/applications/nightlights/nightlights_global_subnat.csv')
    } else if (d == 'pop') {
        fn = file.path(data_dir, 'int/applications/population/population_global_subnat.csv')
    }
    write.csv(x = pop_ln_subnat_out, file = fn)
    
    ## remove output objects
    rm(pop_ln_out, pop_ln_subnat_out)
}

