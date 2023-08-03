# ECCU codes

This git repository stores files that train the prediction model and predict poverty and asset measures in ECCU countries

- 0_generate_latlons_csv.py: 
    - This script extracts lat/lon information from global-scale grid data and the output will be inputted to the MOSAIKS file API
    - Input: data/int/grids/WORLD_16_640_UAR_1000000_0.npz
    - Output: data/int/feature_matrices/latlons*.csv

- 1_clean_global_mosaiks_features_coords.py:
    - This script cleans global-scale grid MOSAIKS features
    - Input: data/int/feature_matrices/latlons*.zip
    - Output: data/int/feature_matrices/grid_features.pkl

- 2_create_shps.R:
    - This script creates rectangles around global-scale grid MOSAIKS features and voronoi polygons
    - Input: data/int/feature_matrices/grid_features.pkl & data/raw/survey/Barbados-Survey-of-Living-Condition-2016/Data BSLC2016/RT001_Public.dta & data/int/feature_matrices/Mosaiks_features_*.csv
    - Output: data/int/shp/global_recs.shp & data/int/shp/brb_ed_voronoi_poly.shp & data/int/shp/*_mosaiks_voronoi_poly.shp

- 3_clean_pop_nl.py:
    - This script matches pop and NL data with shapefile
    - Input: data/raw/nightlights/...tif & data/raw/population/gpw_v4_population_density...tif & data/int/shp/*.shp
    - Output: data/int/nightlights/*_nl*.pkl & data/int/population/*_population.pkl

- 4_aggregate_mosaiks_features.py:
    - This script aggregates MOSAIKS features to the level of analysis
    - Input: data/int/features_matrices/Mosaiks_features_*.csv & data/raw/shp/gadm41_*_shp/gadm41_*_1.shp & data/int/population/*_mosaiks_population.pkl
    - Output: data/int/features_matrices/*_mosaiks_features.pkl

- 5_clean_surveys.py:
    - This script cleans surveys and census data
    - Input: data/raw/surveys/Barbados-Survey-of-Living-Conditions-2016/Data BSLC2016/*.dta & data/raw/surveys/Saint Lucia Census and Labor Survey/2010 Census Dataset/person_house_merged.dta
    - Output: data/int/income/brb_ed_income.pkl & data/int/income/lca_settle_income.pkl

- 6_extract_weights_vector*.py:
    - This script trains the prediction model and extracts the weights

- 7_predict_*.py
    - This script predicts the variables of interest

- 8_descriptive_stats.py
    - This script shows descriptive statistics


- 1_aggregate_global_pop.R:
    - This script aggregates global population density to national and subnational levels

- 2_extract_eccu_pop.R:
    - This script computes the national-average and subnational-average population density in each ECCU country from the raster data and creates grid-level population density data for neighboring countries

- 3_match_mosaiks_pop.R:
    - This script matches MOSAIKS feature coordinates with population count data and computes total population in each MOSAIKS feature coordinate in ECCU countries

- 4_create_voronoi_poly.R:
    - This script creates voronoi polygons for Barbados enumeration districts

- 2_extract_weights_vector_*.py:
    - This script trains the prediction model for different outcome variables and recovers the weights vector

- 3_aggregate_mosaiks_features.py
    - This script aggregates MOSAIKS features to national and subnational levels

- 4_predict_eccu_*.py:
    - This script extracts the weights vector and predicts different outcome variables in ECCU countries

