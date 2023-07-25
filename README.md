# ECCU codes

This git repository stores files that train the prediction model and predict poverty and asset measures in ECCU countries

- 0_generate_latlons_csv.py: 
    - This script extracts lat/lon information from global-scale grid data and the output will be inputted to the MOSAIKS file API

- 1_clean_global_mosaiks_features.py:
    - This script matches global-scale grid data and MOSAIKS features

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

