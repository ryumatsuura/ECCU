README

The following files implement the prediction exercises:

0_generate_latlons_csv.py: 
    This script extracts lat/lon information from global-scale grid data
    and the output will be inputted to the MOSAIKS file API

1_clean_global_mosaiks_features.py:
    This script matches global-scale grid data and MOSAIKS features

2_extract_weights_vector.py:
    This script trains the prediction model for population density 
    and recovers the weights vector

extract_eccu_population.R:
    This script computes the national-average and subnational-average population 
    density in each ECCU country from the raster data and creates grid-level
    population density data for Barbados

match_mosaiks_population.R:
    This script matches MOSAIKS feature coordinates with population count data
    and compute total population in each MOSAIKS feature coordinate

3_predict_eccu.py:
    This script extracts the weights vector and predicts the population density in ECCU countries

