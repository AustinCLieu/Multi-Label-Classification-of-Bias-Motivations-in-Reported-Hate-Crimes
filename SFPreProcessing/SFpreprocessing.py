import pandas as pd

# paths
sf_data1 = "Data/SFdata.govdata.csv"
sf_data2 = "Data/SFgov.orgdata.csv"

# Features we want
feat1_wanted = ["record_id", "suspects_race_as_a_group", "most_serious_location", "most_serious_bias", "most_serious_bias_type"]
feat2_wanted = ["RecordId", "SuspectsRaceAsAGroup", "MostSeriousUCR", "MostSeriousLocation", "MostSeriousBias", "MostSeriousBiasType"]

# read full files
dataframe1 = pd.read_csv(sf_data1)
dataframe2 = pd.read_csv(sf_data2)

# keep only the columns that we want in each dataframe
cols1 = []
cols2 = []
for feat in feat1_wanted:
    if feat in dataframe1.columns:
        cols1.append(feat)
for feat in feat2_wanted:
    if feat in dataframe2.columns:
        cols2.append(feat)

# Get reduced dataset with only the features we want
dataframe1_reduced = dataframe1[cols1].copy()
dataframe2_reduced = dataframe2[cols2].copy()

# Write reduced files
dataframe1_reduced.to_csv("sf1_reduced.csv", index = False)
dataframe2_reduced.to_csv("sf2_reduced.csv", index = False)

print("Saved sf1_reduced.csv:", dataframe1_reduced.shape)
print("saved sf2_reduced.csv:", dataframe2_reduced.shape)

# Make sure files have the same exact name/label for each category

# Merge both datasets together
