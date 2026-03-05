import pandas as pd

# paths
sf1_path = "Data/sf1_reduced.csv"
sf2_path = "Data/sf2_reduced.csv"

# read csv and drop the original headers for new column names
dataframe1 = pd.read_csv(sf1_path)
dataframe2 = pd.read_csv(sf2_path)

# Change original column names for new column names
canonical_feat_labels = ["RecordID", 
                         "SuspectsRaceAsAGroup", 
                         "MostSeriousLocation", 
                         "MostSeriousBias", 
                         "MostSeriousBiasType"
                         ]
dataframe1.columns = canonical_feat_labels
dataframe2.columns = canonical_feat_labels

# Change Record Id's to have same formatting in case they're different (type string, remove whitespace)
dataframe1["RecordID"] = dataframe1["RecordID"].astype("string").str.strip()
dataframe2["RecordID"] = dataframe2["RecordID"].astype("string").str.strip()

# Combine both dataframes
combined = pd.concat([dataframe1, dataframe2], ignore_index = True)

# Remove any duplicates and only keed one listing of the duplicate case
combined = combined.drop_duplicates(subset = ["RecordID"], keep = "first").reset_index(drop = True)

# Print overlap counts to see
overlap = set(dataframe1["RecordID"]).intersection(set(dataframe2["RecordID"]))
print("Overlap RecordIDs:", len(overlap))
print("Dataframe1's rows:", len(dataframe1))
print("Dataframe2's rows:", len(dataframe2))
print("Combined rows:", len(combined))

# Outputted overlap counts
"""
Overlap RecordIDs: 1608
Dataframe1's rows: 1804
Dataframe2's rows: 32312
Combined rows: 32508
"""

# Write merged files
combined.to_csv("Data/SF_merged_data.csv", index = False)
print("Saved dataframe to Data/SF_merged_data.csv")