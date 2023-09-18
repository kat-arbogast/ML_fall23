import pandas as pd


lightning_fires = pd.read_csv('./DirtyData/Wildfire_Data/lightining_fires_kaggle/US_Lightning_Forest_Fires.csv')

print(lightning_fires.info())

# removing the index columns and the FIPS since they had a lot of missing data
lightning_fires = lightning_fires.drop(columns=["Unnamed: 0", "index", "FIPS_CODE", "FIPS_NAME"])
lightning_fires["Fire_Date"] = pd.to_datetime(lightning_fires["Fire_Date"])
lightning_fires = lightning_fires.dropna()

print(lightning_fires.info())
print(lightning_fires.head(n=5))

lightning_fires.to_csv("./CleanData/lightning_wildfires_clean.csv", index = False)