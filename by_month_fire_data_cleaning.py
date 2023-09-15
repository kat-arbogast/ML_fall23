import pandas as pd

# Per Month Wildfire Data
jan_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_January.csv', skiprows=1)
feb_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_February.csv', skiprows=1)
mar_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_March.csv', skiprows=1)
apr_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_April.csv', skiprows=1)
may_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_May.csv', skiprows=1)
jun_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_June.csv', skiprows=1)
jul_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_July.csv', skiprows=1)
aug_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_August.csv', skiprows=1)
sep_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_September.csv', skiprows=1)
oct_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_October.csv', skiprows=1)
nov_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_November.csv', skiprows=1)
dec_burn = pd.read_csv('./DirtyData/Wildfire_Data/By_month_fires/US_Wildfires_December.csv', skiprows=1)

print(dec_burn.head())