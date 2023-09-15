import sqlite3
import pandas as pd


def database_connection(relative_path, table_name):
    cnx = sqlite3.connect(relative_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", cnx)

    cnx.close()
    
    return df


# Wildfire Data
us_wildfires_2mil = database_connection('./Wildfire_Data/US_2mil_wildfires_kaggle/FPA_FOD_20221014.sqlite', 'Fires')
ca_wildfires = pd.read_csv('./Wildfire_Data/ca_wildfires_kaggle/California_Fire_Incidents.csv')
or_weather_wildfires = pd.read_csv('./Wildfire_Data/oregon_wildfire_kaggle/Wildfire_Weather_Merged_new.csv',
                                   dtype={
                                        'Cause_Comments' : 'str',
                                        'DistFireNumber' : 'str'
                                    })
lightning_fires = pd.read_csv('./Wildfire_Data/lightining_fires_kaggle/US_Lightning_Forest_Fires.csv')

# Per Month Wildfire Data
jan_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_January.csv', skiprows=1)
feb_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_February.csv', skiprows=1)
mar_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_March.csv', skiprows=1)
apr_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_April.csv', skiprows=1)
may_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_May.csv', skiprows=1)
jun_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_June.csv', skiprows=1)
jul_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_July.csv', skiprows=1)
aug_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_August.csv', skiprows=1)
sep_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_September.csv', skiprows=1)
oct_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_October.csv', skiprows=1)
nov_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_November.csv', skiprows=1)
dec_burn = pd.read_csv('./Wildfire_Data/By_month_fires/US_Wildfires_December.csv', skiprows=1)


# Climate Change Kaggle Data
global_temp_by_city = pd.read_csv('./Weather_drought_data/climate_change_kaggle/GlobalLandTemperaturesByCity.csv')
global_temp_by_country = pd.read_csv('./Weather_drought_data/climate_change_kaggle/GlobalLandTemperaturesByCountry.csv')
global_temp_by_major_city = pd.read_csv('./Weather_drought_data/climate_change_kaggle/GlobalLandTemperaturesByMajorCity.csv')
global_temp_by_state = pd.read_csv('./Weather_drought_data/climate_change_kaggle/GlobalLandTemperaturesByState.csv')
global_temp_avg = pd.read_csv('./Weather_drought_data/climate_change_kaggle/GlobalTemperatures.csv')


# Hourly Weather Data
weather_city_attributes = pd.read_csv('./Weather_drought_data/hour_weather_data/city_attributes.csv')
weeather_humidity = pd.read_csv('./Weather_drought_data/hour_weather_data/humidity.csv')
weather_pressure = pd.read_csv('./Weather_drought_data/hour_weather_data/pressure.csv')
weather_temperature = pd.read_csv('./Weather_drought_data/hour_weather_data/temperature.csv')
weather_description = pd.read_csv('./Weather_drought_data/hour_weather_data/weather_description.csv')
weather_wind_direction = pd.read_csv('./Weather_drought_data/hour_weather_data/wind_direction.csv')
weather_wind_speed = pd.read_csv('./Weather_drought_data/hour_weather_data/wind_speed.csv')


# Drought Montior Data
dm_national_total_area = pd.read_csv('./Weather_drought_data/drought_monitor_data/dm_export_20000101_20230101-national-totalArea-catergorical.csv')
dm_regional_total_area = pd.read_csv('./Weather_drought_data/drought_monitor_data/dm_export_20000101_20230101-region-totalArea-catergorical.csv')
dm_state_total_area = pd.read_csv('./Weather_drought_data/drought_monitor_data/dm_export_20000101_20230101-state-totalArea-catergorical.csv')
dm_RDEWS_total_area = pd.read_csv('./Weather_drought_data/drought_monitor_data/dm_export_20000101_20230101_RDEWS_totalArea.csv')




