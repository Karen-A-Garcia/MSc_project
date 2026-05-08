from scipy.stats import skew, kurtosis
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cftime
from netCDF4 import Dataset 
import glob 
from datetime import date, timedelta

ERA_monthly_file = "/home/karengarcia/downloads-karengarcia/ERA5/ERA5_specific_hum.nc"
CanAM_monthly_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Monthly/hus_Amon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc"
ERA_monthly = xr.open_dataset(ERA_monthly_file).q
CanAM_monthly = (xr.open_dataset(CanAM_monthly_file).hus).isel(time=slice(-11*12,None))

years = [str(year) for year in np.arange(2004,2015,1)]

full_water=[]
full_time=[]

for year in years:
    for file_path in glob.glob(f"/home/karengarcia/downloads-karengarcia/MLS_data/*{year}.nc"):
        with Dataset(file_path, 'r') as ds:
            
            data =      ds.groups['H2O PressureGrid'] 
            # potK_data = ds.groups["H2O ThetaGrid"] 
            #grabbing data
            MLS_lon =   data.variables['lon'][:]             #units = "degrees_east" (72 values)
            MLS_lat =   data.variables['lat'][:]             #units = "degrees_north" (45 values)
            time =      data.variables['time'][:]            #units = "days since 1950-01-01" (12 values)
            water =     data.variables["value"][:]           #value(time, lev, lon, lat)
            pres =      data.variables["lev"][:]             #units = "hPa" (values 45) Pressure
            
            # changing time from dates into date times because it hates me apparently
            dates = []
            start = date(1950,1,1) #starting date
            for t in time:  
                delta = timedelta(days=float(t))
                dates.append(start + delta)
            full_water.append(water)  #making an array with all the time values in datetime
            full_time.append(dates)

time_flat = np.concatenate(full_time, axis=0)
MLS_monthly = np.concatenate(full_water, axis=0)  # shape: (total_time, lev, lon, lat) 

#100hPA 
CanAM_100 = CanAM_monthly.sel(plev=100*100)
# display(CanAM_100)
ERA5_100 = ERA_monthly.sel(pressure_level=100)
MLS_100 = MLS_monthly[:,12,:,:]

#70hPa 
CanAM_70 = CanAM_monthly.sel(plev=70*100)
# display(CanAM_100)
ERA5_70 = ERA_monthly.sel(pressure_level=70)
MLS_70 = MLS_monthly[:,12,:,:] #68.1292hPa

#50hPa 
CanAM_50 = CanAM_monthly.sel(plev=50*100)
ERA5_50 = ERA_monthly.sel(pressure_level=50)
MLS_50 = MLS_monthly[:,15,:,:] 

#CanAM Stats 
mean_CanAM_q_100 = CanAM_100.mean(dim='time')
var_CanAM_q100 = CanAM_100.var(dim='time')
sk_CanAM_q_100 = skew(CanAM_100, keepdims = True)
kurt_CanAM_q_100 = kurtosis(CanAM_100, keepdims = True) 

mean_CanAM_q_70 = CanAM_70.mean(dim='time')
var_CanAM_q_70 = CanAM_70.var(dim='time')
sk_CanAM_q_70 = skew(CanAM_70, keepdims = True)
kurt_CanAM_q_70 = kurtosis(CanAM_70, keepdims = True) 

mean_CanAM_q_50 = CanAM_50.mean(dim='time')
var_CanAM_q_50 = CanAM_50.var(dim='time')
sk_CanAM_q_50 = skew(CanAM_50, keepdims = True)
kurt_CanAM_q_50 = kurtosis(CanAM_50, keepdims = True) 


#ERA Stats 
mean_ERA_q_100 = ERA5_100.mean(dim='valid_time')
var_ERA_q100 = ERA5_100.var(dim='valid_time')
sk_ERA_q_100 = skew(ERA5_100, keepdims = True)
kurt_ERA_q_100 = kurtosis(ERA5_100, keepdims = True) 

mean_ERA_q_70 = ERA5_70.mean(dim='valid_time')
var_ERA_q_70 = ERA5_70.var(dim='valid_time')
sk_ERA_q_70 = skew(ERA5_70, keepdims = True)
kurt_CanAM_q_70 = kurtosis(ERA5_70, keepdims = True) 

mean_ERA_q_50 = ERA5_50.mean(dim='valid_time')
var_ERA_q_50 = ERA5_50.var(dim='valid_time')
sk_ERa_q_50 = skew(ERA5_50, keepdims = True)
kurt_ERA_q_50 = kurtosis(ERA5_50, keepdims = True)

#MLS Stats
mean_MLS_q_100 = np.mean(MLS_100, axis =0) 
var_MLS_q_100 = np.var(MLS_100, axis =0) 
sk_MLS_q_100 = skew(MLS_100, axis =0) 
kurt_MLS_q_100 = kurtosis(MLS_100, axis =0) 

mean_MLS_q_70 = np.mean(MLS_70, axis =0) 
var_MLS_q_70 = np.var(MLS_70, axis =0)
sk_MLS_q_70 = skew(MLS_70, axis =0) 
kurt_MLS_q_70 = kurtosis(MLS_70, axis =0)  

mean_MLS_q_50  = np.mean(MLS_50, axis =0) 
var_MLS_q_50 = np.var(MLS_50, axis =0) 
sk_MLS_q_50 = skew(MLS_50, axis =0) 
kurt_MLS_q_50 = kurtosis(MLS_50, axis =0) 

lon_CAN = CanAM_100.lon
lat_CAN = CanAM_100.lat 

lat_ERA = ERA5_100.latitude 
lon_ERA = ERA5_100.longitude

lon_MLS = MLS_lon 
lat_MLS = MLS_lat 



vmin = 1e-15
vmax = 1e-11
norm = colors.LogNorm(vmin=vmin, vmax=vmax)
levels = np.logspace(np.log10(vmin), np.log10(vmax))
proj = ccrs.PlateCarree(central_longitude=0)
fig, axes = plt.subplots(nrows=3,ncols=4, figsize = (24,12) ,subplot_kw={'projection': proj})

axes[0,0].contourf(lon_MLS, lat_MLS, mean_MLS_q_100.T)
axes[1,0].contourf(lon_ERA, lat_ERA, mean_ERA_q_100)
axes[2,0].contourf(lon_CAN, lat_CAN, mean_CanAM_q_100)

axes[0,1].contourf(lon_MLS, lat_MLS, var_MLS_q_100.T)
axes[1,1].contourf(lon_ERA, lat_ERA, var_ERA_q_100)
axes[2,1].contourf(lon_CAN, lat_CAN, var_CanAM_q_100)

# axes[0,2].contourf(lon_MLS, lat_MLS, sk_MLS_q_100.T)
# axes[1,2].contourf(lon_ERA, lat_ERA, sk_ERA_q_100)
# axes[2,2].contourf(lon_CAN, lat_CAN, sk_CanAM_q_100)