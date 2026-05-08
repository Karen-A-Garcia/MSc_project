import numpy as np 
import xarray as xr  
import os
import glob 

input_dir = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/6hourly/"
hus_paths = sorted(glob.glob(os.path.join(input_dir, "hus_6hrLev_CanESM5_historical_r1i1p2f1_gn_2000*.nc"))) 
hus_6hr = xr.open_mfdataset(hus_paths)

hus_6hr["plev"] = hus_6hr["ap"] + hus_6hr["b"] * hus_6hr["ps"] 

#Boolean mask of upper stratosphere i.e. where the pressure is less than 50hPa
upper_stratosphere = hus_6hr["plev"] <= 50*100
hus_upstrat = hus_6hr.where(upper_stratosphere)
dq_dt = hus_upstrat.diff(dim='time')
print(dq_dt)
# print(hus_upstrat['dq_dt'].max().values)
# print(hus_upstrat['dq_dt'].min().values)
