import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
import matplotlib.colors as mcolors 
import os
import glob

g = 9.81
R_ideal = 8.314               # Pa m^3 / (mol K)
m_dryair = 28.97 / 1000       # kg / mol
ref_press = 1013.25 * 100     # Pa
R_dry_air = 287.05            # J / (kg K)
chunk_div = 50
target_lapserate = 2
precip = 8
max_height_cap =70
year=2014

chunks = {"valid_time": 200,
          "latitude": 50,
          "longitude": 50}

temp_ERA = xr.open_dataset("/home/karengarcia/downloads-karengarcia/ERA5/Hourly/temperature/2014/Regridded/ERA5_temperature_2014_6hourly_CanAM5.nc", chunks=chunks)
cp = xr.open_dataset("/home/karengarcia/downloads-karengarcia/ERA5/Hourly/convective_precipitation/2014/Regridded/ERA5_convective_precipitation_2014_6hourly_CanAM5.nc", chunks=chunks)
cloud_ice = xr.open_mfdataset("/home/karengarcia/downloads-karengarcia/ERA5/Hourly/specific_cloud_ice_water_content/2014/Regridded/ERA5_specific_cloud_ice_water_content_2014_6hourly_CanAM5.nc", chunks=chunks)
cloud_liquid = xr.open_mfdataset("/home/karengarcia/downloads-karengarcia/ERA5/Hourly/specific_cloud_liquid_water_content/2014/Regridded/ERA5_specific_cloud_liquid_water_content_2014_6hourly_CanAM5.nc", chunks=chunks)

        
temp_ERA['pressure_level'] = (temp_ERA['pressure_level']).astype('float32')
temp_ERA['pressure_level'].encoding = {}

dT = temp_ERA.t.diff('pressure_level')
dP = temp_ERA.pressure_level.diff('pressure_level')

T_low = temp_ERA.t.isel(pressure_level=slice(0, -1))
T_high = temp_ERA.t.isel(pressure_level=slice(1, None))
T_mid = (T_low + T_high) / 2

P_low = temp_ERA.pressure_level.isel(pressure_level=slice(0, -1))
P_high = temp_ERA.pressure_level.isel(pressure_level=slice(1, None))
P_mid = (P_low + P_high) / 2

dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
lapse_rate = -(dT / dz) * 1000

####Adding new criteria 

#Lapse Rate Tropopause (LRT)
tropopause_mask = lapse_rate <= target_lapserate
lrt_index = tropopause_mask.argmax(dim='pressure_level').compute()
lrt_pressure = temp_ERA['pressure_level'].isel(pressure_level=lrt_index)

#COLD POINT TROPOPAUSE (CPT)
cpt_index = temp_ERA.t.argmin(dim='pressure_level').compute() 
cpt_pressure = temp_ERA['pressure_level'].isel(pressure_level=cpt_index)

index_diff = np.abs(cpt_index - lrt_index)
#New condition: If the CPT and LRT are more than LRT and meets height cap
use_cpt_condition = (index_diff >= 3)

true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
final_tp_index    = xr.where(use_cpt_condition, cpt_index, lrt_index)
print("Tropopause calculation successful")
#######

cloud_ice['pressure_level'] = (cloud_ice['pressure_level']).astype('float32')
cloud_ice['pressure_level'].encoding = {}

cloud_liquid['pressure_level'] = (cloud_liquid['pressure_level']).astype('float32')
cloud_liquid['pressure_level'].encoding = {}

cloud_total = cloud_ice.ciwc + cloud_liquid.clwc
nlev = temp_ERA.sizes["pressure_level"]
level_indices = xr.DataArray(np.arange(nlev),
                        dims=["pressure_level"],
                        coords={"pressure_level": temp_ERA.pressure_level})

above_tp = temp_ERA.pressure_level <= true_tropopause_p
above_tp = above_tp.broadcast_like(cloud_total)
cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 1e-8) & (cp['cp']*4000 >= 1e-3)
cloud_above_tp_and_prc = cloud_above_tp_and_prc.any(dim="pressure_level")
cloud_above_tp_and_prc = cloud_above_tp_and_prc.astype("int8")
cloud_above_tp_and_prc = cloud_above_tp_and_prc.to_dataset(name="overshoot")
cloud_above_tp_and_prc = cloud_above_tp_and_prc.assign_coords({"latitude": temp_ERA.lat,
                                                        "longitude": temp_ERA.lon})

for v in cloud_above_tp_and_prc.variables:
        cloud_above_tp_and_prc[v].encoding = {}

cloud_above_tp_and_prc.to_netcdf(f"/home/karengarcia/data-karengarcia/Overshooting/ERA5_coarse/1e-3/ERA_overshoot_{str(year)}.nc")
print(f"Files saved to /home/karengarcia/data-karengarcia/Overshooting/ERA5_coarse/1e-3/ERA_overshoot_{str(year)}.nc")
print("Done!")