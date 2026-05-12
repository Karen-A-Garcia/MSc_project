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
precip = 4
max_height_cap =70

chunks = {"valid_time": 584,
          "pressure_level": -1,
          "latitude": 360,
          "longitude": 360}

hours = ['0000', '0100', '0200', '0300', '0400' ,'0500','0600', '0700', '0800', '0900', '1000', '1100', '1200','1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
year = 2014

ERA_folder = "/home/karengarcia/downloads-karengarcia/ERA5/Hourly/"
# for hour in hours:
#         temp_file = sorted(glob.glob(os.path.join(ERA_folder,f"temperature/{str(year)}/ERA5_temperature_{str(year)}_*_Hourly_output.nc")))
#         cloud_ice_w_file = sorted(glob.glob(os.path.join(ERA_folder,f"specific_cloud_ice_water_content/{str(year)}/ERA5_specific_cloud_ice_water_content_{str(year)}_{str(hour)}_Hourly_output.nc")))
#         cloud_liquid_w_file = sorted(glob.glob(os.path.join(ERA_folder,f"specific_cloud_liquid_water_content/{str(year)}/ERA5_specific_cloud_liquid_water_content_{str(year)}_{str(hour)}_Hourly_output.nc")))
#         convective_precip_file = sorted(glob.glob(os.path.join(ERA_folder,f"convective_precipitation/{str(year)}/ERA5_convective_precipitation_{str(year)}_{str(hour)}_Hourly_output.nc")))

#         temp_ERA = xr.open_dataset(temp_file, chunks=chunks)
#         cp = xr.open_dataset(convective_precip_file, chunks=chunks)
#         cloud_ice = xr.open_dataset(cloud_ice_w_file, chunks=chunks)
#         cloud_liquid = xr.open_dataset(cloud_liquid_w_file, chunks=chunks)

#         #regridding (0.25x0.25 -> 2.5x2.5) Block averaging
#         temp_ERA = temp_ERA.coarsen(latitude=10, longitude=10, boundary=exact).mean()
#         cp = cp.coarsen(latitude=10, longitude=10, boundary=exact).mean()
#         cloud_ice = cloud_ice.coarsen(latitude=10, longitude=10, boundary=exact).mean()
#         cloud_liquid = cloud_liquid.coarsen(latitude=10, longitude=10, boundary=exact).mean()
        
#         temp_ERA['pressure_level'] = (temp_ERA['pressure_level']).astype('float32')
#         temp_ERA['pressure_level'].encoding = {}

#         dT = temp_ERA.t.diff('pressure_level')
#         dP = temp_ERA.pressure_level.diff('pressure_level')

#         T_low = temp_ERA.t.isel(pressure_level=slice(0, -1))
#         T_high = temp_ERA.t.isel(pressure_level=slice(1, None))
#         T_mid = (T_low + T_high) / 2

#         P_low = temp_ERA.pressure_level.isel(pressure_level=slice(0, -1))
#         P_high = temp_ERA.pressure_level.isel(pressure_level=slice(1, None))
#         P_mid = (P_low + P_high) / 2

#         dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
#         lapse_rate = -(dT / dz) * 1000

#         tropopause_mask = lapse_rate <= target_lapserate
#         tropopause_index = tropopause_mask.argmax(dim="pressure_level")

#         cloud_ice['pressure_level'] = (cloud_ice['pressure_level']).astype('float32')
#         cloud_ice['pressure_level'].encoding = {}

#         cloud_liquid['pressure_level'] = (cloud_liquid['pressure_level']).astype('float32')
#         cloud_liquid['pressure_level'].encoding = {}

#         cloud_total = cloud_ice.ciwc + cloud_liquid.clwc

#         nlev = temp_ERA.sizes["pressure_level"]

#         level_indices = xr.DataArray(np.arange(nlev),
#                                         dims=["pressure_level"],
#                                         coords={"pressure_level": temp_ERA.pressure_level})

#         above_tp = level_indices > tropopause_index
#         above_tp = above_tp.broadcast_like(cloud_total)
#         if precip == 8:   #8mm/day
#                 cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= 0.333/1000)
#         if precip == 4:   #4mm/day
#                 cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= 0.166/1000)
#         cloud_above_tp_and_prc = cloud_above_tp_and_prc.any(dim="pressure_level")

#         cloud_above_tp_and_prc = cloud_above_tp_and_prc.astype("int8")

#         cloud_above_tp_and_prc = cloud_above_tp_and_prc.to_dataset(name="overshoot")

#         cloud_above_tp_and_prc = cloud_above_tp_and_prc.assign_coords({"latitude": temp_ERA.latitude,
#                                                         "longitude": temp_ERA.longitude})

#         for v in cloud_above_tp_and_prc.variables:
#                 cloud_above_tp_and_prc[v].encoding = {}

#         cloud_above_tp_and_prc.to_netcdf(f"/home/karengarcia/data-karengarcia/Overshooting/ERA5/Hourly/{str(precip)}mm/ERA_overshoot_{str(year)}_{str(hour)}_{str(precip)}mm.nc")
#         print(f"Files saved to /home/karengarcia/data-karengarcia/Overshooting/ERA5/Hourly/{str(precip)}mm/ERA_overshoot_{str(year)}_{str(hour)}_{str(precip)}mm.nc")
#         print("Done!")
        
        
        
temp_file = sorted(glob.glob(os.path.join(ERA_folder,f"temperature/{str(year)}/ERA5_temperature_{str(year)}_*_Hourly_output.nc")))
cloud_ice_w_file = sorted(glob.glob(os.path.join(ERA_folder,f"specific_cloud_ice_water_content/{str(year)}/ERA5_specific_cloud_ice_water_content_{str(year)}_*_Hourly_output.nc")))
cloud_liquid_w_file = sorted(glob.glob(os.path.join(ERA_folder,f"specific_cloud_liquid_water_content/{str(year)}/ERA5_specific_cloud_liquid_water_content_{str(year)}_*_Hourly_output.nc")))
convective_precip_file = sorted(glob.glob(os.path.join(ERA_folder,f"convective_precipitation/{str(year)}/ERA5_convective_precipitation_{str(year)}_*_Hourly_output.nc")))

temp_ERA = xr.open_mfdataset(temp_file, chunks=chunks, combine='nested', concat_dim='valid_time')
cp = xr.open_mfdataset(convective_precip_file, combine='nested', concat_dim='valid_time')
cloud_ice = xr.open_mfdataset(cloud_ice_w_file, chunks=chunks, combine='nested', concat_dim='valid_time')
cloud_liquid = xr.open_mfdataset(cloud_liquid_w_file, chunks=chunks, combine='nested', concat_dim='valid_time')

#regridding (0.25x0.25 -> 2.5x2.5) Block averaging
temp_ERA = temp_ERA.coarsen(latitude=10, longitude=10, boundary="trim").mean()
cp = cp.coarsen(latitude=10, longitude=10, boundary="trim").mean()
cloud_ice = cloud_ice.coarsen(latitude=10, longitude=10, boundary="trim").mean()
cloud_liquid = cloud_liquid.coarsen(latitude=10, longitude=10, boundary="trim").mean()
        
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
use_cpt_condition = (index_diff >= 3) & (cpt_pressure >= max_height_cap) 

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
if precip == 8:   #8mm/day
        cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= 0.333/1000)
if precip == 4:   #4mm/day
        cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= 0.166/1000)
cloud_above_tp_and_prc = cloud_above_tp_and_prc.any(dim="pressure_level")

cloud_above_tp_and_prc = cloud_above_tp_and_prc.astype("int8")

cloud_above_tp_and_prc = cloud_above_tp_and_prc.to_dataset(name="overshoot")

cloud_above_tp_and_prc = cloud_above_tp_and_prc.assign_coords({"latitude": temp_ERA.latitude,
                                                        "longitude": temp_ERA.longitude})

for v in cloud_above_tp_and_prc.variables:
        cloud_above_tp_and_prc[v].encoding = {}

cloud_above_tp_and_prc.to_netcdf(f"/home/karengarcia/data-karengarcia/Overshooting/ERA5_coarse/{str(precip)}mm/ERA_overshoot_{str(year)}_{str(precip)}mm.nc")
print(f"Files saved to /home/karengarcia/data-karengarcia/Overshooting/ERA5_coarse/{str(precip)}mm/ERA_overshoot_{str(year)}_{str(precip)}mm.nc")
print("Done!")