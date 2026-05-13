import numpy as np
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

g =                9.81
R_ideal =          8.314
m_dryair =         28.97/1000
ref_press =        1013.25
R_dry_air =        287.05
target_lapserate = 2
precip_threshold = 4
year             = 2014

chunks = {'time':250,
          'latitude': 360,
          'longitude': 360}

####### UTLS Boundaries #######
top_boundary = 50
bottom_boundary = 400
max_height_cap = 70
###################################
input_dir  = "downloads-karengarcia/ERA5/Hourly"
temp_paths = sorted(glob.glob(os.path.join(input_dir, f'temperature/2014/Regridded/ERA5_temperature_2014_6hourly_CanAM5.nc')))
clw_paths  = sorted(glob.glob(os.path.join(input_dir, f'specific_cloud_liquid_water_content/2014/Regridded/ERA5_specific_cloud_liquid_water_content_2014_6hourly_CanAM5.nc')))
cic_paths  = sorted(glob.glob(os.path.join(input_dir, f'specific_cloud_ice_water_content/2014/Regridded/ERA5_specific_cloud_ice_water_content_2014_6hourly_CanAM5.nc')))
prc_paths  = sorted(glob.glob(os.path.join(input_dir, f'convective_precipitation/2014/Regridded/ERA5_convective_precipitation_2014_6hourly_CanAM5.nc')))

temp_ERA = xr.open_mfdataset(temp_paths, chunks = chunks)
clw_ERA = xr.open_mfdataset(clw_paths, chunks=chunks)
cic_ERA = xr.open_mfdataset(cic_paths, chunks=chunks)
prc_ERA = xr.open_mfdataset(prc_paths, chunks=chunks)

#### Tropopause ####
# UTLS mask
plev_mask = ((temp_ERA["pressure_level"] >= top_boundary) &
            (temp_ERA["pressure_level"] <= bottom_boundary)).compute()

UTLS_ta = temp_ERA.where(plev_mask, drop=True)

dT = UTLS_ta.t.diff("pressure_level")
dP = UTLS_ta["pressure_level"].diff("pressure_level")

T_low  = UTLS_ta.t.isel(pressure_level=slice(0, -1))
T_high = UTLS_ta.t.isel(pressure_level=slice(1, None)).assign_coords(pressure_level=T_low.pressure_level)
T_mid  = (T_low + T_high) / 2

P_low  = UTLS_ta["pressure_level"].isel(pressure_level=slice(0, -1))
P_high = UTLS_ta["pressure_level"].isel(pressure_level=slice(1, None)).assign_coords(pressure_level=P_low.pressure_level)
P_mid  = (P_low + P_high) / 2

# Hydrostatic dz
dz = -(R_dry_air * T_mid) / (g * P_mid*100) * dP*100
lapse_rate = - (dT / dz) * 1000

# LAPSE RATE TROPOPAUSE (LRT)
tropopause_mask     = lapse_rate <= target_lapserate
lrt_index = tropopause_mask.argmax(dim="pressure_level").compute()
lrt_pressure = UTLS_ta["pressure_level"].isel(pressure_level=lrt_index)

# COLD POINT TROPOPAUSE (CPT)
cpt_index = UTLS_ta.t.argmin(dim="pressure_level").compute()
cpt_pressure = UTLS_ta["pressure_level"].isel(pressure_level=cpt_index)

index_diff = np.abs(cpt_index - lrt_index)
# Condition: If CPT is significantly different from LRT and meets height cap
use_cpt_condition = (index_diff >= 3) & (cpt_pressure >= max_height_cap)

# Final Tropopause Pressure and Index selection
true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
final_tp_index = xr.where(use_cpt_condition, cpt_index, lrt_index)
print("Tropopause calculation successful")
#### Overshooting Calculation ####
cloud_total         = cic_ERA.ciwc + clw_ERA.clwc
nlev                = temp_ERA.sizes["pressure_level"]
level_indices       = xr.DataArray(np.arange(nlev),
                            dims = ["pressure_level"],
                            coords ={"pressure_level": temp_ERA.pressure_level})

above_tp = temp_ERA.pressure_level <= true_tropopause_p
above_tp = above_tp.broadcast_like(cloud_total)
ice_above_trop = (cloud_total.where(above_tp)).sum(dim="pressure_level")
print("Max:",ice_above_trop.max().values, "kg/kg")
print("Min:",ice_above_trop.min().values, "kg/kg")

################ Three criteria: ################
# 1. Total cloud (sum of cic and clw) above the tropopause is bigger than zero
# 2. Precipitation threshold must be met (mm/day)
# 5. Total cloud (sum of cic and clw) above the tropopause is bigger than 10^-5 kg/kg

#option 1
ice_threshold = 0
above_tp = ((ice_above_trop > 0))
above_tp = above_tp.astype('int8')
above_tp = above_tp.to_dataset(name='Option_1')
above_tp = above_tp.assign_coords({"lon": cic_ERA.lon,
                                "lat": cic_ERA.lat})
for v in above_tp.variables:
    above_tp[v].encoding = {}
output_path = f'/home/karengarcia/criteria_testing/Option_1/ERA5_overshoot_option1_{str(year)}.nc'
above_tp.to_netcdf(output_path)
print(f"File saved to", output_path)

#option 2
precip1 = 4
prc4mm = prc_ERA["cp"]*24000 >= precip1 #Convert (meter/hour to mm/day)
prc4mm = prc4mm.astype('int8')
prc4mm = prc4mm.to_dataset(name='Option_2')
prc4mm = prc4mm.assign_coords({"lon": cic_ERA.lon,
                                "lat": cic_ERA.lat})

for v in prc4mm.variables:
    prc4mm[v].encoding = {}

output_path = f'/home/karengarcia/criteria_testing/Option_2/ERA5_overshoot_option2_{str(year)}.nc'
prc4mm.to_netcdf(output_path)
print(f"File saved to", output_path)

#option 3
precip2 = 8
prc8mm = prc_ERA["cp"]*24000 >= precip2 #Convert (meter/hour to mm/day)
prc8mm = prc8mm.astype('int8')
prc8mm = prc8mm.to_dataset(name='Option_3')
prc8mm = prc8mm.assign_coords({"lon": cic_ERA.lon,
                                "lat": cic_ERA.lat})

for v in prc8mm.variables:
    prc8mm[v].encoding = {}

output_path = f'/home/karengarcia/criteria_testing/Option_3/ERA5_overshoot_option3_{str(year)}.nc'
prc8mm.to_netcdf(output_path)
print(f"File saved to", output_path)

#option 12
above_tp_and_prc4 = (ice_above_trop > 0) & \
                (prc_ERA["cp"]*24000 >= precip1) #Convert (meter/hour to mm/day)

above_tp_and_prc4 = above_tp_and_prc4.astype('int8')
above_tp_and_prc4 = above_tp_and_prc4.to_dataset(name='Option_12')
above_tp_and_prc4 = above_tp_and_prc4.assign_coords({"lon": cic_ERA.lon,
                                                "lat": cic_ERA.lat})

for v in above_tp_and_prc4.variables:
    above_tp_and_prc4[v].encoding = {}
output_path = f'/home/karengarcia/criteria_testing/Option_12/ERA5_overshoot_option12_{str(year)}.nc'
above_tp_and_prc4.to_netcdf(output_path)
print(f"File saved to", output_path)

#option 13
above_tp_and_prc8 = (ice_above_trop > 0) & \
                (prc_ERA["cp"]*24000 >= precip2) #Convert (meter/hour to mm/day)

above_tp_and_prc8 = above_tp_and_prc8.astype('int8')
above_tp_and_prc8 = above_tp_and_prc8.to_dataset(name='Option_13')
above_tp_and_prc8 = above_tp_and_prc8.assign_coords({"lon": cic_ERA.lon,
                                                "lat": cic_ERA.lat})

for v in above_tp_and_prc8.variables:
    above_tp_and_prc8[v].encoding = {}
output_path = f'/home/karengarcia/criteria_testing/Option_13/ERA5_overshoot_option13_{str(year)}.nc'
above_tp_and_prc8.to_netcdf(output_path)
print(f"File saved to", output_path)

#option 5
above_tp = ((ice_above_trop >= 1e-5))
above_tp = above_tp.astype('int8')
above_tp = above_tp.to_dataset(name='Option_5')
above_tp = above_tp.assign_coords({"lon": cic_ERA.lon,
                                "lat": cic_ERA.lat})
for v in above_tp.variables:
    above_tp[v].encoding = {}
output_path = f'/home/karengarcia/criteria_testing/Option_5/ERA5_overshoot_option5_{str(year)}.nc'
above_tp.to_netcdf(output_path)
print(f"File saved to", output_path)

#option 52 
above_tp5_and_prc4 = (ice_above_trop >=1e-5) & \
                (prc_ERA["cp"]*24000 >= precip1) #Convert (meter/hour to mm/day)

above_tp5_and_prc4 = above_tp5_and_prc4.astype('int8')
above_tp5_and_prc4 = above_tp5_and_prc4.to_dataset(name='Option_52')
above_tp5_and_prc4 = above_tp5_and_prc4.assign_coords({"lon": cic_ERA.lon,
                                                "lat": cic_ERA.lat})

for v in above_tp5_and_prc4.variables:
    above_tp5_and_prc4[v].encoding = {}
output_path = f'/home/karengarcia/criteria_testing/Option_52/ERA5_overshoot_option52_{str(year)}.nc'
above_tp5_and_prc4.to_netcdf(output_path)
print(f"File saved to", output_path)

#option 53
above_tp5_and_prc8 = (ice_above_trop >= 1e-5) & \
                (prc_ERA["cp"]*24000 >= precip2) #Convert (meter/hour to mm/day)

above_tp5_and_prc8 = above_tp5_and_prc8.astype('int8')
above_tp5_and_prc8 = above_tp5_and_prc8.to_dataset(name='Option_53')
above_tp5_and_prc8 = above_tp5_and_prc8.assign_coords({"lon": cic_ERA.lon,
                                                "lat": cic_ERA.lat})

for v in above_tp5_and_prc8.variables:
    above_tp5_and_prc8[v].encoding = {}
output_path = f'/home/karengarcia/criteria_testing/Option_53/ERA5_overshoot_option53_{str(year)}.nc'
above_tp5_and_prc8.to_netcdf(output_path)
print(f"File saved to", output_path)