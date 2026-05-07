import numpy as np
import xarray as xr 
import os
import glob 

R_ideal = 8.314               # Pa m^3 / (mol K)
m_dryair = 28.97 / 1000       # kg / mol
ref_press = 1013.25 * 100     # Pa
R_dry_air = 287.05            # J / (kg K)
chunk_div = 50
target_lapserate = 2
precip = 8
g = 9.81                      #m/s^2

chunks = {"valid_time": 31,
         "pressure_level": -1,
          "latitude": 360,
          "longitude": 360}


year = 2014
input_dir = "/home/karengarcia/downloads-karengarcia/ERA5/Hourly/"
temp_paths = sorted(glob.glob(os.path.join(input_dir, "temperature/2014/ERA5_temperature_*1200*_Hourly_output.nc"))) 
temp_ERA = xr.open_mfdataset(temp_paths, chunks=chunks) 

q_paths = sorted(glob.glob(os.path.join(input_dir,"specific_humidity/ERA5_specific_humidity_*1200*_Hourly_output.nc"))) 
q_ERA = xr.open_mfdataset(q_paths, chunks=chunks) 


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

tropopause_mask = lapse_rate <= target_lapserate
tropopause_index = tropopause_mask.argmax(dim="pressure_level") 
tropopause_pressure = temp_ERA['pressure_level'].isel(pressure_level = tropopause_index.compute())

print("Tropopause calculation successful") 

strat_mask = q_ERA['pressure_level'] <= tropopause_pressure 
q_strat = q_ERA.q.where(strat_mask) 
dP = np.abs(q_ERA["pressure_level"].diff('pressure_level'))
#dP and q are now different length
#Need to make sure that they are the same length
q_aligned = q_strat.isel(pressure_level = slice(0, -1))

# integral of  (q * dP) / g (Units: kg/m^2)
W_strat = (q_aligned * dP*100).sum("pressure_level", skipna=True) / g
 

W_strat = W_strat.to_dataset(name="SWC")
W_strat = W_strat.assign_coords({"lat": q_ERA.latitude,
                                 "lon": q_ERA.longitude})

for v in W_strat:
    W_strat[v].encoding = {}

print("Stratospheric water column (kg/m^2) calculated.")
W_strat.to_netcdf("ERA5_stratospheric_water_column.nc")