import numpy as np
import xarray as xr 
import os
import glob 

g = 9.81
R_dry_air = 287.05            # J / (kg K)
target_lapserate = 2          # K/km
upper_bound = 50 * 100        # 50 hPa -> Pa
precip_thr = 8                # mm/day
lower_bound = 400 * 100       # 400 hPa -> Pa

chunks = {"lat":  361, 
          "lon":  576,
          "time": 40}

 
input_path = "/home/karengarcia/downloads-karengarcia/MERRA-2/tavg3_3d_asm_Nv/"
as_files = sorted(glob.glob(os.path.join(input_path,"MERRA2_400.tavg3_3d_asm_Nv.20141*.nc4")))
q     = xr.open_mfdataset(as_files, chunks=chunks)["QV"]
temp  = xr.open_mfdataset(as_files, chunks=chunks)["T"]
press = xr.open_mfdataset(as_files, chunks=chunks)["PL"]

mask = (press >= upper_bound) & (press <= lower_bound)
temp_masked = temp.where(mask)     #masking the temperature
press_masked = press.where(mask)   #masking the pressure

#     #taking the change in temp and pressure along the lev dimension
dT = temp_masked.diff('lev')
dP = press_masked.diff('lev')

    
T_mid = (temp_masked.isel(lev=slice(0,-1)) + temp_masked.isel(lev=slice(1,None)).values) / 2
P_mid = (press_masked.isel(lev=slice(0,-1)) + press_masked.isel(lev=slice(1,None)).values) / 2

dz = -(R_dry_air * T_mid) / (g * P_mid) * dP 
lapse_rate = -(dT / dz) * 1000 #K/m -> K/km

reversed_lapse = lapse_rate.sortby('lev', ascending=False)
trop_idx = (reversed_lapse <= target_lapserate).argmax(dim='lev').compute()
final_pressures = P_mid.sortby('lev', ascending=False).isel(lev=trop_idx).compute()

strat_mask = press <= final_pressures
q_strat = q.where(strat_mask)
press_strat = press.where(strat_mask)
dP = np.abs(press_strat.diff('lev'))

q_aligned = q_strat.isel(lev = slice(0, -1))

# integral of  (q * dP) / g (Units: kg/m^2)
W_strat = (q_aligned * dP*100).sum("lev", skipna=True) / g
 

W_strat = W_strat.to_dataset(name="SWC")

W_strat = W_strat.assign_coords({"lat": q.lat,
                                 "lon": q.lon})

for v in W_strat:
    W_strat[v].encoding = {}

print("Stratospheric water column (kg/m^2) calculated.")
W_strat.to_netcdf("MERRA2_stratospheric_water_column.nc")




