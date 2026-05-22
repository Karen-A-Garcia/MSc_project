import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
year = 2014
chunks = {"time": 250}
g = 9.81                      # m/s^2
R_dry_air = 287.05            # J/(kg K)
target_lapserate = 2.0        # K/km
upper_bound = 50 * 100        # 50 hPa -> Pa
lower_bound = 400 * 100       # 400 hPa -> Pa
max_height_cap = 70 * 100     # 70 hPa -> Pa

input_dir = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly"
mst_Np = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_mst_Np.2014_6hourly_CanAM5_grid.nc"
mst_Ne = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_mst_Ne.2014_6hourly_CanAM5_grid.nc"
asm_Nv = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_asm_Nv.2014_6hourly_CanAM5_grid.nc"

temp = ((xr.open_dataset(asm_Nv, chunks=chunks))['T']).sortby("lev", ascending=False)
press= ((xr.open_dataset(asm_Nv, chunks=chunks))['PL']).sortby("lev", ascending=False)
sphum= ((xr.open_dataset(asm_Nv, chunks=chunks))['QV']).sortby("lev", ascending=False)
# temp = temp.isel(time=0).sel(lat=0,lon=0 ,method='nearest')
# press= press.isel(time=0).sel(lat=0,lon=0 ,method='nearest')
# sphum= sphum.isel(time=0).sel(lat=0,lon=0 ,method='nearest')

#Masking the pressure between 400hPa and 50hPa so that I can restrict the tropopause location

mask = (press >= upper_bound) & (press <= lower_bound)
temp_masked = temp.where(mask)
press_masked = press.where(mask)

# #taking the change in temp and pressure along the lev dimension
dT = temp_masked.diff('lev')
dP = press_masked.diff('lev')

T_mid = (temp_masked.isel(lev=slice(0,-1)) + temp_masked.isel(lev=slice(1,None)).values) / 2
P_mid = (press_masked.isel(lev=slice(0,-1)) + press_masked.isel(lev=slice(1,None)).values) / 2

# # Hydrostatic dz
dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
lapse_rate = - (dT / dz) * 1000

# # LAPSE RATE TROPOPAUSE (LRT)
tropopause_mask     = lapse_rate <= target_lapserate
lrt_index = tropopause_mask.argmax(dim="lev").compute()
lrt_pressure = press.isel(lev=lrt_index)

# # COLD POINT TROPOPAUSE (CPT)
cpt_index = temp_masked.T.argmin(dim="lev").compute()
cpt_pressure = press.isel(lev=cpt_index)

index_diff = np.abs(cpt_index - lrt_index)
# # Condition: If CPT is significantly different from LRT and meets height cap
use_cpt_condition = (index_diff >= 3) & (cpt_pressure >= max_height_cap)

# # Final Tropopause Pressure and Index selection
true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
final_tp_index = xr.where(use_cpt_condition, cpt_index, lrt_index)
print("Tropopause calculation successful")

#Stratosphere mask
nlev                = temp.sizes["lev"]
level_indices       = xr.DataArray(np.arange(nlev),
                            dims = ["lev"],
                            coords ={"lev": temp.lev})

above_tp = press <= true_tropopause_p
print("Above tropopause:", above_tp.values)
above_tp = above_tp.broadcast_like(sphum)
sphum_above_trop = sphum.where(above_tp)
press_strat = press.where(above_tp)
print("Specific humidity above tropopause:", sphum_above_trop.values)

dP = np.abs(press_strat.diff('lev'))

q_aligned = sphum_above_trop.isel(lev = slice(0, -1))

# integral of  (q * dP) / g (Units: kg/m^2)
W_strat = (q_aligned * dP*100).sum("lev", skipna=True) / g
print("Total column integrated water:",W_strat.values)
 

W_strat = W_strat.to_dataset(name="SWC")

W_strat = W_strat.assign_coords({"lat": temp.lat,
                                 "lon": temp.lon})

for v in W_strat:
    W_strat[v].encoding = {}

print("Stratospheric water column (kg/m^2) calculated.")
W_strat.to_netcdf("MERRA2_stratospheric_water_column.nc")




