import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
temp = temp.sel(lat=slice(-30,30))
press= press.sel(lat=slice(-30,30))
sphum= sphum.sel(lat=slice(-30,30))
# print(temp)
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

# print(press.values)

above_tp = press <= true_tropopause_p

#Taking the values of SHUM above the tropopause, everything else should be zero
shum_above_trop = sphum.where(above_tp, 0,0)
dP = np.abs(press.diff(dim='lev'))
# print(dP.values)
# #Realigning becasue dP array is shorter because of the differencing
shum_above_trop = shum_above_trop.isel(lev=slice(0,-1))

# # #Integral of (q*dP)/g (Units: kg*m^2)
W_strat = (shum_above_trop*dP).sum(dim='lev') / g
E_radius = 6371000.0

# 2. Calculate the grid spacing in radians
# Assumes regular spacing. d_lon and d_lat will be scalars.
d_lon = np.deg2rad(temp.lon.diff("lon").mean())
d_lat = np.deg2rad(temp.lat.diff("lat").mean())

# #Calculate the area of each cell
cell_area = (E_radius**2) * np.cos(np.deg2rad(temp.lat)) * d_lat * d_lon
total_mass = (W_strat * cell_area).sum(dim=["lat", "lon"])
total_mass_val = (total_mass.compute())#.rolling(valid_time=1).mean()
print(total_mass_val.values/1e9, "Tg")
plt.figure(figsize=(16, 6))
plt.plot(total_mass_val['time'], total_mass_val.values/1e9)
plt.xlabel("Time (YYYY-MM)")
plt.ylabel("Mass (Tg)")
plt.title("MERRA-2 Total Integrated Stratospheric Water Vapour (30N to 30S)")
final_plot_path = f"/home/karengarcia/MSc_project/Figures/MERRA_Tropics_Water_Budget.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {final_plot_path}")