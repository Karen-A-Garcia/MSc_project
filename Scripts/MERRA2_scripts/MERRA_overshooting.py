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
precip_thr = 8 #mm/day

input_dir = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly"
mst_Np = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_mst_Np.2014_6hourly_CanAM5_grid.nc"
mst_Ne = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_mst_Ne.2014_6hourly_CanAM5_grid.nc"
asm_Nv = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_asm_Nv.2014_6hourly_CanAM5_grid.nc"

temp = ((xr.open_dataset(asm_Nv, chunks=chunks))['T']).sortby("lev", ascending=False)
press= ((xr.open_dataset(asm_Nv, chunks=chunks))['PL']).sortby("lev", ascending=False)
QI = ((xr.open_dataset(asm_Nv, chunks=chunks))['QI']).sortby("lev", ascending=False)
QL = ((xr.open_dataset(asm_Nv, chunks=chunks))['QL']).sortby("lev", ascending=False)
cmfmc_edges = ((xr.open_dataset(mst_Ne, chunks=chunks))['CMFMC']).sortby("lev", ascending=False)
mst_Np_PFLCU = (xr.open_dataset(mst_Np, chunks=chunks))["PFLCU"]
mst_Np_PFICU = (xr.open_dataset(mst_Np, chunks=chunks))["PFICU"]
precip = mst_Np_PFLCU.isel(lev=slice(0,5)).sum(dim='lev') + mst_Np_PFICU.isel(lev=slice(0,5)).sum(dim='lev')
mass_flux = (cmfmc_edges.isel(lev=slice(0, -1)) + 
             cmfmc_edges.isel(lev=slice(1, None)).values) / 2
mass_flux = mass_flux.assign_coords(lev=temp.lev)

mask = (press >= upper_bound) & (press <= lower_bound)

temp_masked = temp.where(mask)
press_masked = press.where(mask)

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
#print("CPT Pressure:", cpt_pressure.values)

index_diff = np.abs(cpt_index - lrt_index)
# # Condition: If CPT is significantly different from LRT and meets height cap
use_cpt_condition = (index_diff >= 3) & (cpt_pressure >= max_height_cap)

# # Final Tropopause Pressure and Index selection
true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
final_tp_index = xr.where(use_cpt_condition, cpt_index, lrt_index)
print("Tropopause calculation successful")



# #### Overshooting Calculation ####
cloud_total         = QI + QL
nlev                = temp.sizes["lev"]
level_indices       = xr.DataArray(np.arange(nlev),
                            dims = ["lev"],
                            coords ={"lev": temp.lev})

above_tp = press <= true_tropopause_p

above_tp = above_tp.broadcast_like(cloud_total)
ice_above_trop = (cloud_total.where(above_tp)).sum(dim="lev")
cmu_above_trop = (mass_flux.where(above_tp)).sum(dim="lev")
    
overshoot = (ice_above_trop > 0) & \
            (precip*86400 >= precip_thr) & \
            (cmu_above_trop > 0)
overshoot = overshoot.astype("int8")
overshoot = overshoot.to_dataset(name="overshoot")
overshoot = overshoot.assign_coords({"latitude": QI.lat,
                                     "longitude": QI.lon})

for v in overshoot.variables:
        overshoot[v].encoding = {}

overshoot.to_netcdf(f"/home/karengarcia/data-karengarcia/Overshooting/MERRA_coarse/{str(precip)}mm/MERRA_overshoot_{str(year)}_{str(precip)}mm.nc")
print(f"Files saved to /home/karengarcia/data-karengarcia/Overshooting/MERRA_coarse/{str(precip)}mm/MERRA_overshoot_{str(year)}_{str(precip)}mm.nc")
print("Done!")