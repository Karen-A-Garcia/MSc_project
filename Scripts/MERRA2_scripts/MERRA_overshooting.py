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
QI = ((xr.open_dataset(asm_Nv, chunks=chunks))['QI']).sortby("lev", ascending=False)
QL = ((xr.open_dataset(asm_Nv, chunks=chunks))['QL']).sortby("lev", ascending=False)
cmfmc_edges = ((xr.open_dataset(mst_Ne, chunks=chunks))['CMFMC']).sortby("lev", ascending=False)
# cmfmc_edges = cmfmc_edges.isel(time=0).sel(lon=0, lat=1.40625)
# temp = temp.isel(time=0).sel(lon=0, lat=1.40625)
# press= press.isel(time=0).sel(lon=0, lat=1.40625)
# QI = QI.isel(time=0).sel(lon=0, lat=1.40625)
mst_Np_PFLCU = (xr.open_dataset(mst_Np, chunks=chunks))["PFLCU"]
mst_Np_PFICU = (xr.open_dataset(mst_Np, chunks=chunks))["PFICU"]
precip = mst_Np_PFLCU.isel(lev=slice(0,5)).sum(dim='lev') + mst_Np_PFICU.isel(lev=slice(0,5)).sum(dim='lev')
# precip = precip.isel(time=0).sel(lon=0, lat=1.40625)
# cmfmc_edges = (xr.open_dataset(mst_Ne, chunks=chunks))['CMFMC']
mass_flux = (cmfmc_edges.isel(lev=slice(0, -1)) + 
             cmfmc_edges.isel(lev=slice(1, None)).values) / 2
mass_flux = mass_flux.assign_coords(lev=temp.lev)

#Masking the pressure between 400hPa and 50hPa so that I can restrict the tropopause location

mask = (press >= upper_bound) & (press <= lower_bound)
#print("Pressure values", press.values)
#print("UTLS Mask:", mask.values)
temp_masked = temp.where(mask)
press_masked = press.where(mask)
#print("press_masked", press_masked.values)

# #taking the change in temp and pressure along the lev dimension
dT = temp_masked.diff('lev')
#print("dT", dT.values)
dP = press_masked.diff('lev')
#print("dP:", dP.values)

T_mid = (temp_masked.isel(lev=slice(0,-1)) + temp_masked.isel(lev=slice(1,None)).values) / 2
P_mid = (press_masked.isel(lev=slice(0,-1)) + press_masked.isel(lev=slice(1,None)).values) / 2

# # Hydrostatic dz
dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
#print("dz:", dz.values)
lapse_rate = - (dT / dz) * 1000
#print('lapse rate:', lapse_rate.values)

# # LAPSE RATE TROPOPAUSE (LRT)
tropopause_mask     = lapse_rate <= target_lapserate
#print("Lapse rate mask:", tropopause_mask.values)
lrt_index = tropopause_mask.argmax(dim="lev").compute()
#print("lrt_index:", lrt_index.values)
lrt_pressure = press.isel(lev=lrt_index)
#print("lrt pressure:", lrt_pressure.values)

# # COLD POINT TROPOPAUSE (CPT)
cpt_index = temp_masked.T.argmin(dim="lev").compute()
#print("Cold Point TP Index:", cpt_index.values)
cpt_pressure = press.isel(lev=cpt_index)
#print("CPT Pressure:", cpt_pressure.values)

index_diff = np.abs(cpt_index - lrt_index)
#print("index diff:", index_diff)
# # Condition: If CPT is significantly different from LRT and meets height cap
use_cpt_condition = (index_diff >= 3) & (cpt_pressure >= max_height_cap)
#print(use_cpt_condition.values)

# # Final Tropopause Pressure and Index selection
true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
#print("Final tropopause pressure:", true_tropopause_p.values)
final_tp_index = xr.where(use_cpt_condition, cpt_index, lrt_index)
#print("Final tropopause index:",final_tp_index.values)
#print("Tropopause calculation successful")



# #### Overshooting Calculation ####
cloud_total         = QI + QL
nlev                = temp.sizes["lev"]
#print("nlev:", nlev)
level_indices       = xr.DataArray(np.arange(nlev),
                            dims = ["lev"],
                            coords ={"lev": temp.lev})

above_tp = press <= true_tropopause_p
#print("Above tropopause:", above_tp.values)
above_tp = above_tp.broadcast_like(cloud_total)
ice_above_trop = (cloud_total.where(above_tp)).sum(dim="lev")
dmcu_above_trop = (mass_flux.where(above_tp)).sum(dim="lev")
    
overshoot_mask = (ice_above_trop > 0) & \
                     (precip*86400 >= precip_thr) & \
                     (mass_flux.where(above_tp) > 0)
    
    daily_overshoot = overshoot_mask.any(dim="lev").astype("int8")
    return daily_overshoot.to_dataset(name="overshoot")


for f in asm_files:
    date_label = os.path.basename(f).split('.')[-2]
    out_path = os.path.join(output_dir, f"MERRA_overshoot_{str(precip_thr)}mm_{date_label}.nc")
    
    ds_day = process_day(f,precip_thr)
    ds_day = ds_day.coarsen(lat=5, lon=4, boundary="trim").mean()
    ds_day.to_netcdf(out_path)

all_daily_files = sorted(glob.glob(os.path.join(output_dir, f"MERRA_overshoot_{str(precip_thr)}mm_*.nc")))
ds_month = xr.open_mfdataset(all_daily_files, chunks={'time': 8})

monthly_cumulative = ds_month['overshoot'].sum(dim="time").compute()

lon = monthly_cumulative["lon"]
lat = monthly_cumulative["lat"] 

fig, ax = plt.subplots(1, 1, figsize=(24, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines(color='black', linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4)
# ax.add_feature(cfeature.LAND)


levels = np.arange(1, 28, 1,dtype = int) 
cmap = plt.get_cmap("Blues").copy()
cmap.set_under('white', alpha=0)

norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='min')

cf = ax.pcolormesh(lon, lat, monthly_cumulative,
                   transform=ccrs.PlateCarree(), 
                   cmap=cmap, 
                   norm=norm)

gl = ax.gridlines(draw_labels=True, alpha=0.2)
gl.top_labels = False; gl.right_labels = False

cbar = fig.colorbar(cf, ax=ax, orientation='vertical')
cbar.set_label("Total Overshooting Events", fontsize=12)

plt.title(f"MERRA-2 Overshooting Events 2014 Precipitation Threshold: {precip_thr}mm/day", fontsize=16)

final_plot_path = f"/home/karengarcia/MERRA_Overshooting_2014_Cumulative_{str(precip_thr)}mm.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Cumulative map saved to: {final_plot_path}")