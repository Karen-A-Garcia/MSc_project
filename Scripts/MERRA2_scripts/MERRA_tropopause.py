import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors 
import glob
import os

# --- Constants ---
g = 9.81
R_dry_air = 287.05            # J / (kg K)
target_lapserate = 2          # K/km
upper_bound = 50 * 100        # 50 hPa -> Pa
precip_thr = 4                # mm/day
lower_bound = 400 * 100       # 400 hPa -> Pa

chunks = {"lat": 361, 
          "lon": 576}

# --- Directory Setup ---
input_base = "/home/karengarcia/downloads-karengarcia/MERRA-2/"
output_dir = f"/home/karengarcia/data-karengarcia/Overshooting/MERRA2/{str(precip_thr)}mm/"
os.makedirs(output_dir, exist_ok=True)

# Get all files for June 2014 from the ASM collection
asm_files = sorted(glob.glob(os.path.join(input_base, "tavg3_3d_asm_Nv/*.2014*.nc4")))

def process_day(asm_file, precip_thr):
    """Processes a single day and returns a 2D overshoot count dataset."""
    date_str = os.path.basename(asm_file).split('.')[-2]
    
    precip_file = os.path.join(input_base, f"tavg3_3d_mst_Np/MERRA2_400.tavg3_3d_mst_Np.{date_str}.nc4") 
    mass_file = os.path.join(input_base, f"tavg3_3d_mst_Ne/MERRA2_400.tavg3_3d_mst_Ne.{date_str}.nc4") 
    
    ds_asm = xr.open_dataset(asm_file, chunks=chunks) #72 levels (model pressures)
    ds_prec = xr.open_dataset(precip_file, chunks=chunks) #42 levels but they get integrated so it doesn't matter
    ds_mass = xr.open_dataset(mass_file, chunks=chunks) #73 levelsc (model edges)
    
    #averaging over the model edges to make sure they assmilimation files and the moist files have the same lev dimensions
    cmfmc_edges = ds_mass['CMFMC']
    mass_flux = (cmfmc_edges.isel(lev=slice(0, -1)) + 
                 cmfmc_edges.isel(lev=slice(1, None)).values) / 2
    mass_flux = mass_flux.assign_coords(lev=ds_asm.lev) 
    
    #Masking the pressure between 400hPa and 50hPa so that I can restrict the tropopause location
    mask = (ds_asm['PL'] >= upper_bound) & (ds_asm['PL'] <= lower_bound)
    temp_masked = ds_asm['T'].where(mask)     #masking the temperature
    press_masked = ds_asm['PL'].where(mask)   #masking the pressure

    #taking the change in temp and pressure along the lev dimension
    dT = temp_masked.diff('lev')
    dP = press_masked.diff('lev')

    
    T_mid = (temp_masked.isel(lev=slice(0,-1)) + temp_masked.isel(lev=slice(1,None)).values) / 2
    P_mid = (press_masked.isel(lev=slice(0,-1)) + press_masked.isel(lev=slice(1,None)).values) / 2

    dz = -(R_dry_air * T_mid) / (g * P_mid) * dP 
    lapse_rate = -(dT / dz) * 1000 #K/m -> K/km

    reversed_lapse = lapse_rate.sortby('lev', ascending=False)
    trop_idx = (reversed_lapse <= target_lapserate).argmax(dim='lev').compute()
    final_pressures = P_mid.sortby('lev', ascending=False).isel(lev=trop_idx).compute()
    
    # 3. Overshooting Mask
    #Cloud ice plus cloud liquid = full cloud
    cloud_total = ds_asm['QI'].where(mask) + ds_asm['QL'].where(mask)
    above_tp = P_mid < final_pressures
    
    #vertical intergration of the  ice convective precipitation and liquid convective precipitation
    precip = ds_prec["PFLCU"].sum(dim='lev') + ds_prec["PFICU"].sum(dim='lev')
    
    overshoot_mask = (cloud_total.isel(lev=slice(0,-1)).where(above_tp) > 0) & \
                     (precip >= precip_thr / 86400) & \
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