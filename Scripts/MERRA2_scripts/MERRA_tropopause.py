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
precip_thr = 8                # mm/day
lower_bound = 400 * 100       # 400 hPa -> Pa
max_height_cap = 70 *100

chunks = {"lat": 361, 
          "lon": 576}

# --- Directory Setup ---
input_base = "/home/karengarcia/downloads-karengarcia/MERRA-2/"
output_dir = f"/home/karengarcia/data-karengarcia/Overshooting/MERRA2/Tropopause/"
os.makedirs(output_dir, exist_ok=True)

# Get all files for June 2014 from the ASM collection
asm_files = sorted(glob.glob(os.path.join(input_base, "tavg3_3d_asm_Nv/*.201401*.nc4")))

def process_tropopause(asm_file):
    """Processes a single day and returns a 2D overshoot count dataset."""
    date_str = os.path.basename(asm_file).split('.')[-2]
    
    ds_asm = xr.open_dataset(asm_file, chunks=chunks) #72 levels (model pressures)
    ds_asm = ds_asm.sortby('lev', ascending=False)
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
    
    #LAPSE RATE TROPOPAUSE (LRT)
    tropopause_mask = lapse_rate <= target_lapserate
    lrt_index = tropopause_mask.argmax(dim='lev').compute()
    lrt_pressure = press_masked.isel(lev=lrt_index)
    
    #COLD POINT TROPOPAUSE (CPT)
    cpt_index = temp_masked.argmin(dim='lev').compute()
    cpt_pressure = press_masked.isel(lev=cpt_index)
    
    index_diff = np.abs(cpt_index - lrt_index)
    
    use_cpt_condition = (index_diff >=3) & (cpt_pressure >= max_height_cap)
    
    true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
    # final_tp_index    = xr.where(use_cpt_condition, cpt_index, lrt_index)
    
    # Return the daily mean (averaging across the 'time' dimension within the file)
    return true_tropopause_p.mean(dim='time').to_dataset(name="tp_pressure")

for f in asm_files:
    date_label = os.path.basename(f).split('.')[-2]
    out_path = os.path.join(output_dir, f"MERRA_tp_pressure_{date_label}.nc")
    
    if not os.path.exists(out_path):
        ds_day = process_tropopause(f)
        ds_day.to_netcdf(out_path)
        print(f"Processed: {date_label}")

all_daily_files = sorted(glob.glob(os.path.join(output_dir, "MERRA_tp_pressure_*.nc")))
ds_year = xr.open_mfdataset(all_daily_files, combine='nested', concat_dim='time')

# Calculate mean and convert Pa to hPa
annual_mean_hpa = ds_year['tp_pressure'].mean(dim="time").compute() / 100.0

# --- Step 3: Plotting ---
fig, ax = plt.subplots(1, 1, figsize=(18, 9), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines(resolution='110m', color='black', linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

# Typical tropopause pressure range: ~80 hPa (Tropics) to ~350 hPa (Poles)
levels = np.arange(80, 420, 20)
cmap = plt.get_cmap("RdYlBu_r") # Reversed: Blue is high altitude (low pressure)

cf = ax.contourf(annual_mean_hpa.lon, annual_mean_hpa.lat, annual_mean_hpa,
                 levels=levels,
                 transform=ccrs.PlateCarree(), 
                 cmap=cmap, 
                 extend='both')

gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
cbar.set_label("Mean Tropopause Pressure (hPa)", fontsize=12)

plt.title("MERRA-2 Annual Mean Tropopause Pressure", fontsize=16, pad=20)

final_plot_path = "/home/karengarcia/MSc_project/MERRA_Annual_Mean_Tropopause.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Annual mean map saved to: {final_plot_path}")
plt.show()