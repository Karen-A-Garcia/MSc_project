from datetime import date
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

######## MLS #########
vmr_to_kgkg = 18.015 / 28.964 
file_paths = sorted(glob.glob("/home/karengarcia/downloads-karengarcia/MLS_data/v06/*.nc"))
file_paths = [f for f in file_paths if any(str(y) in f for y in range(2004, 2015))]

# Open multi-file dataset directly
ds_mls = xr.open_mfdataset(file_paths, group="H2O PressureGrid", combine="by_coords")

# Perform spatial slicing and average
mls_subset = ds_mls.sel(lat=slice(-25, 25))
zonal_avg_xr = mls_subset["value"].mean(dim=["lat", "lon"]).load() # .load() pulls data into memory

time_flat = zonal_avg_xr["time"].values
pres = zonal_avg_xr["lev"].values
zonal_avg = zonal_avg_xr.values

ds_mls.close() 

############################
########## CanAM5 ##########
############################
max_lat = 25
min_lat = -25
num_years = 11

ds_can = xr.open_dataset("/home/karengarcia/downloads-karengarcia/ESGF_downloads/Monthly/hus_Amon_CanESM5_amip_r1i1p2f1_gn_195001-201412.nc")

# Pull trailing years using negative indexing
data_hus_5 = ds_can.isel(time=slice(-12 * num_years, None)).sel(
    lat=slice(min_lat, max_lat), plev=slice(15000, 0)
)
mean_data_hus_5 = data_hus_5.mean(dim=["lat", "lon"]).load()

# Use xarray's native .dt accessor to calculate decimal times without Pandas
t_dt = mean_data_hus_5["time"].dt
times_decimal = (t_dt.year + (t_dt.month - 1) / 12 + (t_dt.day - 1) / 365).values

ds_can.close()

############################
##########  ERA5  ##########
############################
file_path_era = "/home/karengarcia/downloads-karengarcia/ERA5/Monthly/ERA5_monthly_q_1979_2019_128x64.nc"
ds_era = xr.open_dataset(file_path_era)

ds_ERA5_sub = ds_era.sel(lat=slice(-25, 25)).isel(valid_time=slice((-12*16),-12*5))
mean_ds = ds_ERA5_sub.mean(dim=["lat", "lon"]).load()

ds_era.close()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, figsize=(16, 16))

cf1 = ax1.contourf(
    time_flat, 
    pres, 
    zonal_avg.T * vmr_to_kgkg, 
    np.arange(1e-6, 6.1e-6, 1e-7), # Matches your ERA5 and CanESM5 levels perfectly now!
    cmap="Blues"
)
cf2 = ax2.contourf(mean_ds["valid_time"][7:], mean_ds["pressure_level"], (mean_ds["q"].T)[:, 7:], np.arange(1e-6, 6.1e-6, 1e-7), cmap="Blues")
cf3 = ax3.contourf(times_decimal[7:], (mean_data_hus_5["plev"] / 100), (mean_data_hus_5["hus"].T)[:, 7:], np.arange(1e-6, 6.1e-6, 1e-7), cmap="Blues")

# Labels & Limits
ax1.set_title("MLS Zonal Mean (25S - 25N)", fontsize=14)
ax2.set_title("ERA5 Zonal Mean (25S - 25N)", fontsize=14)
ax3.set_title("CanESM5 Zonal Mean (25S - 25N)", fontsize=14)

ax1.set_ylabel("Pressure (hPa)")
ax2.set_ylabel("Pressure (hPa)")
ax3.set_ylabel("Pressure (hPa)")
ax1.set_xlabel("Time")
ax2.set_xlabel("Time")
ax3.set_xlabel("Time")

ax1.set_xlim([date(2004, 8, 15), date(2014, 12, 15)])
ax1.set_ylim(110, 1)
ax3.set_ylim(110, 1)

# Colorbars
cbar1 = fig.colorbar(cf1, ax=ax1)
cbar1.set_label("Specific Humdity (kg/kg)")
cbar2 = fig.colorbar(cf2, ax=ax2)
cbar2.set_label("Specific Humdity (kg/kg)")
cbar3 = fig.colorbar(cf3, ax=ax3)
cbar3.set_label("Specific Humdity (kg/kg)")

final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Tape_recorder.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")