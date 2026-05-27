import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors 

all_daily_files = "/home/karengarcia/data-karengarcia/Overshooting/MERRA_coarse/new_criteria/MERRA_overshoot_2014.nc"
ds_month = xr.open_dataset(all_daily_files, chunks={'time': 8})

monthly_cumulative = ds_month['overshoot'].sum(dim="time").compute()

lon = monthly_cumulative["lon"]
lat = monthly_cumulative["lat"] 

fig, ax = plt.subplots(1, 1, figsize=(24, 8), 
                       subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines(color='black', linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4)
# ax.add_feature(cfeature.LAND)


levels = np.arange(1, 500, 10,dtype = int) 
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
cbar.set_label("Occurences (Count)", fontsize=12)

plt.title(f"MERRA-2 Overshooting Events 2014", fontsize=16)

final_plot_path = f"/home/karengarcia/MSc_project/Figures/OC_after_testing/MERRA_Overshooting_2014_npc.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Cumulative map saved to: {final_plot_path}")



all_daily_files = "/home/karengarcia/data-karengarcia/Overshooting/MERRA_coarse/1e-3/MERRA_overshoot_2014.nc"
ds_month = xr.open_dataset(all_daily_files, chunks={'time': 8})

monthly_cumulative = ds_month['overshoot'].sum(dim="time").compute()

lon = monthly_cumulative["lon"]
lat = monthly_cumulative["lat"] 

fig, ax = plt.subplots(1, 1, figsize=(24, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines(color='black', linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4)
# ax.add_feature(cfeature.LAND)


# levels = np.arange(1, 500, 10,dtype = int) 
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
cbar.set_label("Occurences (Count)", fontsize=12)

plt.title(f"MERRA-2 Overshooting Events", fontsize=16)

final_plot_path = f"/home/karengarcia/MSc_project/Figures/OC_after_testing/MERRA_Overshooting_2014.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Cumulative map saved to: {final_plot_path}")