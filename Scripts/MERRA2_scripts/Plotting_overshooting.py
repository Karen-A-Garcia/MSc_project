import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors 

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