import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors 


precip = 4
file = f"/home/karengarcia/data-karengarcia/Overshooting/ERA5_coarse/{str(precip)}mm/ERA_overshoot_2014_{str(precip)}mm.nc"
ds_year = xr.open_mfdataset(file)

full_year = ds_year["overshoot"].mean(dim="valid_time") * 100

fig, ax = plt.subplots(1, 1,figsize=(18, 9),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.coastlines(resolution='110m', color='black', linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

cmap = plt.get_cmap("Blues")

levels = np.linspace(0, 100, 21)   # 10 intervals
norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N)

cf = ax.pcolormesh(full_year.lon,full_year.lat,full_year.values,
    norm=norm,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    shading="auto"
)

gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

cbar = fig.colorbar(cf, ax=ax, orientation='vertical',
                    shrink=0.7, pad=0.02)

cbar.set_label("Frequency of Occurrence (%)", fontsize=12)

plt.title(f"ERA5 Coarse Grained to CanAM Spatially and Temporally ({str(precip)}mm/day Threshold)",fontsize=16)

final_plot_path = f"/home/karengarcia/MSc_project/ERA5_OC_coarse_{str(precip)}mm_day.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Annual mean map saved to: {final_plot_path}")