import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors 


file = f'/home/karengarcia/criteria_testing/Option_5/ERA5_overshoot_option5_2014.nc' 
ds = xr.open_mfdataset(file)

occurence = ds["Option_5"].sum(dim="valid_time")

fig, ax = plt.subplots(1, 1,figsize=(18, 9),
                       subplot_kw={'projection': ccrs.PlateCarree()})

ax.coastlines(resolution='110m', color='black', linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

cmap = plt.get_cmap("Blues")
levels = np.linspace(1, occurence.max().values, 51)   # 51 intervals
norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N)

cf = ax.pcolormesh(occurence.lon,occurence.lat,occurence.values,
    norm=norm,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    shading="auto"
)

gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
gl.top_labels = False
gl.right_labels = False

cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)

cbar.set_label("Occurrence (Count)", fontsize=12)

plt.title(f"ERA5 Criteria Testing \n Option 5: Ice over the tropopause (kg/kg>=1e-5)",fontsize=16)

final_plot_path = f"/home/karengarcia/criteria_testing/Figures/ERA5_option5_test.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {final_plot_path}")