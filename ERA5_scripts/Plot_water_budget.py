import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
import matplotlib.colors as mcolors 

wb_file = "/home/karengarcia/ERA5_stratospheric_water_column.nc"
wb = xr.open_dataset(wb_file)
annual_mean = wb.mean(dim="valid_time")
# print(annual_mean['SWC'].mean().values)
# print(annual_mean['SWC'].min().values)


proj = ccrs.PlateCarree()
fig, ax = plt.subplots(1, 1, figsize=(24, 8), subplot_kw={'projection': proj})
ax.coastlines(color='black', linewidth=0.6, linestyle='--')
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.9)

cf1 = ax.pcolormesh(wb.longitude, wb.latitude, annual_mean["SWC"],
                    vmin = 0,
                    vmax = 0.04,
                    transform=ccrs.PlateCarree(), 
                    cmap="Blues")

gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
gl.top_labels = False
gl.right_labels = False

# Colorbar
cbar1 = fig.colorbar(cf1, ax=ax, orientation='vertical')
cbar1.set_label("Column integrated water vapour (kg/m^2)", fontsize=14)


plt.title("ERA5 Annual Mean Stratospheric Total Column Water Vapor", fontsize=16)
outpng = f"/home/karengarcia/ERA_SWB.png"
plt.savefig(outpng, dpi=300, bbox_inches='tight')
plt.close(fig)
print('Figure saved to', outpng)
