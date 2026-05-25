import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors 


# file = f'/home/karengarcia/criteria_testing/Option_34/MERRA_overshoot_option34_2014.nc' 
# ds = xr.open_mfdataset(file)

# occurence = ds["Option_34"].sum(dim="time")

# fig, ax = plt.subplots(1, 1,figsize=(18, 9),
#                        subplot_kw={'projection': ccrs.PlateCarree()})

# ax.coastlines(resolution='110m', color='black', linewidth=1)
# ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

# cmap = plt.get_cmap("Blues")
# levels = np.linspace(1, occurence.max().values, 51)   # 51 intervals
# norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N)

# cf = ax.pcolormesh(occurence.lon,occurence.lat,occurence.values,
#     norm=norm,
#     transform=ccrs.PlateCarree(),
#     cmap=cmap,
#     shading="auto"
# )

# gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False

# cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)

# cbar.set_label("Occurrence per year (Count)", fontsize=12)

# plt.title(f"MERRA Criteria Testing \n Option 3 and 4: Precip >= 8 mm/day and mass flux over the tropopause (kg/m^2s>=0) ",fontsize=16)

# final_plot_path = f"/home/karengarcia/criteria_testing/Figures/MERRA_option34_test.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
# print(f"Figure saved to: {final_plot_path}")



thresholds = np.array([1e-10,1e-09,1e-08,1e-07,1e-06,1e-05,1e-04,1e-03])
limits = [1200,1100,1100,900,500,300,10,10]
limits2= [500, 500, 500, 500,400,400,200,10]
for i in range(len(thresholds)):
    file = f'/home/karengarcia/criteria_testing/Ice_thresholds/ERA5_above_trop_{str(thresholds[i])}_2014.nc' 
    ds = xr.open_mfdataset(file)

    occurence = ds["Ice_above_tp"].sum(dim="time")

    fig, ax = plt.subplots(1, 1,figsize=(18, 9),
                        subplot_kw={'projection': ccrs.PlateCarree()})

    ax.coastlines(resolution='110m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

    cmap = plt.get_cmap("Blues")
    levels = np.linspace(1, limits[i], 50)  
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

    plt.title(f"MERRA-2 Criteria Testing \n Combined Cloud Water and Cloud Ice Over the Tropopause (kg/kg $\geq$ {str(thresholds[i])})",fontsize=16)

    final_plot_path = f"/home/karengarcia/criteria_testing/Figures/MERRA_ice_above_trop_{str(thresholds[i])}kg_test.png"
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {final_plot_path}")
    
    file = f'/home/karengarcia/criteria_testing/Mass_flux_thresholds/MERRA_above_trop_{str(thresholds[i])}_2014.nc' 
    ds = xr.open_mfdataset(file)
    occurence = ds["Mass_flux_above_tp"].sum(dim="time")

    fig, ax = plt.subplots(1, 1,figsize=(18, 9),
                        subplot_kw={'projection': ccrs.PlateCarree()})

    ax.coastlines(resolution='110m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

    cmap = plt.get_cmap("Blues")
    levels = np.linspace(1, limits2[i], 50)  
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

    plt.title(f"MERRA Criteria Testing \n Mass flux over the tropopause (kg/m$^{2}$s $\geq$ {str(thresholds[i])})",fontsize=16)

    final_plot_path = f"/home/karengarcia/criteria_testing/Figures/MERRA_mf_above_trop_{str(thresholds[i])}kgm2s_test.png"
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {final_plot_path}")