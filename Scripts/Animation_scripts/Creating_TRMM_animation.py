import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import cartopy.crs as ccrs
import moviepy
import pandas as pd
import numpy as np

list_date = pd.date_range(start='2014-01-01', end='2014-12-31', freq='D')
list_date_str = list_date.strftime('%Y%m%d')
print(list_date_str)

file_name = []
fps = 6 
for dates in list_date_str:
    file_path = f"/home/karengarcia/downloads-karengarcia/TRMM_Data/GPM_3IMERGDE_07-20251103_213430/3B-DAY-E.MS.MRG.3IMERG.{str(dates)}-S000000-E235959.V07B.nc4"

    with xr.open_dataset(file_path) as ds:
        ds = ds.isel(time=0)

        fig = plt.figure(figsize=(15, 5))

        vmin = 0.1
        vmax = 1000
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 50)

        proj = ccrs.PlateCarree(central_longitude=180)
        ax = fig.add_subplot(1, 1, 1, projection=proj)

        cf = ax.contourf(ds['lon'], ds['lat'], ds["precipitation"].T,levels=levels, norm=norm, cmap='viridis_r')
        

    # # Add coastlines and labels
        ax.coastlines()
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{dates} TRMM map")

        cbar = fig.colorbar(cf, ax=ax, orientation="vertical", label="Daily mean precipitation rate (mm/day)")
        cbar.set_ticks([0.1,1, 10, 100, 1000])
        cbar.set_ticklabels(['0.1','1', '10', '100', '1000'])

        plt.tight_layout()
        plt.savefig(f"/home/karengarcia/MSc_project_backup/Animations/TRMM/TRMM_{dates}.png", dpi=300, bbox_inches='tight')
        plt.close() 
        file_name.append(f"/home/karengarcia/MSc_project_backup/Animations/TRMM/TRMM_{dates}.png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_name, fps=fps)
clip.write_videofile('/home/karengarcia/MSc_project_backup/Daily_TRMM_2014_log.mp4')