import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import moviepy


file_name = []
fps=5
num_years = 11
with xr.open_dataset('/home/karengarcia/downloads-karengarcia/ESGF_downloads/Monthly/hus_Amon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc') as ds:
    len_time = len(ds['time']) #Length of time variable

    ds = (ds.isel(time=range(len_time-(12*num_years),len_time))) # Selecting the last 11 years in the file
    
    lon, lat = np.meshgrid(ds["lon"], ds["lat"])

    #going through every year
    for i in range(len(ds['time'])):
            ds_150 = (ds.sel(plev=15000)).isel(time=i)
            ds_100 = (ds.sel(plev=10000)).isel(time=i)
            ds_70 =  (ds.sel(plev=7000)).isel(time=i)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                            sharey=True,
                                            figsize=(16, 8),
                                            subplot_kw={'projection': ccrs.PlateCarree()})

            cf1 = ax1.contourf(lon, lat, ds_150["hus"], np.arange(1e-6, 35e-6, 1e-7), cmap='viridis_r', transform=ccrs.PlateCarree())
            cf2 = ax2.contourf(lon, lat, ds_100["hus"], np.arange(1e-6, 7e-6, 1e-7), cmap='viridis_r', transform=ccrs.PlateCarree())
            cf3 = ax3.contourf(lon, lat, ds_70["hus"], np.arange(1e-6, 7e-6, 1e-7), cmap='viridis_r', transform=ccrs.PlateCarree())

            # Titles and labels
            ax1.set_title(f"{(ds_150['time']).values} 150hPa", fontsize=14)
            ax2.set_title(f"{(ds_100['time']).values} 100hPa", fontsize=14)
            ax3.set_title(f"{(ds_70['time']).values} 70hPa", fontsize=14)

            # Add coastlines
            ax1.coastlines()
            ax2.coastlines()
            ax3.coastlines()

            #colorbars
            cbar1 = fig.colorbar(cf1, ax=ax1)
            cbar1.set_label("Specific Humdity")

            cbar2 = fig.colorbar(cf2, ax=ax2)
            cbar2.set_label("Specific Humdity")
            
            cbar3 = fig.colorbar(cf3, ax=ax3)
            cbar3.set_label("Specific Humdity")

            plt.tight_layout()
            plt.savefig(f"/home/karengarcia/MSc_project_backup/Animations/70_100_150/specific_hum_{str(ds_150['time'].values)}.png", dpi=300, bbox_inches='tight')
            plt.close() 
            file_name.append(f"/home/karengarcia/MSc_project_backup/Animations/70_100_150/specific_hum_{str(ds_150['time'].values)}.png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_name, fps=fps)
clip.write_videofile('/home/karengarcia/MSc_project_backup/specific_humidity_video.mp4')