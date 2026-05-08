import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import cartopy.crs as ccrs
import moviepy
import pandas as pd
import numpy as np

list_date = pd.date_range(start='2014-01-01', end='2014-12-31', freq='D')
list_date_str = list_date.strftime('%Y%m%d')

file_name = []
fps = 6 
num_years = 1 

for i in range(len(list_date_str)):
    file_path_TRMM = f"/home/karengarcia/downloads-karengarcia/TRMM_Data/GPM_3IMERGDE_07-20251103_213430/3B-DAY-E.MS.MRG.3IMERG.{str(list_date_str[i])}-S000000-E235959.V07B.nc4"
    file_path_CAN = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/prc_day_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc"
    file_path_ERA = f"/home/karengarcia/downloads-karengarcia/ERA5/ERA5_mean_conv_pc_daily.nc"

    with xr.open_dataset(file_path_TRMM) as ds:
        TRMM_ds = ds.isel(time=0) #making it 2D
        lon_TRMM, lat_TRMM = np.meshgrid(TRMM_ds["lon"], TRMM_ds["lat"])
        
    with xr.open_dataset(file_path_CAN) as ds:
        len_time = len(ds['time']) #Length of time variable
        ds = (ds.isel(time=range(len_time-(365*num_years),len_time))) # Selecting the last year in the file
        CAN_ds = ds 
        lon_CAN, lat_CAN = np.meshgrid(CAN_ds["lon"], CAN_ds["lat"])

    with xr.open_dataset(file_path_ERA) as ds:
        ERA_ds=ds
        lon_ERA, lat_ERA = np.meshgrid(ERA_ds["longitude"], ERA_ds["latitude"])

        ds_ERA5 =  ERA_ds.isel(valid_time=i)
        ds_CAN5 =  CAN_ds.isel(time=i)
        
        vmin = 1e-5
        vmax = 1e-2
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 50)

        vmin_TRMM = 0.1
        vmax_TRMM = 1000
        norm_TRMM = colors.LogNorm(vmin=vmin_TRMM, vmax=vmax_TRMM)
        levels_TRMM = np.logspace(np.log10(vmin_TRMM), np.log10(vmax_TRMM), 50)


        proj = ccrs.PlateCarree(central_longitude=180)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 8), subplot_kw={'projection': proj})

        #TRMM 
        cf1 = ax1.contourf(TRMM_ds['lon'], TRMM_ds['lat'], TRMM_ds["precipitation"].T, 
                           levels=levels_TRMM, norm=norm_TRMM, cmap='viridis_r',
                           transform=ccrs.PlateCarree())  # data is in PlateCarree coordinates
        # ERA5
        cf2 = ax2.contourf(lon_ERA, lat_ERA, ds_ERA5["avg_cpr"],
                           levels=levels, norm=norm, cmap='viridis_r',
                           transform=ccrs.PlateCarree())  # data is in PlateCarree coordinates
        # CanESM5
        cf3 = ax3.contourf(lon_CAN, lat_CAN, ds_CAN5["prc"],
                           levels=levels, norm=norm, cmap='viridis_r',
                           transform=ccrs.PlateCarree())  
        #adding coastlines
        ax1.coastlines()
        ax2.coastlines()
        ax3.coastlines()

        #setting titles
        ax1.set_title(f"TRMM {(str(TRMM_ds['time'].values))[:10]} Mean Precipitation")    
        ax2.set_title(f"ERA5 {(str(ds_ERA5['valid_time'].values))[:10]} Convective Precipitation")
        ax3.set_title(f"CanESM5 {(str(ds_CAN5['time'].values))[:11]} Convective Precipitation")

        #setting colorbars
        cbar1 = fig.colorbar(cf1, ax=ax1, orientation="vertical", label="Mean precipitation rate (mm/day)")
        cbar2 = fig.colorbar(cf2, ax=ax2, orientation="vertical", label=f"kg/m$^{2}/day$")
        cbar3 = fig.colorbar(cf3, ax=ax3, orientation="vertical", label=f"kg/m$^{2}/day$")

        cbar1.set_ticks([0.1,1, 10, 100, 1000])
        cbar1.set_ticklabels(['0.1','1', '10', '100', '1000'])
        cbar2.set_ticks([1e-5,1e-4, 1e-3, 1e-2])
        cbar2.set_ticklabels(['1e-5','1e-4', '1e-3', '1e-2'])
        cbar3.set_ticks([1e-5,1e-4, 1e-3, 1e-2])
        cbar3.set_ticklabels(['1e-5','1e-4', '1e-3', '1e-2'])

        #saving image
        plt.tight_layout()
        plt.savefig(f"/home/karengarcia/MSc_project_backup/Animations/Convec_precip/Combined/prc_{(str(ds_ERA5['valid_time'].values))[:10]}.png", dpi=300, bbox_inches='tight')
        plt.close() 
        file_name.append(f"/home/karengarcia/MSc_project_backup/Animations/Convec_precip/Combined/prc_{(str(ds_ERA5['valid_time'].values))[:10]}.png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_name, fps=fps)
clip.write_videofile('/home/karengarcia/MSc_project_backup/Daily_prc_w_TRMM_video.mp4')