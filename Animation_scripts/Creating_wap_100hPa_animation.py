import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import moviepy 

fps = 6
file_name = []
num_years = 1
################### Vertical Velocity###################
#CanESM
wap_file_CAN  = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/wap/wap_day_CanESM5_historical_r1i1p1f1_gn_20110101-20141231.nc"
ds_wap_CAN = xr.open_dataset(wap_file_CAN, chunks={})
time_len = ds_wap_CAN.sizes["time"]
wap_CAN = ds_wap_CAN.sel(plev=10000).isel(time=slice(time_len-365*num_years, time_len))

#ERA
file_path_ERA = "/home/karengarcia/downloads-karengarcia/ERA5/Daily/Vertical_velocity/ERA5_Vertical_velocity_2014_FULLYEAR.nc"
ds_ERA = xr.open_dataset(file_path_ERA, chunks={})
wap_ERA = ds_ERA.sel(pressure_level=100)
    

for i in range(len(wap_CAN.time)):
    #Taking one day
    wap_C = wap_CAN["wap"].isel(time=i)
    wap_E = wap_ERA["w"].isel(valid_time=i)

    #Plotting
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 8), subplot_kw={'projection': proj})

    # ERA5
    cf1 = ax1.contourf(wap_E.longitude, wap_E.latitude, wap_E, 
                       np.linspace(-0.1, 0.1, 51), cmap='bwr_r')
    # CanESM5
    cf2 = ax2.contourf(wap_C.lon, wap_C.lat, wap_C,
                       np.linspace(-0.1, 0.1, 51), cmap='bwr_r')
    #Coastlines
    ax1.coastlines()
    ax2.coastlines()
    #Setting titles
    ax1.set_title(f"ERA5 {str(wap_E.valid_time.values)[:10]} Vertical Velocity (100hPa)")
    ax2.set_title(f"CanESM5 {str(wap_C.time.values)[:10]} Vertical Velocity (100hPa)")
    #Colorbars
    plt.colorbar(cf1, ax=ax1, orientation="vertical", label="Pa/s")
    plt.colorbar(cf2, ax=ax2, orientation="vertical", label="Pa/s")

    plt.tight_layout()
    plt.savefig(f"/home/karengarcia/MSc_project_backup/Animations/Wap_at_100hPa/wap_{(str(wap_C['time'].values))[:10]}.png", dpi=300, bbox_inches='tight')
    plt.close() 
    file_name.append(f"/home/karengarcia/MSc_project_backup/Animations/Wap_at_100hPa/wap_{(str(wap_C['time'].values))[:10]}.png")

#Creating movie
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_name, fps=fps)
clip.write_videofile('/home/karengarcia/MSc_project_backup/MP4_videos/Daily_wap_2014.mp4')