import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import moviepy
import glob
import numpy as np

file_name = []
fps=6
####----------------------------------------------------SLICING DATA---------------------------------------------------####
#ERA5
with xr.open_dataset('/home/karengarcia/downloads-karengarcia/ERA5/Daily/Temperature/ERA5_temperature_2014_FULLYEAR.nc') as ds:
    ERA_temp = ((ds.sel(latitude=slice(30, 20))) #Selecting between 20-30N
                .mean(['latitude']))             #Taking the mean value
    time_len_ERA = len(ERA_temp['valid_time'])   #value of how many time values
    ERA_lon = (ERA_temp['longitude'].values)

with xr.open_dataset('/home/karengarcia/downloads-karengarcia/ERA5/Daily/Specific_humidity/ERA5_specific_humidity_2014_FULLYEAR.nc') as ds:
    ERA_sphum = ((ds.sel(latitude=slice(30, 20))) #Selecting between 20-30N
                 .mean(['latitude']))             #Taking the mean value
    time_len_ERA = len(ERA_sphum['valid_time'])   #value of how many time values


#CanESM5
with xr.open_dataset('/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/ta/ta_day_CanESM5_historical_r1i1p1f1_gn_20110101-20141231.nc') as ds:
    time_len =len(ds['time'])                               #value of how many time values

    CanESM_temp=(((ds.sel(lat=slice(20, 30)))               #Selecting between 20-30N
            .isel(time=slice(time_len-365,time_len)))       #Selecting the last year
            .mean(['lat']))                                 #Making the mean along all latitudes
    CanESM_lon= (CanESM_temp['lon'].values)                 #Making a list of all the longitude values

with xr.open_dataset('/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/hus/hus_day_CanESM5_historical_r1i1p2f1_gn_20110101-20141231.nc') as ds:
    CanESM_sphum=(((ds.sel(lat=slice(20, 30),               #Selecting between 20-30N
              plev=slice(25000,0)))                         #Only taking from 200hPa and above    
              .isel(time=slice(time_len-365,time_len)))     #Only taking the last year
              .mean(['lat']))
    

vmin = 1e-6
vmax = 1e-4
norm = colors.LogNorm(vmin=vmin, 
                      vmax=vmax)
levels = np.logspace(np.log10(vmin), 
                     np.log10(vmax), 21)

for t in range(time_len_ERA):
    #temp
    sphum_ERA= ERA_sphum.isel(valid_time=t)
    sphum_CanESM = CanESM_sphum.isel(time=t)

    int_ds_C = CanESM_temp.isel(time=t)
    int_ds_E = ERA_temp.isel(valid_time=t)


    min_temp_level_CanESM = []
    min_temp_CanESM = []

    min_temp_level_ERA = []
    min_temp_ERA = []

#----------GETTING ALL THE TROPOPAUSE LEVELS-------------------
    for l in CanESM_lon:
        ta_list = (int_ds_C.sel(lon=l))['ta'].values

        plev = int_ds_C['plev'].values
        min_index = np.argmin(ta_list)
        min_temp_level_CanESM.append(plev[min_index])
        min_temp_CanESM.append(min(ta_list))

    for l in ERA_lon:
        ta_list = (int_ds_E.sel(lon=l))['ta'].values

        plev = int_ds_E['pressure_level'].values
        min_index = np.argmin(ta_list)
        min_temp_level_ERA.append(plev[min_index])
        min_temp_ERA.append(min(ta_list))

#---------PLLLLOOOTTTIING--------------------------------------
    fig, (ax1,ax2) = plt.subplots(2, 1, sharey=True,figsize=(20, 8))

    #ERA5
    cf1 = ax1.contourf(sphum_ERA['longitude'], sphum_ERA['pressure_level'], sphum_ERA["q"], levels=levels, norm=norm, cmap='viridis_r')
    ax1.plot(CanESM_lon, np.array(min_temp_level_CanESM) / 100, color='red', linewidth=2, label='Min Temp Level')
    #CanESM5
    cf2 = ax2.contourf(sphum_CanESM['lon'], sphum_CanESM['plev']/100, sphum_CanESM["hus"], levels=levels, norm=norm, cmap='viridis_r')
    ax2.plot(CanESM_lon, np.array(min_temp_level_CanESM) / 100, color='red', linewidth=2, label='Min Temp Level')
        
#---------SETTING ALL THE LABELS-----------------------------
    ax1.set_ylim(120, 10)
    ax1.set_ylabel("Pressure (hPa)", fontsize=14)
    ax1.set_xlabel("Longitude (Degrees East)", fontsize=14)

    ax2.set_ylim(120, 10)
    ax2.set_ylabel("Pressure (hPa)", fontsize=14)
    ax2.set_xlabel("Longitude (Degrees East)", fontsize=14)

    ax1.set_title(f"ERA5 {(str(sphum_ERA['valid_time'].values))[:10]} Mean Specific Humidity (20N-30N)", fontsize=14)
    ax2.set_title(f"CanESM5 {(str(sphum_CanESM['time'].values))[:11]} Mean Specific Humidity (20N-30N)", fontsize=14)

    cbar1 = fig.colorbar(cf1, label='Specific Humidity')
    cbar1.set_ticks([1e-6,1e-5, 1e-4])
    cbar1.set_ticklabels(["1e-6","1e-5", "1e-4"])

    cbar2 = fig.colorbar(cf2, label='Specific Humidity')
    cbar2.set_ticks([1e-6,1e-5, 1e-4])
    cbar2.set_ticklabels(["1e-6","1e-5", "1e-4"])

    plt.tight_layout()
    plt.savefig(f"/home/karengarcia/MSc_project_backup/Animations/Specific_hum/specific_hum_{(str(sphum_CanESM['time'].values))[:11]}.png", dpi=300, bbox_inches='tight')
    plt.close() 
    file_name.append(f"/home/karengarcia/MSc_project_backup/Animations/Specific_hum/specific_hum_{(str(sphum_CanESM['time'].values))[:11]}.png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_name, fps=fps)
clip.write_videofile('/home/karengarcia/MSc_project_backup/Specific_humidity_CanESM5_video.mp4')