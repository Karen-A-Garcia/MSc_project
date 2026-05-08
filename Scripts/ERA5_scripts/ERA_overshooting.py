import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.feature as cfeature 
import cartopy.crs as ccrs
import numpy as np
import matplotlib.colors as mcolors 
import os
import glob


###### HOURLY OUPUT BY YEAR #########

# g = 9.81
# R_ideal = 8.314               # Pa m^3 / (mol K)
# m_dryair = 28.97 / 1000       # kg / mol
# ref_press = 1013.25 * 100     # Pa
# R_dry_air = 287.05            # J / (kg K)
# chunk_div = 50
# target_lapserate = 2
# precip = 8

# chunks = {"valid_time": 31,
#           "pressure_level": -1,
#           "latitude": 360,
#           "longitude": 360}

# years = np.arange(2004,2015,1)

# for year in years:

#     temp_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/temperature/ERA5_temperature_{str(year)}_1200_Hourly_output.nc"
#     cloud_ice_w_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/specific_cloud_ice_water_content/ERA5_specific_cloud_ice_water_content_{str(year)}_1200_Hourly_output.nc"
#     cloud_liquid_w_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/specific_cloud_liquid_water_content/ERA5_specific_cloud_liquid_water_content_{str(year)}_1200_Hourly_output.nc"
#     convective_precip_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/convective_precipitation/ERA5_convective_precipitation_{str(year)}_1200_Hourly_output.nc"

#     temp_ERA = xr.open_dataset(temp_file, chunks=chunks)
#     cp = xr.open_dataset(convective_precip_file) 

#     # Convert hPa → Pa safely
#     temp_ERA['pressure_level'] = (temp_ERA['pressure_level']).astype('float32')
#     temp_ERA['pressure_level'].encoding = {}

#     dT = temp_ERA.t.diff('pressure_level')
#     dP = temp_ERA.pressure_level.diff('pressure_level')

#     T_low = temp_ERA.t.isel(pressure_level=slice(0, -1))
#     T_high = temp_ERA.t.isel(pressure_level=slice(1, None))
#     T_mid = (T_low + T_high) / 2

#     P_low = temp_ERA.pressure_level.isel(pressure_level=slice(0, -1))
#     P_high = temp_ERA.pressure_level.isel(pressure_level=slice(1, None))
#     P_mid = (P_low + P_high) / 2

#     dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
#     lapse_rate = -(dT / dz) * 1000

#     tropopause_mask = lapse_rate <= target_lapserate
#     tropopause_index = tropopause_mask.argmax(dim="pressure_level")

#     cloud_ice = xr.open_dataset(cloud_ice_w_file, chunks=chunks)
#     cloud_liquid = xr.open_dataset(cloud_liquid_w_file, chunks=chunks)

#     cloud_ice['pressure_level'] = (cloud_ice['pressure_level']).astype('float32')
#     cloud_ice['pressure_level'].encoding = {}

#     cloud_liquid['pressure_level'] = (cloud_liquid['pressure_level']).astype('float32')
#     cloud_liquid['pressure_level'].encoding = {}

#     cloud_total = cloud_ice.ciwc + cloud_liquid.clwc

#     nlev = temp_ERA.sizes["pressure_level"]

#     level_indices = xr.DataArray(np.arange(nlev),
#                                 dims=["pressure_level"],
#                                 coords={"pressure_level": temp_ERA.pressure_level})

#     above_tp = level_indices > tropopause_index
#     above_tp = above_tp.broadcast_like(cloud_total)

#     cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= 0.333/1000)
#     cloud_above_tp_and_prc = cloud_above_tp_and_prc.any(dim="pressure_level")

#     cloud_above_tp_and_prc = cloud_above_tp_and_prc.astype("int8")

#     cloud_above_tp_and_prc = cloud_above_tp_and_prc.to_dataset(name="overshoot")

#     cloud_above_tp_and_prc = cloud_above_tp_and_prc.assign_coords({"latitude": temp_ERA.latitude,
#                                                 "longitude": temp_ERA.longitude})

#     for v in cloud_above_tp_and_prc.variables:
#         cloud_above_tp_and_prc[v].encoding = {}

#     cloud_above_tp_and_prc.to_netcdf(f"/home/karengarcia/data-karengarcia/Overshooting/ERA5/ERA_overshoot_{str(year)}_1200_{str(precip)}mm.nc")
#     print(f"Files saved to /home/karengarcia/data-karengarcia/Overshooting/ERA5/ERA_overshoot_{str(year)}_1200_{str(precip)}mm.nc")
#     print("Done!")

# #### TESTING DAILY OUTPUTS ##### 
# import xarray as xr 
# import matplotlib.pyplot as plt 
# import cartopy.feature as cfeature 
# import cartopy.crs as ccrs
# import numpy as np

# g = 9.81
# R_ideal = 8.314               # Pa m^3 / (mol K)
# m_dryair = 28.97 / 1000       # kg / mol
# ref_press = 1013.25 * 100     # Pa
# R_dry_air = 287.05            # J / (kg K)
# chunk_div = 50
# target_lapserate = 2
# precip = 4


# cp_file   = "downloads-karengarcia/ERA5/Daily/convective_precipitation/ERA5_June2014_convective_precipitation.nc"
# ciwc_file = "downloads-karengarcia/ERA5/Daily/specific_cloud_ice_water_content/ERA5_June2014_specific_cloud_ice_water_content.nc"
# clwc_file = "downloads-karengarcia/ERA5/Daily/specific_cloud_liquid_water_content/ERA5_June2014_specific_cloud_liquid_water_content.nc"
# temp_file = "downloads-karengarcia/ERA5/Daily/temperature/ERA5_June2014_temperature.nc"
 
# temp_ERA = xr.open_dataset(temp_file)
# cloud_ice = xr.open_dataset(ciwc_file)
# cloud_liquid = xr.open_dataset(clwc_file)
# cp = xr.open_dataset(cp_file)
# print("Done reading files")

# temp_ERA['pressure_level'] = (temp_ERA['pressure_level']).astype('float32')
# temp_ERA['pressure_level'].encoding = {}

# dT = temp_ERA.t.diff('pressure_level')
# dP = temp_ERA.pressure_level.diff('pressure_level')

# T_low = temp_ERA.t.isel(pressure_level=slice(0, -1))
# T_high = temp_ERA.t.isel(pressure_level=slice(1, None))
# T_mid = (T_low + T_high) / 2

# P_low = temp_ERA.pressure_level.isel(pressure_level=slice(0, -1))
# P_high = temp_ERA.pressure_level.isel(pressure_level=slice(1, None))
# P_mid = (P_low + P_high) / 2

# dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
# lapse_rate = -(dT / dz) * 1000

# print("Done calculating lapse rate")

# tropopause_mask = lapse_rate <= target_lapserate
# tropopause_index = tropopause_mask.argmax(dim="pressure_level")


# cloud_ice['pressure_level'] = (cloud_ice['pressure_level']).astype('float32')
# cloud_ice['pressure_level'].encoding = {}

# cloud_liquid['pressure_level'] = (cloud_liquid['pressure_level']).astype('float32')
# cloud_liquid['pressure_level'].encoding = {}

# cloud_total = cloud_ice.ciwc + cloud_liquid.clwc

# nlev = temp_ERA.sizes["pressure_level"]

# level_indices = xr.DataArray(np.arange(nlev),
#                                 dims=["pressure_level"],
#                                 coords={"pressure_level": temp_ERA.pressure_level})

# above_tp = level_indices > tropopause_index
# above_tp = above_tp.broadcast_like(cloud_total)

# cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= precip/1000)
# cloud_above_tp_and_prc = cloud_above_tp_and_prc.any(dim="pressure_level")

# cloud_above_tp_and_prc = cloud_above_tp_and_prc.astype("int8")

# cloud_above_tp_and_prc = cloud_above_tp_and_prc.to_dataset(name="overshoot")

# cloud_above_tp_and_prc = cloud_above_tp_and_prc.assign_coords({"latitude": temp_ERA.latitude,
#                                                 "longitude": temp_ERA.longitude})

# for v in cloud_above_tp_and_prc.variables:
#         cloud_above_tp_and_prc[v].encoding = {}

# cloud_above_tp_and_prc.to_netcdf(f"/home/karengarcia/data-karengarcia/Overshooting/ERA5/ERA_overshoot_Daily_June2014_{str(precip)}mm.nc")
# print(f"Files saved to /home/karengarcia/data-karengarcia/Overshooting/ERA5/ERA_overshoot_Daily_June2014_{str(precip)}mm.nc")
# print("Done!")

# file = f"/home/karengarcia/data-karengarcia/Overshooting/ERA5/ERA_overshoot_Daily_June2014_{str(precip)}mm.nc"
# overshoot = xr.open_dataset(file) 

# overshoot_sum = overshoot['overshoot'].sum(dim="valid_time")

# lon_ERA = overshoot_sum["longitude"]
# lat_ERA = overshoot_sum["latitude"] 

# proj = ccrs.PlateCarree()
# fig, ax = plt.subplots(1, 1, figsize=(24, 8), subplot_kw={'projection': proj})

# ax.coastlines(color='black', linewidth=0.6, linestyle='--')
# ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

# levels = np.arange(0, 3, 1)
# norm = mcolors.BoundaryNorm(levels, ncolors=plt.get_cmap("Blues").N)

# cf1 = ax.pcolormesh(lon_ERA, lat_ERA, overshoot_sum.values,
#                     transform=ccrs.PlateCarree(), 
#                     cmap="Blues", 
#                     norm=norm)

# gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
# gl.top_labels = False
# gl.right_labels = False

# cbar1 = fig.colorbar(cf1, ax=ax, orientation='vertical')
# cbar1.set_label("Overshooting events (Count)", fontsize=14)
# cbar1.set_ticks(np.arange(0, 3, 1))

# plt.title(f"ERA5 Overshooting Event Map June 2014 with {str(precip)}mm/day Precipitation Threshold (Daily)", fontsize=16)

# outpng = f"/home/karengarcia/ERA_Overshooting_Event_Map_June2014_{str(precip)}mm.png"
# plt.savefig(outpng, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved to", outpng)


###### 2014 0600-1200 HOURLY OUTPUT #######

g = 9.81
R_ideal = 8.314               # Pa m^3 / (mol K)
m_dryair = 28.97 / 1000       # kg / mol
ref_press = 1013.25 * 100     # Pa
R_dry_air = 287.05            # J / (kg K)
chunk_div = 50
target_lapserate = 2
precip = 8

chunks = {"valid_time": 31,
         "pressure_level": -1,
          "latitude": 360,
          "longitude": 360}

# hours = ['0600', '0700', '0800', '0900', '1000', '1100', '1200','1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
# hours = ['0000', '0100', '0200', '0300', '0500']
hours = ['0400']
year = 2014
for hour in hours:
        temp_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/temperature/{str(year)}/ERA5_temperature_{str(year)}_{str(hour)}_Hourly_output.nc"
        cloud_ice_w_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/specific_cloud_ice_water_content/{str(year)}/ERA5_specific_cloud_ice_water_content_{str(year)}_{str(hour)}_Hourly_output.nc"
        cloud_liquid_w_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/specific_cloud_liquid_water_content/{str(year)}/ERA5_specific_cloud_liquid_water_content_{str(year)}_{str(hour)}_Hourly_output.nc"
        convective_precip_file = f"/home/karengarcia/downloads-karengarcia/ERA5/Hourly/convective_precipitation/{str(year)}/ERA5_convective_precipitation_{str(year)}_{str(hour)}_Hourly_output.nc"

        temp_ERA = xr.open_dataset(temp_file, chunks=chunks)
        cp = xr.open_dataset(convective_precip_file) 

        # Convert hPa → Pa safely
        temp_ERA['pressure_level'] = (temp_ERA['pressure_level']).astype('float32')
        temp_ERA['pressure_level'].encoding = {}

        dT = temp_ERA.t.diff('pressure_level')
        dP = temp_ERA.pressure_level.diff('pressure_level')

        T_low = temp_ERA.t.isel(pressure_level=slice(0, -1))
        T_high = temp_ERA.t.isel(pressure_level=slice(1, None))
        T_mid = (T_low + T_high) / 2

        P_low = temp_ERA.pressure_level.isel(pressure_level=slice(0, -1))
        P_high = temp_ERA.pressure_level.isel(pressure_level=slice(1, None))
        P_mid = (P_low + P_high) / 2

        dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
        lapse_rate = -(dT / dz) * 1000

        tropopause_mask = lapse_rate <= target_lapserate
        tropopause_index = tropopause_mask.argmax(dim="pressure_level")

        cloud_ice = xr.open_dataset(cloud_ice_w_file, chunks=chunks)
        cloud_liquid = xr.open_dataset(cloud_liquid_w_file, chunks=chunks)

        cloud_ice['pressure_level'] = (cloud_ice['pressure_level']).astype('float32')
        cloud_ice['pressure_level'].encoding = {}

        cloud_liquid['pressure_level'] = (cloud_liquid['pressure_level']).astype('float32')
        cloud_liquid['pressure_level'].encoding = {}

        cloud_total = cloud_ice.ciwc + cloud_liquid.clwc

        nlev = temp_ERA.sizes["pressure_level"]

        level_indices = xr.DataArray(np.arange(nlev),
                                        dims=["pressure_level"],
                                        coords={"pressure_level": temp_ERA.pressure_level})

        above_tp = level_indices > tropopause_index
        above_tp = above_tp.broadcast_like(cloud_total)
        if precip == 8:   #8mm/day
                cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= 0.333/1000)
        if precip == 4:   #4mm/day
                cloud_above_tp_and_prc = (cloud_total.where(above_tp) > 0) & (cp['cp'] >= 0.166/1000)
        cloud_above_tp_and_prc = cloud_above_tp_and_prc.any(dim="pressure_level")

        cloud_above_tp_and_prc = cloud_above_tp_and_prc.astype("int8")

        cloud_above_tp_and_prc = cloud_above_tp_and_prc.to_dataset(name="overshoot")

        cloud_above_tp_and_prc = cloud_above_tp_and_prc.assign_coords({"latitude": temp_ERA.latitude,
                                                        "longitude": temp_ERA.longitude})

        for v in cloud_above_tp_and_prc.variables:
                cloud_above_tp_and_prc[v].encoding = {}

        cloud_above_tp_and_prc.to_netcdf(f"/home/karengarcia/data-karengarcia/Overshooting/ERA5/Hourly/{str(precip)}mm/ERA_overshoot_{str(year)}_{str(hour)}_{str(precip)}mm.nc")
        print(f"Files saved to /home/karengarcia/data-karengarcia/Overshooting/ERA5/Hourly/{str(precip)}mm/ERA_overshoot_{str(year)}_{str(hour)}_{str(precip)}mm.nc")
        print("Done!")