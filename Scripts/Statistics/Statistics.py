from scipy.stats import skew, kurtosis
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import matplotlib.colors as colors
import cftime
import glob
from netCDF4 import Dataset
from datetime import date, timedelta
from matplotlib.cbook import get_sample_data

chunk_div = 5
##Daily output
# q_50_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/hus_day_CanESM5_50hPa_historical_r1i1p2f1_gn_19410101-20141231.nc"
q_50_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/hus_day_CanESM5_50hPa_amip_r1i1p2f1_gn_19500101-20141231.nc"
# q_100_file= "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/hus_day_CanESM5_100hPa_historical_r1i1p2f1_gn_19410101-20141231.nc"
q_100_file= "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/hus_day_CanESM5_100hPa_amip_r1i1p2f1_gn_19500101-20141231.nc"
#Monthly output

q_100_daily = (xr.open_dataset(q_100_file).hus).chunk({'time': chunk_div})
q_50_daily  = (xr.open_dataset( q_50_file).hus).chunk({'time': chunk_div})

#Moments for 50hPa
m50_q_d = q_50_daily.mean(dim='time')
v50_q_d = q_50_daily.var(dim='time')
sk50q_d = skew(q_50_daily, keepdims = True)
k50_q_d = kurtosis(q_50_daily, keepdims = True)
#Moments for 100hPa
m100_q_d = q_100_daily.mean(dim='time')
v100_q_d = q_100_daily.var(dim='time')
sk100q_d = skew(q_100_daily, keepdims = True)
k100_q_d = kurtosis(q_100_daily, keepdims = True)
lon_CAN = q_100_daily.lon
lat_CAN = q_100_daily.lat

# vmin = 1e-15
# vmax = 1e-11
# norm = colors.LogNorm(vmin=vmin, vmax=vmax)
# levels = np.logspace(np.log10(vmin), np.log10(vmax))
# proj = ccrs.Sinusoidal(central_longitude=0)

# fig, axes = plt.subplots(nrows=2, ncols=4,figsize=(24, 6),
#                          sharex=True, sharey=True, subplot_kw={"projection": proj})

# cf1 = axes[0, 0].contourf(lon_CAN, lat_CAN, m100_q_d / 1e-6,
#                           levels=np.linspace(1.5, 5, 201), transform=ccrs.PlateCarree())
# axes[0, 0].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0, 0].set_title("Mean of q at 100 hPa")
# cbar1 = fig.colorbar(cf1, ax=axes[0, 0], orientation="vertical")
# cbar1.set_label(f"Mass fraction of water vapor ($10^{-6}$)")
# cbar1.set_ticks([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
# cf2 = axes[1, 0].contourf(lon_CAN, lat_CAN, m50_q_d / 1e-6, 
#                           levels=np.linspace(1.5, 5, 201),transform=ccrs.PlateCarree())
# axes[1, 0].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1, 0].set_title("Mean of q at 50 hPa")
# cbar2 = fig.colorbar(cf2, ax=axes[1, 0], orientation="vertical")
# cbar2.set_label(f"Mass fraction of water vapor ($10^{-6}$)")
# cbar2.set_ticks([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])


# #variance
# cf3 = axes[0,1].contourf(lon_CAN, lat_CAN, v100_q_d ,
#                          levels = levels, norm = norm, transform=ccrs.PlateCarree())
# # cl3 = axes[0,1].contour(lon_CAN, lat_CAN, m100_t_d,  colors="black", linewidths=0.8)
# # axes[0,1].clabel(cl3, inline=True, fontsize=8, fmt="%.0f K")
# axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,1].set_title("Variance of q at 100 hPa")
# cbar3 = fig.colorbar(cf3, ax=axes[0,1])
# cbar3.set_ticks([1e-15, 1e-14,1e-13,1e-12 ,1e-11])
# cbar3.set_ticklabels(["1e-15", "1e-14","1e-13", "1e-12","1e-11"])
# cf4 = axes[1,1].contourf(lon_CAN, lat_CAN, v50_q_d, 
#                          levels = levels, norm = norm, transform=ccrs.PlateCarree())
# # cl4 = axes[1,1].contour(lon_CAN, lat_CAN, m50_t_d,  colors="black", linewidths=0.8)
# # axes[1,1].clabel(cl4, inline=True, fontsize=8, fmt="%.0f K")
# axes[1,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,1].set_title("Variance of q at 50 hPa")
# cbar4 = fig.colorbar(cf4, ax=axes[1,1])
# cbar4.set_ticks([1e-15, 1e-14,1e-13,1e-12 ,1e-11])
# cbar4.set_ticklabels(["1e-15", "1e-14","1e-13", "1e-12","1e-11"])

# #Skew
# cf5 = axes[0,2].contourf(lon_CAN, lat_CAN, sk100q_d[0,:,:], 
#                          levels = np.arange(-5,5.2,0.2), cmap = "seismic", transform=ccrs.PlateCarree())
# # cl5 = axes[0,2].contour(lon_CAN, lat_CAN, m100_t_d,  colors="black", linewidths=0.8)
# # axes[0,2].clabel(cl5, inline=True, fontsize=8, fmt="%.0f K")
# axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,2].set_title("Skew of q at 100 hPa")
# cbar5 = fig.colorbar(cf5, ax=axes[0,2])
# cbar5.set_ticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
# cbar5.set_ticklabels(["","-4","","-2","","0","","2","","4",""])
# cf6 = axes[1,2].contourf(lon_CAN, lat_CAN, sk50q_d[0,:,:], 
#                          levels = np.arange(-5,5.2,0.2), cmap = "seismic", transform=ccrs.PlateCarree())
# # cl6 = axes[1,2].contour(lon_CAN, lat_CAN, m50_t_d,  colors="black", linewidths=0.8)
# # axes[1,2].clabel(cl6, inline=True, fontsize=8, fmt="%.0f K")
# axes[1,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,2].set_title("Skew of q at 50 hPa")
# cbar6= fig.colorbar(cf6, ax=axes[1,2])
# cbar6.set_ticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
# cbar6.set_ticklabels(["","-4","","-2","","0","","2","","4",""])

# #Kurtosis
# cf7 = axes[0,3].contourf(lon_CAN, lat_CAN, k100_q_d[0,:,:], 
#                          levels = np.arange(-10,10.2,0.2), cmap = "seismic", transform=ccrs.PlateCarree())
# # cl7 = axes[0,3].contour(lon_CAN, lat_CAN, m100_t_d,  colors="black", linewidths=0.8)
# # axes[0,3].clabel(cl7, inline=True, fontsize=8, fmt="%.0f K")
# axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,3].set_title("Kurtosis of q at 100 hPa")
# cbar7 = fig.colorbar(cf7, ax=axes[0,3])
# cbar7.set_ticks([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
# cbar7.set_ticklabels(["-10","","-8","","-6","","-4","","-2","","0","","2","","4","","6","","8","","10"])
# cf8 = axes[1,3].contourf(lon_CAN, lat_CAN, k50_q_d[0,:,:], 
#                          levels = np.arange(-10,10.2,0.2), cmap = "seismic", transform=ccrs.PlateCarree())
# # cl8 = axes[1,3].contour(lon_CAN, lat_CAN, m50_t_d,  colors="black", linewidths=0.8)
# # axes[1,3].clabel(cl8, inline=True, fontsize=8, fmt="%.0f K")
# axes[1,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,3].set_title("Kurtosis of q at 50 hPa")
# cbar8 = fig.colorbar(cf8, ax=axes[1,3])
# cbar8.set_ticks([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
# cbar8.set_ticklabels(["-10","","-8","","-6","","-4","","-2","","0","","2","","4","","6","","8","","10"])

# fig.suptitle("CanAM Daily Statistical Moments (1950-2014)", fontsize=15)
# plt.tight_layout()
# outpng = f"/home/karengarcia/MSc_project_backup/Seasonal_distributions/Seasonal_Distributions_1950-2014_amip.png"
# plt.savefig(outpng, dpi=300, bbox_inches='tight')
# plt.close(fig)


# vmin = 1e-15
# vmax = 1e-11
# norm = colors.LogNorm(vmin=vmin, vmax=vmax)
# levels = np.logspace(np.log10(vmin), np.log10(vmax))
# proj = ccrs.Sinusoidal(central_longitude=0)
# fig, axes = plt.subplots(nrows=2,ncols=4, sharex=True, sharey=True, figsize = (24,6) ,subplot_kw={'projection': proj})
# #Skew
# cf1 = axes[0,0].contourf(lon_CAN, lat_CAN, sk100q_d[0,:,:], 
#                          levels = np.linspace(0,4,201), cmap = "Reds", transform=ccrs.PlateCarree())
# axes[0,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,0].set_title("Positive Skew of q at 100 hPa")
# cbar1 = fig.colorbar(cf1, ax=axes[0,0])
# cbar1.set_ticks([0,1,2,3,4])
# cbar1.set_ticklabels(["0","1","2","3","4"])
# cf2 = axes[1,0].contourf(lon_CAN, lat_CAN, sk50q_d[0,:,:], 
#                          levels = np.linspace(0,5,201), cmap = "Reds", transform=ccrs.PlateCarree())
# axes[1,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,0].set_title("Positive Skew of q at 50 hPa")
# cbar2 = fig.colorbar(cf1, ax=axes[1,0])
# cbar2.set_ticks([0,1,2,3,4])
# cbar2.set_ticklabels(["0","1","2","3","4"])

# cf3 = axes[0,1].contourf(lon_CAN, lat_CAN, sk100q_d[0,:,:], 
#                          levels = np.linspace(-0.3,0,201), cmap = "Blues_r", transform=ccrs.PlateCarree())
# axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,1].set_title("Negative Skew of q at 100 hPa")
# cbar3 = fig.colorbar(cf3, ax=axes[0,1])
# cbar3.set_ticks([-0.3,-0.2,-0.1,0])
# cbar3.set_ticklabels(["-0.3","-0.2","-0.1" ,"0"])

# cf4 = axes[1,1].contourf(lon_CAN, lat_CAN, sk50q_d[0,:,:], 
#                          levels = np.linspace(-0.3,0,201), cmap = "Blues_r", transform=ccrs.PlateCarree())
# axes[1,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,1].set_title("Negative Skew of q at 50 hPa")
# cbar4 = fig.colorbar(cf4, ax=axes[1,1])
# cbar4.set_ticks([-0.3,-0.2,-0.1,0])
# cbar4.set_ticklabels(["-0.3","-0.2","-0.1" ,"0"])

# #Kurtosis
# cf5 = axes[0,2].contourf(lon_CAN, lat_CAN, k100_q_d[0,:,:], 
#                          levels = np.linspace(0,11,201), cmap = "Reds", transform=ccrs.PlateCarree())
# axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,2].set_title("Positive Kurtosis of q' at 100 hPa")
# cbar5 = fig.colorbar(cf5, ax=axes[0,2])
# cbar5.set_ticks([0,1,2,3,4,5,6,7,8,9,10])
# cbar5.set_ticklabels(["0","","2","","4","","6","","8","","10"])

# cf6 = axes[1,2].contourf(lon_CAN, lat_CAN, k50_q_d[0,:,:], 
#                          levels = np.linspace(0,11,201), cmap = "Reds", transform=ccrs.PlateCarree())
# axes[1,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,2].set_title("Positive Kurtosis of q' at 50 hPa")
# cbar6 = fig.colorbar(cf6, ax=axes[1,2])
# cbar6.set_ticks([0,1,2,3,4,5,6,7,8,9,10])
# cbar6.set_ticklabels(["0","","2","","4","","6","","8","","10"])

# cf7 = axes[0,3].contourf(lon_CAN, lat_CAN, k100_q_d[0,:,:], 
#                          levels = np.linspace(-2.1,0,201), cmap = "Blues_r", transform=ccrs.PlateCarree())
# axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,3].set_title("Negative Kurtosis of q' at 100 hPa")
# cbar7 = fig.colorbar(cf7, ax=axes[0,3])
# cbar7.set_ticks([-2,-1.5,-1,-0.5,0])
# cbar7.set_ticklabels(["-2","-1.5","-1","-0.5","0"])

# cf8 = axes[1,3].contourf(lon_CAN, lat_CAN, k50_q_d[0,:,:], 
#                          levels = np.linspace(-2.1,0,201), cmap = "Blues_r", transform=ccrs.PlateCarree())
# axes[1,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,3].set_title("Negative Kurtosis of q' at 50 hPa")
# cbar8 = fig.colorbar(cf8, ax=axes[1,3])
# cbar8.set_ticks([-2,-1.5,-1,-0.5,0])
# cbar8.set_ticklabels(["-2","-1.5","-1","-0.5","0"])

# fig.suptitle("CanAM Third and Fourth Moments (1941-2014)", fontsize=15)
# plt.tight_layout()
# outpng = f"/home/karengarcia/MSc_project_backup/Seasonal_distributions/Skew_Kurtosis_1950-2014_amip.png"
# plt.savefig(outpng, dpi=300, bbox_inches='tight')
# plt.close(fig)

def stat_plots_seasonal(q_50_daily, q_100_daily, lat= float, lon = float, location = 'str', first_year = int , last_year=int, title_name=str):
        """
        This function will create a 8 subplot figure for seasonal analysis of a specific lat and lon
        """
        DJF_100 = []
        MAM_100 = []
        JJA_100 = []
        SON_100 = []

        DJF_50 = []
        MAM_50 = []
        JJA_50 = []
        SON_50 = []

        fig = plt.figure(figsize=(10, 5))
        proj = ccrs.PlateCarree(central_longitude=0)
        # Set up the axes with the Sinusoidal projection
        ax = plt.axes(projection=proj)
        ax.plot(lon,lat, "ro")
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.coastlines()
        ax.set_global()
        outpng = f"/home/karengarcia/MSc_project_backup/Maps/{title_name}_map.png"
        plt.savefig(outpng, dpi=300, bbox_inches='tight')
        plt.close(fig)

        if lon<0:
                lon = lon+360

        years = np.arange(first_year,last_year)
        for year in years: 
                beginning_DJF = cftime.DatetimeNoLeap(year-1, 12, 1, 12, 0, 0, 0, has_year_zero=True)
                end_DJF = cftime.DatetimeNoLeap(year, 2, 28, 12, 0, 0, 0, has_year_zero=True)
                beginning_MAM = cftime.DatetimeNoLeap(year, 3, 1, 12, 0, 0, 0, has_year_zero=True)
                end_MAM =cftime.DatetimeNoLeap(year, 5, 31, 12, 0, 0, 0, has_year_zero=True)
                beginning_JJA = cftime.DatetimeNoLeap(year, 6, 1, 12, 0, 0, 0, has_year_zero=True)
                end_JJA = cftime.DatetimeNoLeap(year, 8, 31, 12, 0, 0, 0, has_year_zero=True)
                beginning_SON = cftime.DatetimeNoLeap(year, 8, 1, 12, 0, 0, 0, has_year_zero=True)
                end_SON = cftime.DatetimeNoLeap(year, 11, 30, 12, 0, 0, 0, has_year_zero=True)

                DJF_q_50 = q_50_daily.sel(time=slice(beginning_DJF, end_DJF)).sel(lat=lat,lon=lon, method="nearest")
                DJF_q_100 = q_100_daily.sel(time=slice(beginning_DJF, end_DJF)).sel(lat=lat,lon=lon, method="nearest")
                MAM_q_50 = q_50_daily.sel(time=slice(beginning_MAM, end_MAM)).sel(lat=lat,lon=lon, method="nearest")
                MAM_q_100 = q_100_daily.sel(time=slice(beginning_MAM, end_MAM)).sel(lat=lat,lon=lon, method="nearest")
                JJA_q_50 = q_50_daily.sel(time=slice(beginning_JJA, end_JJA)).sel(lat=lat,lon=lon, method="nearest")
                JJA_q_100 = q_100_daily.sel(time=slice(beginning_JJA, end_JJA)).sel(lat=lat,lon=lon, method="nearest")
                SON_q_50 = q_50_daily.sel(time=slice(beginning_SON, end_SON)).sel(lat=lat,lon=lon, method="nearest")
                SON_q_100 = q_100_daily.sel(time=slice(beginning_SON, end_SON)).sel(lat=lat,lon=lon, method="nearest")
                DJF_100.append(DJF_q_100)
                MAM_100.append(MAM_q_100)
                JJA_100.append(JJA_q_100)
                SON_100.append(SON_q_100)

                DJF_50.append(DJF_q_50)
                MAM_50.append(MAM_q_50)
                JJA_50.append(JJA_q_50)
                SON_50.append(SON_q_50)

        DJF_q_50 = xr.merge(DJF_50)
        DJF_q_100 = xr.merge(DJF_100)
        MAM_q_50 = xr.merge(MAM_50)
        MAM_q_100 = xr.merge(MAM_100)
        JJA_q_50 = xr.merge(JJA_50)
        JJA_q_100 = xr.merge(JJA_100)
        SON_q_50 = xr.merge(SON_50)
        SON_q_100 = xr.merge(SON_100)

        mean_DJF_q_100 = DJF_q_100.mean(dim='time')
        var_DJF_q_100 = DJF_q_100.var(dim='time')
        sk_DJF_q_100 = skew(DJF_q_100.hus, keepdims = True)
        kurt_DJF_q_100 = kurtosis(DJF_q_100.hus, keepdims = True) 
        mean_DJF_q_50 = DJF_q_50.mean(dim='time')
        var_DJF_q_50 = DJF_q_50.var(dim='time')
        sk_DJF_q_50 = skew(DJF_q_50.hus, keepdims = True)
        kurt_DJF_q_50 = kurtosis(DJF_q_50.hus, keepdims = True) 

        mean_MAM_q_100 = MAM_q_100.mean(dim='time')
        var_MAM_q_100 = MAM_q_100.var(dim='time')
        sk_MAM_q_100 = skew(MAM_q_100.hus, keepdims = True)
        kurt_MAM_q_100 = kurtosis(MAM_q_100.hus, keepdims = True) 
        mean_MAM_q_50 = MAM_q_50.mean(dim='time')
        var_MAM_q_50 = MAM_q_50.var(dim='time')
        sk_MAM_q_50 = skew(MAM_q_50.hus, keepdims = True)
        kurt_MAM_q_50 = kurtosis(MAM_q_50.hus, keepdims = True) 

        mean_JJA_q_100 = JJA_q_100.mean(dim='time')
        var_JJA_q_100 = JJA_q_100.var(dim='time')
        sk_JJA_q_100 = skew(JJA_q_100.hus, keepdims = True)
        kurt_JJA_q_100 = kurtosis(JJA_q_100.hus, keepdims = True) 
        mean_JJA_q_50 = JJA_q_50.mean(dim='time')
        var_JJA_q_50 = JJA_q_50.var(dim='time')
        sk_JJA_q_50 = skew(JJA_q_50.hus, keepdims = True)
        kurt_JJA_q_50 = kurtosis(JJA_q_50.hus, keepdims = True) 

        mean_SON_q_100 = SON_q_100.mean(dim='time')
        var_SON_q_100 = SON_q_100.var(dim='time')
        sk_SON_q_100 = skew(SON_q_100.hus, keepdims = True)
        kurt_SON_q_100 = kurtosis(SON_q_100.hus, keepdims = True) 
        mean_SON_q_50 = SON_q_50.mean(dim='time')
        var_SON_q_50 = SON_q_50.var(dim='time')
        sk_SON_q_50 = skew(SON_q_50.hus, keepdims = True)
        kurt_SON_q_50 = kurtosis(SON_q_50.hus, keepdims = True) 

        fig, axes = plt.subplots(nrows=2,ncols=4, sharex=True, sharey=True, figsize = (18,6))
        bin_width=0.2e-6
        num_bins = int(np.ceil((DJF_q_100['hus'].max().values - DJF_q_100['hus'].min().values)/bin_width))
        newax = fig.add_axes([0.85, 0.85, 0.2, 0.2], anchor='NE', zorder=-1)
        im = plt.imread(get_sample_data(outpng))
        newax.imshow(im)
        newax.axis('off')
        #### DJF## 
        axes[0,0].hist(DJF_q_100.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[0,0].set_title("100 hPa (DJF)")
        axes[0,0].set_ylabel("Frequency")
        axes[0,0].set_xlim([1e-6,10e-6])
        props1 = dict(boxstyle='round', facecolor='blue', alpha=0.1)
        textstr1 = f"Mean: {str(mean_DJF_q_100['hus'].values)} \nVariance: {str(var_DJF_q_100['hus'].values)} \nSkew: {str(round((sk_DJF_q_100[0]),3))} \nKurtosis: {str(round((kurt_DJF_q_100[0]),3))}"
        axes[0,0].text(0.45, 0.9, textstr1, transform=axes[0,0].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)
        num_bins = int(np.ceil((DJF_q_50['hus'].max().values - DJF_q_50['hus'].min().values)/(bin_width/2)))
        axes[1,0].hist(DJF_q_50.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[1,0].set_title("50 hPa (DJF)")
        axes[1,0].set_ylabel("Frequency")
        textstr2 = f"Mean: {str(mean_DJF_q_50['hus'].values)} \nVariance: {str(var_DJF_q_50['hus'].values)} \nSkew: {str(round((sk_DJF_q_50[0]),3))} \nKurtosis: {str(round((kurt_DJF_q_50[0]),3))}"
        axes[1,0].text(0.45, 0.9, textstr2, transform=axes[1,0].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)
        
        ## MAM ##
        num_bins = int(np.ceil((MAM_q_100['hus'].max().values - MAM_q_100['hus'].min().values)/bin_width))
        axes[0,1].hist(MAM_q_100.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[0,1].set_title("100 hPa (MAM)")
        textstr3 = f"Mean: {str(mean_MAM_q_100['hus'].values)} \nVariance: {str(var_MAM_q_100['hus'].values)} \nSkew: {str(round((sk_MAM_q_100[0]),3))} \nKurtosis: {str(round((kurt_MAM_q_100[0]),3))}"
        axes[0,1].text(0.45, 0.9, textstr3, transform=axes[0,1].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)
        num_bins = int(np.ceil((MAM_q_50['hus'].max().values - MAM_q_50['hus'].min().values)/(bin_width/2)))
        axes[1,1].hist(MAM_q_50.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[1,1].set_title("50 hPa (MAM)")
        textstr4 = f"Mean: {str(mean_MAM_q_50['hus'].values)} \nVariance: {str(var_MAM_q_50['hus'].values)} \nSkew: {str(round((sk_MAM_q_50[0]),3))} \nKurtosis: {str(round((kurt_MAM_q_50[0]),3))}"
        axes[1,1].text(0.45, 0.9, textstr4, transform=axes[1,1].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)

        ### JJA ### 
        num_bins = int(np.ceil((JJA_q_100['hus'].max().values - JJA_q_100['hus'].min().values)/bin_width))
        axes[0,2].hist(JJA_q_100.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[0,2].set_title("100 hPa(JJA)")
        textstr5 = f"Mean: {str(mean_JJA_q_100['hus'].values)} \nVariance: {str(var_JJA_q_100['hus'].values)}\nSkew: {str(round((sk_JJA_q_100[0]),3))} \nKurtosis: {str(round((kurt_JJA_q_100[0]),3))}"
        axes[0,2].text(0.45, 0.9, textstr5, transform=axes[0,2].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)
        num_bins = int(np.ceil((JJA_q_50['hus'].max().values - JJA_q_50['hus'].min().values)/(bin_width/2)))
        axes[1,2].hist(JJA_q_50.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[1,2].set_title("50 hPa (JJA)")
        textstr6 = f"Mean: {str(mean_JJA_q_50['hus'].values)} \nVariance: {str(var_JJA_q_50['hus'].values)} \nSkew: {str(round((sk_JJA_q_50[0]),3))} \nKurtosis: {str(round((kurt_JJA_q_50[0]),3))}"
        axes[1,2].text(0.43, 0.9, textstr6, transform=axes[1,2].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)

        ### SON ### 
        num_bins = int(np.ceil((SON_q_100['hus'].max().values - SON_q_100['hus'].min().values)/bin_width))
        axes[0,3].hist(SON_q_100.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[0,3].set_title("100 hPa (SON)")
        textstr7 = f"Mean: {str(mean_SON_q_100['hus'].values)} \nVariance: {str(var_SON_q_100['hus'].values)}\nSkew: {str(round((sk_SON_q_100[0]),3))} \nKurtosis: {str(round((kurt_SON_q_100[0]),3))}"
        axes[0,3].text(0.45, 0.9, textstr7, transform=axes[0,3].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)
        num_bins = int(np.ceil((SON_q_50['hus'].max().values - SON_q_50['hus'].min().values)/(bin_width/2)))
        axes[1,3].hist(SON_q_50.hus, bins=num_bins, facecolor='blue', alpha=0.5)
        axes[1,3].set_title("50 hPa (SON)")
        textstr8 = f"Mean: {str(mean_SON_q_50['hus'].values)} \nVariance: {str(var_SON_q_50['hus'].values)} \nSkew: {str(round((sk_SON_q_50[0]),3))} \nKurtosis: {str(round((kurt_SON_q_50[0]),3))}"
        axes[1,3].text(0.43, 0.9, textstr8, transform=axes[1,3].transAxes, fontsize=9,
                verticalalignment='top', bbox=props1)
        fig.suptitle(f"Distribution of Specific Humidity in {location} between {str(first_year)} and {str(last_year-1)}, lat = {lat}, lon = {lon}", fontsize = 20)
        outpng = f"/home/karengarcia/MSc_project_backup/Seasonal_distributions/Seasonal_Distributions_{str(first_year)}-{str(last_year-1)}_{title_name}_amip.png"
        plt.savefig(outpng, dpi=300, bbox_inches='tight')
        plt.close(fig)

stat_plots_seasonal(q_50_daily, q_100_daily, 22.8,  -88,   "the Gulf of Mexico",2000 ,2015, "GM")
stat_plots_seasonal(q_50_daily, q_100_daily, -48,   -35,      "the East Coast of Argentina",2000 ,2015,"E_Argentina")
stat_plots_seasonal(q_50_daily, q_100_daily, 30,    31,       "Cairo",2000 ,2015, "Cairo")
stat_plots_seasonal(q_50_daily, q_100_daily, 35,    -12,      "the  Eastern Coast of Morocco",2000 ,2015, "Morocco")
stat_plots_seasonal(q_50_daily, q_100_daily, 29,    15.8,     "Northern Libya",2000 ,2015, "Jufra")
stat_plots_seasonal(q_50_daily, q_100_daily, 23.7,  90.4,     "Bangladesh",2000 ,2015, "BG")
stat_plots_seasonal(q_50_daily, q_100_daily, 28,    87,       "Everest",2000 ,2015, "Everest")
stat_plots_seasonal(q_50_daily, q_100_daily, 8.87,  125.7,    "Timor-Leste",2000 ,2015, "Timor-Leste")
stat_plots_seasonal(q_50_daily, q_100_daily, 10.69, -61.2, "Trinidad and Tobago",2000 ,2015, "TandT")
stat_plots_seasonal(q_50_daily, q_100_daily, 40.4, 49.4, "Baku, Azerbaijan", 2000, 2015, "Baku")
stat_plots_seasonal(q_50_daily, q_100_daily, 37.5, 56.3, "Northwest Iran", 2000, 2015, "Iran")
stat_plots_seasonal(q_50_daily, q_100_daily, 37.5, 56.3, "Western Turkey", 2000, 2015, "Turkey")
stat_plots_seasonal(q_50_daily, q_100_daily, 37.5, 56.3, "Western Turkey", 2000, 2015, "Turkey")
stat_plots_seasonal(q_50_daily, q_100_daily, -4,   -62,   "Amazon", 2000, 2015, "Amazon")




# ###### COMPARING ERA AND CanESM5 ###########

# ERA_monthly_file = "/home/karengarcia/downloads-karengarcia/ERA5/ERA5_specific_hum.nc"
# CanAM_monthly_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Monthly/hus_Amon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc"
# ERA_monthly = xr.open_dataset(ERA_monthly_file).q
# CanAM_monthly = (xr.open_dataset(CanAM_monthly_file).hus).isel(time=slice(-11*12,None))
# # display(CanAM_monthly)

# years = [str(year) for year in np.arange(2004,2015,1)]

# full_water=[]
# full_time=[]

# for year in years:
#     for file_path in glob.glob(f"/home/karengarcia/downloads-karengarcia/MLS_data/*{year}.nc"):
#         with Dataset(file_path, 'r') as ds:
            
#             data =      ds.groups['H2O PressureGrid'] 
#             # potK_data = ds.groups["H2O ThetaGrid"] 
#             #grabbing data
#             MLS_lon =   data.variables['lon'][:]             #units = "degrees_east" (72 values)
#             MLS_lat =   data.variables['lat'][:]             #units = "degrees_north" (45 values)
#             time =      data.variables['time'][:]            #units = "days since 1950-01-01" (12 values)
#             water =     data.variables["value"][:]           #value(time, lev, lon, lat)
#             pres =      data.variables["lev"][:]             #units = "hPa" (values 45) Pressure
            
#             # changing time from dates into date times because it hates me apparently
#             dates = []
#             start = date(1950,1,1) #starting date
#             for t in time:  
#                 delta = timedelta(days=float(t))
#                 dates.append(start + delta)
#             full_water.append(water)  #making an array with all the time values in datetime
#             full_time.append(dates)

# time_flat = np.concatenate(full_time, axis=0)
# MLS_monthly = np.concatenate(full_water, axis=0)  # shape: (total_time, lev, lon, lat) 

# #100hPA 
# CanAM_100 = CanAM_monthly.sel(plev=100*100)
# # display(CanAM_100)
# ERA5_100 = ERA_monthly.sel(pressure_level=100)
# MLS_100 = MLS_monthly[:,12,:,:]

# #70hPa 
# CanAM_70 = CanAM_monthly.sel(plev=70*100)
# # display(CanAM_100)
# ERA5_70 = ERA_monthly.sel(pressure_level=70)
# MLS_70 = MLS_monthly[:,12,:,:] #68.1292hPa

# #50hPa 
# CanAM_50 = CanAM_monthly.sel(plev=50*100)
# ERA5_50 = ERA_monthly.sel(pressure_level=50)
# MLS_50 = MLS_monthly[:,15,:,:] 

# #CanAM Stats 
# mean_CanAM_q_100 = CanAM_100.mean(dim='time')
# var_CanAM_q_100 = CanAM_100.var(dim='time')
# sk_CanAM_q_100 = skew(CanAM_100, keepdims = True)
# kurt_CanAM_q_100 = kurtosis(CanAM_100, keepdims = True) 

# mean_CanAM_q_70 = CanAM_70.mean(dim='time')
# var_CanAM_q_70 = CanAM_70.var(dim='time')
# sk_CanAM_q_70 = skew(CanAM_70, keepdims = True)
# kurt_CanAM_q_70 = kurtosis(CanAM_70, keepdims = True) 

# mean_CanAM_q_50 = CanAM_50.mean(dim='time')
# var_CanAM_q_50 = CanAM_50.var(dim='time')
# sk_CanAM_q_50 = skew(CanAM_50, keepdims = True)
# kurt_CanAM_q_50 = kurtosis(CanAM_50, keepdims = True) 


# #ERA Stats 
# mean_ERA_q_100 = ERA5_100.mean(dim='valid_time')
# var_ERA_q_100 = ERA5_100.var(dim='valid_time')
# sk_ERA_q_100 = skew(ERA5_100, keepdims = True)
# kurt_ERA_q_100 = kurtosis(ERA5_100, keepdims = True) 

# mean_ERA_q_70 = ERA5_70.mean(dim='valid_time')
# var_ERA_q_70 = ERA5_70.var(dim='valid_time')
# sk_ERA_q_70 = skew(ERA5_70, keepdims = True)
# kurt_ERA_q_70 = kurtosis(ERA5_70, keepdims = True) 

# mean_ERA_q_50 = ERA5_50.mean(dim='valid_time')
# var_ERA_q_50 = ERA5_50.var(dim='valid_time')
# sk_ERA_q_50 = skew(ERA5_50, keepdims = True)
# kurt_ERA_q_50 = kurtosis(ERA5_50, keepdims = True)

# #MLS Stats
# mean_MLS_q_100 = np.mean(MLS_100, axis =0) 
# var_MLS_q_100 = np.var(MLS_100, axis =0) 
# sk_MLS_q_100 = skew(MLS_100, axis =0) 
# kurt_MLS_q_100 = kurtosis(MLS_100, axis =0) 

# mean_MLS_q_70 = np.mean(MLS_70, axis =0) 
# var_MLS_q_70 = np.var(MLS_70, axis =0)
# sk_MLS_q_70 = skew(MLS_70, axis =0) 
# kurt_MLS_q_70 = kurtosis(MLS_70, axis =0)  

# mean_MLS_q_50  = np.mean(MLS_50, axis =0) 
# var_MLS_q_50 = np.var(MLS_50, axis =0) 
# sk_MLS_q_50 = skew(MLS_50, axis =0) 
# kurt_MLS_q_50 = kurtosis(MLS_50, axis =0) 

# lon_CAN = CanAM_100.lon
# lat_CAN = CanAM_100.lat 
# lat_ERA = ERA5_100.latitude 
# lon_ERA = ERA5_100.longitude
# lon_MLS = MLS_lon 
# lat_MLS = MLS_lat 


# vmin = 1e-15
# vmax = 1e-11
# norm = colors.LogNorm(vmin=vmin, vmax=vmax)
# levels = np.logspace(np.log10(vmin), np.log10(vmax))
# proj = ccrs.Sinusoidal(central_longitude=0)
# fig, axes = plt.subplots(nrows=2,ncols=4, figsize = (36,11) ,subplot_kw={'projection': proj},layout="constrained")

# # cf1 = axes[0,0].contourf(lon_MLS, lat_MLS, mean_MLS_q_100.T, levels = np.arange(np.min(mean_MLS_q_100),7,0.1))
# # cbar1 = fig.colorbar(cf1, ax=axes[0,0])
# # axes[0,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf1 = axes[0,0].contourf(lon_ERA, lat_ERA, mean_ERA_q_100*(10**6), levels = np.linspace(1,4.2,201))
# axes[0,0].set_title("ERA5 Mean", fontsize=15)
# cbar1 = fig.colorbar(cf1, ax=axes[0,0])
# cbar1.set_ticks([1,2,3,4])
# cbar1.set_ticklabels(["1","2","3","4"])
# axes[0,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf2 = axes[1,0].contourf(lon_CAN, lat_CAN, mean_CanAM_q_100*(10**6), 
#                          levels = np.linspace(1,4.2,201))
# axes[1,0].set_title("CanESM5 Mean", fontsize=15)
# cbar2 = fig.colorbar(cf2, ax=axes[1,0])
# cbar2.set_ticks([1,2,3,4])
# cbar2.set_ticklabels(["1","2","3","4"])
# axes[1,0].coastlines(color='black', linewidth=0.5, linestyle='--')

# # axes[0,1].contourf(lon_MLS, lat_MLS, var_MLS_q_100.T)
# # axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf3 = axes[0,1].contourf(lon_ERA, lat_ERA, var_ERA_q_100 , 
#                          levels = levels, norm=norm)
# axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,1].set_title('ERA5 Variance', fontsize=15)
# cbar3 = fig.colorbar(cf3, ax=axes[0,1])
# cbar3.set_ticks([1e-15,1e-14,1e-13,1e-12 ,1e-11])
# cbar3.set_ticklabels([f"1e-15",f"1e-14",f"1e-13",f"1e-12",f"1e-11"])
# cf4 = axes[1,1].contourf(lon_CAN, lat_CAN, var_CanAM_q_100, 
#                          levels = levels, norm=norm)
# axes[1,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,1].set_title("CanESM5 Variance", fontsize=15)
# cbar4 = fig.colorbar(cf3, ax=axes[1,1])
# cbar4.set_ticks([1e-15,1e-14,1e-13,1e-12 ,1e-11])
# cbar4.set_ticklabels([f"1e-15",f"1e-14",f"1e-13",f"1e-12",f"1e-11"])

# # axes[0,2].contourf(lon_MLS, lat_MLS, sk_MLS_q_100.T)
# # axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf5 = axes[0,2].contourf(lon_ERA, lat_ERA, sk_ERA_q_100[0,:,:], 
#                          levels=np.linspace(-2,2,101), cmap = "seismic")
# axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,2].set_title("ERA5 Skew", fontsize=15)
# cbar5 = fig.colorbar(cf5, ax=axes[0,2])
# cbar5.set_ticks([-2,-1,0,1,2])
# cbar5.set_ticklabels(["-2","-1","0","1","2"])
# cf6 = axes[1,2].contourf(lon_CAN, lat_CAN, sk_CanAM_q_100[0,:,:], 
#                          levels=np.linspace(-2,2,101), cmap = "seismic")
# axes[1,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,2].set_title("CanESM5 Skew", fontsize=15)
# cbar6 = fig.colorbar(cf6, ax=axes[1,2])
# cbar6.set_ticks([-2,-1,0,1,2])
# cbar6.set_ticklabels(["-2","-1","0","1","2"])

# # axes[0,3].contourf(lon_MLS, lat_MLS, kurt_MLS_q_100.T)
# # axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf7 = axes[0,3].contourf(lon_ERA, lat_ERA, kurt_ERA_q_100[0,:,:], 
#                          levels=np.linspace(-4,4,101), cmap = "seismic")
# axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,3].set_title("ERA5 Kurtosis", fontsize=15)
# cbar7 = fig.colorbar(cf7, ax=axes[0,3])
# cbar7.set_ticks([-4,-3,-2,-1,0,1,2,3,4])
# cbar7.set_ticklabels(["4","3","-2","-1","0","1","2","3","4"])
# cf8 = axes[1,3].contourf(lon_CAN, lat_CAN, kurt_CanAM_q_100[0,:,:], 
#                          levels=np.linspace(-4,4,101), cmap = "seismic")
# axes[1,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,3].set_title("CanESM5 Kurtosis", fontsize=15)
# cbar8 = fig.colorbar(cf8, ax=axes[1,3])
# cbar8.set_ticks([-4,-3,-2,-1,0,1,2,3,4])
# cbar8.set_ticklabels(["4","3","-2","-1","0","1","2","3","4"])
# # plt.tight_layout()
# fig.suptitle("Statistical Moments of Monthly Data at 100 hPa (2004-2014)", fontsize=15)
# outpng = f"/home/karengarcia/MSc_project_backup/ERA_CanESM_monthly_moments_100hPa_2004-2014.png"
# plt.savefig(outpng, dpi=300, bbox_inches='tight')


# proj = ccrs.Sinusoidal(central_longitude=0)
# fig, axes = plt.subplots(nrows=2,ncols=4, figsize = (36,11) ,subplot_kw={'projection': proj},layout="constrained")

# # cf1 = axes[0,0].contourf(lon_MLS, lat_MLS, mean_MLS_q_100.T, levels = np.arange(np.min(mean_MLS_q_100),7,0.1))
# # cbar1 = fig.colorbar(cf1, ax=axes[0,0])
# # axes[0,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf1 = axes[0,0].contourf(lon_ERA, lat_ERA, mean_ERA_q_70*(10**6), 
#                          levels = np.linspace(1,4.2,201))
# axes[0,0].set_title("ERA5 Mean", fontsize=15)
# cbar1 = fig.colorbar(cf1, ax=axes[0,0])
# cbar1.set_ticks([1,2,3,4])
# cbar1.set_ticklabels(["1","2","3","4"])
# axes[0,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf2 = axes[1,0].contourf(lon_CAN, lat_CAN, mean_CanAM_q_70*(10**6), 
#                          levels = np.linspace(1,4.2,201))
# axes[1,0].set_title("CanESM5 Mean", fontsize=15)
# cbar2 = fig.colorbar(cf2, ax=axes[1,0])
# cbar2.set_ticks([1,2,3,4])
# cbar2.set_ticklabels(["1","2","3","4"])
# axes[1,0].coastlines(color='black', linewidth=0.5, linestyle='--')

# # axes[0,1].contourf(lon_MLS, lat_MLS, var_MLS_q_100.T)
# # axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf3 = axes[0,1].contourf(lon_ERA, lat_ERA, var_ERA_q_70 , 
#                          levels = levels, norm=norm)
# axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,1].set_title('ERA5 Variance', fontsize=15)
# cbar3 = fig.colorbar(cf3, ax=axes[0,1])
# cbar3.set_ticks([1e-15,1e-14,1e-13,1e-12 ,1e-11])
# cbar3.set_ticklabels([f"1e-15",f"1e-14",f"1e-13",f"1e-12",f"1e-11"])
# cf4 = axes[1,1].contourf(lon_CAN, lat_CAN, var_CanAM_q_70, 
#                          levels = levels, norm=norm)
# axes[1,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,1].set_title("CanESM5 Variance", fontsize=15)
# cbar4 = fig.colorbar(cf3, ax=axes[1,1])
# cbar4.set_ticks([1e-15,1e-14,1e-13,1e-12 ,1e-11])
# cbar4.set_ticklabels([f"1e-15",f"1e-14",f"1e-13",f"1e-12",f"1e-11"])

# # axes[0,2].contourf(lon_MLS, lat_MLS, sk_MLS_q_100.T)
# # axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf5 = axes[0,2].contourf(lon_ERA, lat_ERA, sk_ERA_q_70[0,:,:], 
#                          levels=np.linspace(-2,2,101), cmap = "seismic")
# axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,2].set_title("ERA5 Skew", fontsize=15)
# cbar5 = fig.colorbar(cf5, ax=axes[0,2])
# cbar5.set_ticks([-2,-1,0,1,2])
# cbar5.set_ticklabels(["-2","-1","0","1","2"])
# cf6 = axes[1,2].contourf(lon_CAN, lat_CAN, sk_CanAM_q_70[0,:,:], 
#                          levels=np.linspace(-2,2,101), cmap = "seismic")
# axes[1,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,2].set_title("CanESM5 Skew", fontsize=15)
# cbar6 = fig.colorbar(cf6, ax=axes[1,2])
# cbar6.set_ticks([-2,-1,0,1,2])
# cbar6.set_ticklabels(["-2","-1","0","1","2"])

# # axes[0,3].contourf(lon_MLS, lat_MLS, kurt_MLS_q_100.T)
# # axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf7 = axes[0,3].contourf(lon_ERA, lat_ERA, kurt_ERA_q_70[0,:,:], 
#                          levels=np.linspace(-4,4,101), cmap = "seismic")
# axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,3].set_title("ERA5 Kurtosis", fontsize=15)
# cbar7 = fig.colorbar(cf7, ax=axes[0,3])
# cbar7.set_ticks([-4,-3,-2,-1,0,1,2,3,4])
# cbar7.set_ticklabels(["4","3","-2","-1","0","1","2","3","4"])
# cf8 = axes[1,3].contourf(lon_CAN, lat_CAN, kurt_CanAM_q_70[0,:,:], 
#                          levels=np.linspace(-4,4,101), cmap = "seismic")
# axes[1,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,3].set_title("CanESM5 Kurtosis", fontsize=15)
# cbar8 = fig.colorbar(cf8, ax=axes[1,3])
# cbar8.set_ticks([-4,-3,-2,-1,0,1,2,3,4])
# cbar8.set_ticklabels(["4","3","-2","-1","0","1","2","3","4"])
# # plt.tight_layout()
# fig.suptitle("Statistical Moments of Monthly Data at 70 hPa (1941-2014)", fontsize=15)
# outpng = f"/home/karengarcia/MSc_project_backup/ERA_CanESM_monthly_moments_70hPa_2004-2014.png"
# plt.savefig(outpng, dpi=300, bbox_inches='tight')


# proj = ccrs.Sinusoidal(central_longitude=0)
# fig, axes = plt.subplots(nrows=2,ncols=4, figsize = (36,11), subplot_kw={'projection': proj},layout="constrained")

# # cf1 = axes[0,0].contourf(lon_MLS, lat_MLS, mean_MLS_q_100.T, levels = np.arange(np.min(mean_MLS_q_100),7,0.1))
# # cbar1 = fig.colorbar(cf1, ax=axes[0,0])
# # axes[0,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf1 = axes[0,0].contourf(lon_ERA, lat_ERA, mean_ERA_q_50*(10**6), 
#                          levels = np.linspace(1,4.2,201))
# axes[0,0].set_title("ERA5 Mean", fontsize=15)
# cbar1 = fig.colorbar(cf1, ax=axes[0,0])
# cbar1.set_ticks([1,2,3,4])
# cbar1.set_ticklabels(["1","2","3","4"])
# axes[0,0].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf2 = axes[1,0].contourf(lon_CAN, lat_CAN, mean_CanAM_q_50*(10**6), 
#                          levels = np.linspace(1,4.2,201))
# axes[1,0].set_title("CanESM5 Mean", fontsize=15)
# cbar2 = fig.colorbar(cf2, ax=axes[1,0])
# cbar2.set_ticks([1,2,3,4])
# cbar2.set_ticklabels(["1","2","3","4"])
# axes[1,0].coastlines(color='black', linewidth=0.5, linestyle='--')

# # axes[0,1].contourf(lon_MLS, lat_MLS, var_MLS_q_100.T)
# # axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf3 = axes[0,1].contourf(lon_ERA, lat_ERA, var_ERA_q_50 , 
#                          levels = levels, norm=norm)
# axes[0,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,1].set_title('ERA5 Variance', fontsize=15)
# cbar3 = fig.colorbar(cf3, ax=axes[0,1])
# cbar3.set_ticks([1e-15,1e-14,1e-13,1e-12 ,1e-11])
# cbar3.set_ticklabels([f"1e-15",f"1e-14",f"1e-13",f"1e-12",f"1e-11"])
# cf4 = axes[1,1].contourf(lon_CAN, lat_CAN, var_CanAM_q_50, 
#                          levels = levels, norm=norm)
# axes[1,1].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,1].set_title("CanESM5 Variance", fontsize=15)
# cbar4 = fig.colorbar(cf3, ax=axes[1,1])
# cbar4.set_ticks([1e-15,1e-14,1e-13,1e-12 ,1e-11])
# cbar4.set_ticklabels([f"1e-15",f"1e-14",f"1e-13",f"1e-12",f"1e-11"])

# # axes[0,2].contourf(lon_MLS, lat_MLS, sk_MLS_q_100.T)
# # axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf5 = axes[0,2].contourf(lon_ERA, lat_ERA, sk_ERA_q_50[0,:,:], 
#                          levels=np.linspace(-2,2,101), cmap = "seismic")
# axes[0,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,2].set_title("ERA5 Skew", fontsize=15)
# cbar5 = fig.colorbar(cf5, ax=axes[0,2])
# cbar5.set_ticks([-2,-1,0,1,2])
# cbar5.set_ticklabels(["-2","-1","0","1","2"])
# cf6 = axes[1,2].contourf(lon_CAN, lat_CAN, sk_CanAM_q_50[0,:,:], 
#                          levels=np.linspace(-2,2,101), cmap = "seismic")
# axes[1,2].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,2].set_title("CanESM5 Skew", fontsize=15)
# cbar6 = fig.colorbar(cf6, ax=axes[1,2])
# cbar6.set_ticks([-2,-1,0,1,2])
# cbar6.set_ticklabels(["-2","-1","0","1","2"])

# # axes[0,3].contourf(lon_MLS, lat_MLS, kurt_MLS_q_100.T)
# # axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# cf7 = axes[0,3].contourf(lon_ERA, lat_ERA, kurt_ERA_q_50[0,:,:], 
#                          levels=np.linspace(-4,4,101), cmap = "seismic")
# axes[0,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[0,3].set_title("ERA5 Kurtosis", fontsize=15)
# cbar7 = fig.colorbar(cf7, ax=axes[0,3])
# cbar7.set_ticks([-4,-3,-2,-1,0,1,2,3,4])
# cbar7.set_ticklabels(["4","3","-2","-1","0","1","2","3","4"])
# cf8 = axes[1,3].contourf(lon_CAN, lat_CAN, kurt_CanAM_q_50[0,:,:], 
#                          levels=np.linspace(-4,4,101), cmap = "seismic")
# axes[1,3].coastlines(color='black', linewidth=0.5, linestyle='--')
# axes[1,3].set_title("CanESM5 Kurtosis", fontsize=15)
# cbar8 = fig.colorbar(cf8, ax=axes[1,3])
# cbar8.set_ticks([-4,-3,-2,-1,0,1,2,3,4])
# cbar8.set_ticklabels(["4","3","-2","-1","0","1","2","3","4"])
# # plt.tight_layout()
# fig.suptitle("Statistical Moments of Monthly Data at 50 hPa (2004-2014)", fontsize=15)
# outpng = f"/home/karengarcia/MSc_project_backup/ERA_CanESM_monthly_moments_50hPa_2004-2014.png"
# plt.savefig(outpng, dpi=300, bbox_inches='tight')
