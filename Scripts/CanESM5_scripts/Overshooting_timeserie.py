import numpy as np
import xarray as xr
import os
import glob
import matplotlib.pyplot as plt

chunks_M = {"lat":  361, 
          "lon":  576,
          "time": 40}

chunks_E = {"latitude": 361, 
            "longitude":  576}

MERRA_folder = "/home/karengarcia/data-karengarcia/Overshooting/MERRA2/"
ERA_folder = "/home/karengarcia/data-karengarcia/Overshooting/ERA5/Hourly"

eightmm_files_M = sorted(glob.glob(os.path.join(MERRA_folder,"8mm/MERRA_overshoot_8mm_2014*.nc" )))
fourmm_files_M  = sorted(glob.glob(os.path.join(MERRA_folder,"4mm/MERRA_overshoot_4mm_2014*.nc" )))

eightmm_files_E = sorted(glob.glob(os.path.join(ERA_folder,"8mm/ERA_overshoot_2014_*_8mm.nc" )))
fourmm_files_E  = sorted(glob.glob(os.path.join(ERA_folder,"4mm/ERA_overshoot_2014_*_4mm.nc" )))

overshoots_8mm_M = xr.open_mfdataset(eightmm_files_M, chunks=chunks_M)
overshoots_4mm_M = xr.open_mfdataset( fourmm_files_M, chunks=chunks_M)

overshoots_8mm_E = (xr.open_mfdataset(eightmm_files_E, combine='nested', concat_dim='valid_time')
                    .sortby('valid_time', ascending=True))
overshoots_8mm_E = overshoots_8mm_E.sortby('valid_time', ascending=True)
overshoots_4mm_E = (xr.open_mfdataset( fourmm_files_E, combine='nested', concat_dim='valid_time')
                    .sortby('valid_time', ascending=True))
overshoots_4mm_E = overshoots_4mm_E.sortby('valid_time', ascending=True)

rolling_ave8mm_M = (overshoots_8mm_M.sum(dim=['lat', 'lon'])).rolling(time=8).mean()
rolling_ave4mm_M = (overshoots_4mm_M.sum(dim=['lat', 'lon'])).rolling(time=8).mean()

rolling_ave8mm_E = (overshoots_8mm_E.sum(dim=['latitude', 'longitude'])).rolling(valid_time=24).mean()
rolling_ave4mm_E = (overshoots_4mm_E.sum(dim=['latitude', 'longitude'])).rolling(valid_time=24).mean()

fig, ax = plt.subplots(nrows =2, ncols=1, sharex=True,figsize = (15,5))
ax[0].plot(rolling_ave8mm_M['time'],       rolling_ave8mm_M['overshoot'], "k-")
ax[0].set_ylabel("Number of events")
ax[0].set_title("MERRA2")
ax[1].plot(rolling_ave8mm_E['valid_time'], rolling_ave8mm_E['overshoot'], "b-")
ax[1].set_ylabel("Number of events")
ax[1].set_xlabel("Date (YYYY-MM)")
ax[1].set_title("ERA5")
fig.suptitle("Timeseries of Overshooting Events (8mm/day Precipitation Threshold)", fontsize =16)
outpng = f"/home/karengarcia/Reanalysis_8mm_timeseries.png"
plt.savefig(outpng, dpi=300, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(nrows =2, ncols=1, sharex=True,figsize = (15,5))
ax[0].plot(rolling_ave4mm_M['time'],       rolling_ave4mm_M['overshoot'], "k-")
ax[0].set_ylabel("Number of events")
ax[0].set_title("MERRA2")
ax[1].plot(rolling_ave4mm_E['valid_time'], rolling_ave4mm_E['overshoot'], "b-")
ax[1].set_ylabel("Number of events")
ax[1].set_xlabel("Date (YYYY-MM)")
ax[1].set_title("ERA5")
fig.suptitle("Timeseries of Overshooting Events (4mm/day Precipitation Threshold)", fontsize =16)
outpng = f"/home/karengarcia/Reanalysis_4mm_timeseries.png"
plt.savefig(outpng, dpi=300, bbox_inches='tight')
plt.close(fig)