import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import moviepy.video.io.ImageSequenceClip

chunk_div = 20
num_years = 3

##################
# UTLS Boundaries
top_boundary = 50 * 100       # hPa -> Pa
bottom_boundary = 400 * 100   # hPa -> Pa
##################

target_lapserate = 2.0        # K / km

g = 9.81
R_ideal = 8.314               # Pa m^3 / (mol K)
m_dryair = 28.97 / 1000       # kg / mol
ref_press = 1013.25 * 100     # Pa
R_dry_air = 287.05            # J / (kg K)

# q_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/6hourly/hus_6hrLev_CanESM5_historical_r1i1p2f1_gn_201401010000-201412311800.nc"
# temp10_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/6hourly/ta_6hrLev_CanESM5_historical_r1i1p2f1_gn_201001010000-201012311800.nc"
# temp11_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/6hourly/ta_6hrLev_CanESM5_historical_r1i1p2f1_gn_201101010000-201112311800.nc"
# temp12_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/6hourly/ta_6hrLev_CanESM5_historical_r1i1p2f1_gn_201201010000-201212311800.nc"
# temp13_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/6hourly/ta_6hrLev_CanESM5_historical_r1i1p2f1_gn_201301010000-201312311800.nc"
temp14_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/6hourly/ta_6hrLev_CanESM5_historical_r1i1p2f1_gn_201401010000-201412311800.nc"

# ta10 = xr.open_dataset(temp10_file).chunk({"time": chunk_div})
# ta11 = xr.open_dataset(temp11_file).chunk({"time": chunk_div})
# ta12 = xr.open_dataset(temp12_file).chunk({"time": chunk_div})
# ta13 = xr.open_dataset(temp13_file).chunk({"time": chunk_div})
ta14 = xr.open_dataset(temp14_file).chunk({"time": chunk_div})

ta = xr.merge([ta12, ta13, ta14])
# hus = xr.open_dataset(q_file).chunk({"time": chunk_div})

# Hybrid pressure calculation
ta["plev"] = ta["ap"] + ta["b"] * ta["ps"]
# hus["plev"] = hus["ap"] + hus["b"] * hus["ps"]

# UTLS mask
plev_mask = ((ta["plev"] >= top_boundary) &
             (ta["plev"] <= bottom_boundary)).compute()

UTLS_ta = ta.where(plev_mask, drop=True)
# UTLS_q  = hus.where(plev_mask, drop=True)

print("Selected the UTLS?")
print("You can continue now?")

dT = UTLS_ta.ta.diff('lev')
dP = UTLS_ta['plev'].diff('lev')

T_low  = UTLS_ta.ta.isel(lev=slice(0, -1))
T_high = UTLS_ta.ta.isel(lev=slice(1, None)).assign_coords(lev=T_low.lev)
T_mid  = (T_low + T_high) / 2

P_low  = UTLS_ta['plev'].isel(lev=slice(0, -1))
P_high = UTLS_ta['plev'].isel(lev=slice(1, None)).assign_coords(lev=P_low.lev)
P_mid  = (P_low + P_high) / 2

# Hydrostatic dz
dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
lapse_rate = - (dT / dz) * 1000

# Tropopause mask
tropopause_mask  = lapse_rate <= target_lapserate
tropopause_index = tropopause_mask.argmax(dim="lev").compute()

tropopause_temp     = UTLS_ta.isel(lev=tropopause_index)
tropopause_pressure = UTLS_ta['plev'].isel(lev=tropopause_index)

print("Tropopause calculation successful")
print("Attempting overshooting calculation now")

lon_CAN = ta.lon
lat_CAN = ta.lat

pctisccp_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/pctisccp_CFday_CanESM5_historical_r1i1p2f1_gn_18500101-20141231.nc"
convec_file   = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/prc/prc_day_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc"

# Load last year (2014)
prc = xr.open_dataset(convec_file).isel(time=slice(-365*num_years, None))
cloud_pressure = xr.open_dataset(pctisccp_file).isel(time=slice(-365*num_years, None))

# Convert 6-hourly tropopause to daily (every 4th timestep) (Daily)
tropopause_pressure_daily = tropopause_pressure.isel(time=slice(2, None, 4))

ctp = cloud_pressure.pctisccp

#Boolean mask if prc>1mm/day then true and therefore it there is convective precip and tropopause is below cloud top
above_tropopause_and_prc = (ctp < tropopause_pressure_daily) & ( prc["prc"] * 86400 >= 8)

pctisccp_above_tp = ctp.where(above_tropopause_and_prc)

overshooting_count = above_tropopause_and_prc.sum(dim='time')


# Mask zeros so they remain white on plot
zero_mask = np.ma.masked_where(overshooting_count <= 0, overshooting_count) 

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(1, 1, figsize=(24, 8), subplot_kw={'projection': proj})
ax.coastlines(color='black', linewidth=0.6, linestyle='--')
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

# Plotting with 'extend' adds an arrow to the colorbar for values > 350
cf1 = ax.pcolormesh(lon_CAN, lat_CAN, zero_mask, 
                    transform=ccrs.PlateCarree(), 
                    cmap="Blues")

gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
gl.top_labels = False
gl.right_labels = False

# Colorbar
cbar1 = fig.colorbar(cf1, ax=ax, orientation='vertical')
cbar1.set_label("Overshooting events (Count)", fontsize=14)
cbar1.set_ticks(np.arange(0, np.max(zero_mask.data), 10))

plt.title("Frequency of Overshooting Events (2012-2014)", fontsize=16)
outpng = f"/home/karengarcia/MSc_project_backup/Frequency_map_2012_2014_8mm.png"
plt.savefig(outpng, dpi=300, bbox_inches='tight')
plt.close(fig)

print("Overshooting frequency map complete")

# file_name_overshooting =[]
# for i in range(len(pctisccp_above_tp.time.values)):
#     proj = ccrs.PlateCarree(central_longitude=0)
#     fig, ax1 = plt.subplots(1, 1, figsize=(24, 8), subplot_kw={'projection': proj})
#         # Tropopause pressure contourf
#     cf1 = ax1.contourf(lon_CAN, lat_CAN, pctisccp_above_tp.isel(time=i)/100, levels=np.arange(50,405,5), cmap = "Grays")
#     cbar1 = fig.colorbar(cf1, ax=ax1, label="Cloud top pressure above tropopause (hPa)")
#     ax1.coastlines(color='black', linewidth=0.5, linestyle='--')
#     ax1.add_feature(cfeature.LAND, facecolor='olivedrab')
#     ax1.add_feature(cfeature.OCEAN)
#     ax1.add_feature(cfeature.LAKES)
#     ax1.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
#     ax1.set_global()

#     Cantime = cloud_pressure.time.values[i]
#     ax1.set_title(f"CanESM5 {(str(Cantime))[:19]} Overshooting Map", fontsize =15)

#     plt.tight_layout()
#     outpng = f"/home/karengarcia/MSc_project_backup/Animations/Overshooting/Overshooting_{(str(Cantime))[:19]}.png"
#     plt.savefig(outpng, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     file_name_overshooting.append(outpng) 
    
# print("Overshooting map animation complete")