import numpy as np
import xarray as xr
import glob
import os

g = 9.81
R_dry_air = 287.05
target_lapserate = 2.0  # K/km
top_boundary = 50 * 100      # 50 hPa 
bottom_boundary = 400 * 100 # 400 hPa 


output_dir = "downloads-karengarcia/ESGF_downloads/6hourly"
ta_files = sorted(glob.glob(os.path.join(output_dir, "ta_6hrLev_CanESM5_historical_r1i1p2f1_gn_2014*.nc")))
hus_files = sorted(glob.glob(os.path.join(output_dir, "hus_6hrLev_CanESM5_historical_r1i1p2f1_gn_2014*.nc")))

ta = xr.open_mfdataset(ta_files, chunks={'time': 20})
hus = xr.open_mfdataset(hus_files, chunks={'time': 20})

ta["plev"] = ta["ap"] + ta["b"] * ta["ps"]
hus["plev"] = hus["ap"] + hus["b"] * hus["ps"]

## UTLS mask ##
# In the tropics (between 30N and 30S) the UTLS is defined between 200hPa and 50hPa
# Elsewhere the UTLS is defined between 400hPa and 50hPa
# bottom_floor = xr.where(abs(ta.lat) < 30, 200 * 100, 400 * 100)
plev_mask = ((ta["plev"] >= top_boundary) & 
             (ta["plev"] <= bottom_boundary)).compute()

UTLS_ta = ta.where(plev_mask, drop=False)
UTLS_q  = hus.where(plev_mask, drop=False)

print("Selected the UTLS")
print("You can continue now")

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

strat_mask = hus['plev'] <= tropopause_pressure
q_strat = hus.hus.where(strat_mask)
dP = np.abs(hus['plev'].diff("lev"))
q_aligned = q_strat.isel(lev=slice(0, -1))

# integral of  (q * dP) / g (Units: kg/m^2)
W_strat = (q_aligned * dP).sum("lev", skipna=True) / g
 

W_strat = W_strat.to_dataset(name="SWC")
W_strat = W_strat.assign_coords({"lat": ta.lat,
                                 "lon": ta.lon})

for v in W_strat:
    W_strat[v].encoding = {}

print("Stratospheric water column (kg/m^2) calculated.")
W_strat.to_netcdf("stratospheric_water_column_400.nc")