import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

R_ideal = 8.314               # Pa m^3 / (mol K)
m_dryair = 28.97 / 1000       # kg / mol
ref_press = 1013.25 * 100     # Pa
R_dry_air = 287.05            # J / (kg K)
chunk_div = 50
target_lapserate = 2
precip = 8
g = 9.81                      #m/s^2

####### UTLS Boundaries #######
top_boundary = 50
bottom_boundary = 400
max_height_cap = 70
###################################

chunks = {"valid_time": 31,
         "pressure_level": -1,
          "latitude": 360,
          "longitude": 360}


year = 2014
input_dir = "/home/karengarcia/downloads-karengarcia/ERA5/Hourly/"
temp_paths = sorted(glob.glob(os.path.join(input_dir, "temperature/2014/ERA5_temperature_*1200*_Hourly_output.nc"))) 
temp_ERA = xr.open_mfdataset(temp_paths, chunks=chunks) 

q_paths = sorted(glob.glob(os.path.join(input_dir,"specific_humidity/ERA5_specific_humidity_*1200*_Hourly_output.nc"))) 
q_ERA = xr.open_mfdataset(q_paths, chunks=chunks) 


#### Tropopause ####
# UTLS mask
plev_mask = ((temp_ERA["pressure_level"] >= top_boundary) &
            (temp_ERA["pressure_level"] <= bottom_boundary)).compute()

UTLS_ta = temp_ERA.where(plev_mask, drop=True)

dT = UTLS_ta.t.diff("pressure_level")
dP = UTLS_ta["pressure_level"].diff("pressure_level")

T_low  = UTLS_ta.t.isel(pressure_level=slice(0, -1))
T_high = UTLS_ta.t.isel(pressure_level=slice(1, None)).assign_coords(pressure_level=T_low.pressure_level)
T_mid  = (T_low + T_high) / 2

P_low  = UTLS_ta["pressure_level"].isel(pressure_level=slice(0, -1))
P_high = UTLS_ta["pressure_level"].isel(pressure_level=slice(1, None)).assign_coords(pressure_level=P_low.pressure_level)
P_mid  = (P_low + P_high) / 2

# Hydrostatic dz
dz = -(R_dry_air * T_mid) / (g * P_mid*100) * dP*100
lapse_rate = - (dT / dz) * 1000

# LAPSE RATE TROPOPAUSE (LRT)
tropopause_mask     = lapse_rate <= target_lapserate
lrt_index = tropopause_mask.argmax(dim="pressure_level").compute()
lrt_pressure = UTLS_ta["pressure_level"].isel(pressure_level=lrt_index)

# COLD POINT TROPOPAUSE (CPT)
cpt_index = UTLS_ta.t.argmin(dim="pressure_level").compute()
cpt_pressure = UTLS_ta["pressure_level"].isel(pressure_level=cpt_index)

index_diff = np.abs(cpt_index - lrt_index)
# Condition: If CPT is significantly different from LRT and meets height cap
use_cpt_condition = (index_diff >= 3) & (cpt_pressure >= max_height_cap)

# Final Tropopause Pressure and Index selection
true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
final_tp_index = xr.where(use_cpt_condition, cpt_index, lrt_index)
print("Tropopause calculation successful")

p_vals = temp_ERA['pressure_level'].values

edges = np.concatenate([
    [p_vals[0] - (p_vals[1] - p_vals[0])/2],
    (p_vals[:-1] + p_vals[1:]) / 2,
    [p_vals[-1] + (p_vals[-1] - p_vals[-2])/2]])
 
#Difference all pressure values

dP_all = xr.DataArray(np.abs(np.diff(edges)), coords=[temp_ERA.pressure_levels], dims=['pressure_level'])

above_tp = temp_ERA.lev <= true_tropopause_p


#Taking the values of SHUM above the tropopause, everything else should be zero
shum_above_trop = q_ERA['SHUM'].where(above_tp, 0,0)
dP = np.abs(shum_above_trop['lev'].diff(dim='lev'))

# #Realigning becasue dP array is shorter because of the differencing
shum_above_trop = shum_above_trop.isel(lev=slice(0,-1))

# #Integral of (q*dP)/g (Units: kg*m^2)
W_strat = (shum_above_trop*dP_all).sum(dim='lev') / g
E_radius = 6371000.0

# 2. Calculate the grid spacing in radians
# Assumes regular spacing. d_lon and d_lat will be scalars.
d_lon = np.deg2rad(temp_ERA.longitude.diff("longitude").mean())
d_lat = np.deg2rad(temp_ERA.latitude.diff("latitude").mean())

#Calculate the area of each cell
cell_area = (E_radius**2) * np.cos(np.deg2rad(temp_ERA.lat)) * d_lat * d_lon
total_mass = (W_strat * cell_area).sum(dim=["latitude", "longitude"])
total_mass_val = (total_mass.compute()).rolling(valid_time=1).mean()

plt.figure(figsize=(16, 6))
plt.plot(total_mass_val['time'], total_mass_val.values/1e9)
plt.xlabel("Time (YYYY-MM)")
plt.ylabel("Mass (Tg)")
plt.title("ERA5 Total Integrated Stratospheric Water Vapour (Global)")
final_plot_path = f"/home/karengarcia/MSc_project/Figures/ERA_Global_Water_Budget.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {final_plot_path}")