import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cftime

#This script includes CanAM5, ERA5, and MLS analysis


################ One year analysis of the water budget #############
# Settings and Constants
g = 9.81           # Gravity (m/s^2)
R = 6371000.0      # Earth's radius (meters)
p_c = 100 * 100    # Control surface of 100 hPa (converted to Pa, i.e., 10000 Pa)

chunks = {'time': 120, 'lat': 64, 'lon': 128}
data_dir = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/"

# File Paths
ua_files  = sorted(glob.glob(os.path.join(data_dir, "ua/ua_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))
va_files  = sorted(glob.glob(os.path.join(data_dir, "va/va_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))
hus_files = sorted(glob.glob(os.path.join(data_dir, "hus/hus_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))
wap_files = sorted(glob.glob(os.path.join(data_dir, "wap/wap_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))

ua_ds  = xr.open_mfdataset(ua_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))
va_ds  = xr.open_mfdataset(va_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))
hus_ds = xr.open_mfdataset(hus_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))
wap_ds = xr.open_mfdataset(wap_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))

ua    = ua_ds['ua'].sel(plev=slice(0, p_c))
va    = va_ds['va'].sel(plev=slice(0, p_c))
q     = hus_ds['hus'].sel(plev=slice(0, p_c))
omega = wap_ds['wap'].sel(plev=slice(0, p_c))
q_pc     = hus_ds['hus'].sel(plev=p_c, method='nearest')
omega_pc = wap_ds['wap'].sel(plev=p_c, method='nearest')


# Term 1: Full stratospheric moisture tendency (dQ/dt)
print("Term 1: Full stratospheric moisture tendency")
time_seconds = (q.time - q.time[0]).dt.total_seconds()
dq_dt = q.assign_coords(time=time_seconds).differentiate('time')
dq_dt = dq_dt.assign_coords(time=q.time)

Q_s_spatial = (1 / g) * dq_dt.integrate('plev')

# Term 2: Horizontal Moisture Transport Divergence
print("Term 2: Moisture Transport Divergence")
uq = q * ua
vq = q * va

cos_lat = np.cos(np.radians(q.lat))

# Differentiate by degrees and convert to radians via (180 / pi)
duq_dlon = uq.differentiate('lon') * (180.0 / np.pi)
# FIXED: Multiplied vq by cos_lat BEFORE differentiating by latitude
d_vq_coslat_dlat = (vq * cos_lat).differentiate('lat') * (180.0 / np.pi)

# Spherical coordinate divergence: 
div_h = (1 / (R * cos_lat)) * (duq_dlon + d_vq_coslat_dlat)


Div_h_spatial = -(1 / g) * div_h.integrate('plev')


# Term 3: Vertical Flux Across 100hPa Isobar
print("Term 3: Vertical Flux Across 100hPa")
Flux_v_spatial = -(q_pc * omega_pc) / g


# Spatial Averaging
weights = np.cos(np.radians(q.lat))
weights.name = "weights"

Q_s_ts    = Q_s_spatial.weighted(weights).mean(dim=('lat', 'lon')).compute()
Div_h_ts  = Div_h_spatial.weighted(weights).mean(dim=('lat', 'lon')).compute()
Flux_v_ts = Flux_v_spatial.weighted(weights).mean(dim=('lat', 'lon')).compute()


# Residual -> Sphys_q + Sdiff_q
# Tendency + Divergence_Loss + Vertical_Loss = Residual Sources
Residual_ts = Q_s_ts - Div_h_ts - Flux_v_ts
plot_time = Q_s_ts.indexes['time'].to_datetimeindex()


# Plotting and Figures
print("All plots")

# Plot 1: Tendency
plt.figure(figsize=(14, 7))
plt.plot(plot_time, Q_s_ts, color='black')
plt.title('CanAM5 Amip Full Stratospheric Moisture Tendency \n(Predefined Control Volume - 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'$\frac{\partial Q_s}{\partial t} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.5)
final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/dQsdt_Above100hPa.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

# Plot 2: Horizontal Divergence
plt.figure(figsize=(14, 7))
plt.plot(plot_time, Div_h_ts, color='black')
plt.title('CanAM5 Amip Moisture Transport Divergence \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'$\nabla_h \cdot \mathbf{F}_{q,h} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.5)
final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/divFqh_Above100hPa.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

# Plot 3: Residual Parameterizations (FIXED file name matching title)
plt.figure(figsize=(14, 7))
plt.plot(plot_time, Residual_ts, color='black')
plt.title('CanAM5 Amip Stratospheric Moisture Tendency from Residual \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'$S^{phys}_{q} + S^{diff}_{q} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.5)
final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/Qphys_Qdiff_Above100hPa.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

# Plot 4: Vertical Boundary Flux
plt.figure(figsize=(14, 7))
plt.plot(plot_time, -Flux_v_ts, color='black')
plt.title('CanAM5 Amip Flux of Moisture across 100hPa', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'Vertical Boundary Flux ($\frac{kg}{m^{2} \cdot s}$)', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.5)
final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/Flux100hPa.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

# Plot 5: Combined Plot
plt.figure(figsize=(14, 7))
plt.plot(plot_time, Q_s_ts,
         label=r'Full stratospheric moisture tendency ($\frac{\partial Q_s}{\partial t}$)',
         color='black', linewidth=2)
plt.plot(plot_time, Div_h_ts,
         label=r'Moisture transport divergence ($\nabla_h \cdot \mathbf{F}_{q,h}$)',
         color='crimson', alpha=0.8, linestyle='--')
plt.plot(plot_time, Flux_v_ts,
         label=r'Vertical Boundary Flux Outward ($\frac{q\omega}{g}$ at 100hPa)',
         color='royalblue', alpha=0.8, linestyle=':')
plt.plot(plot_time, Residual_ts,
         label=r'Residual ($S^{phys}_{q} + S^{diff}_{q}$)',
         color='g', alpha=0.8, linestyle='-')
 
plt.title('CanAM5 Stratospheric Water Vapor Budget Components\n(predefined control volume 100hPa)', fontsize=14, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'Moisture Flux Component [ $\frac{kg}{m^{2} \cdot s}$]', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')
final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/All_components_fluxes.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

###############
##### MLS #####
###############


g = 9.81                 #Gravity (m/s^2)
E_radius = 6371000.0     #Radius of the Earth
p_c = 100                #Control Pressure = 100hPa
mmr_wtda = 18.015/28.96  #Molar mass ratio of water to dry air

data_dir = "/home/karengarcia/downloads-karengarcia/MLS_data/v06/"
file_path = sorted(glob.glob(os.path.join(data_dir, "MLS-Aura_L3MB-H2O_v05*20*.nc")))

ds = xr.open_mfdataset(file_path, 
                       group="H2O PressureGrid", 
                       combine="by_coords").chunk(dict(time=-1))
MLS_H20 = ds['value']

MLS_H20 = MLS_H20.fillna(0)

MLS_H20["q"] = (MLS_H20 * mmr_wtda) / (1 + (MLS_H20 * mmr_wtda))
mq = MLS_H20['q']

mtime_seconds = (mq.time - mq.time[0]).dt.total_seconds()
mdq_dt = mq.assign_coords(time=mtime_seconds).differentiate('time')
mdq_dt = mdq_dt.assign_coords(time=mq.time)

mdq_dt_strat = mdq_dt.sel(lev=slice(p_c, None))

mQ_s_spatial = (1 / g) * mdq_dt_strat.integrate('lev')*100

mweights = np.cos(np.radians(mq.lat))
mweights.name = "weights"

mQ_s_ts = mQ_s_spatial.weighted(mweights).mean(dim=('lat', 'lon'), skipna=True).compute()
MLS_mplot_time = mQ_s_ts.indexes['time']

plt.figure(figsize=(14, 7))
plt.plot(MLS_mplot_time, mQ_s_ts, color='black', linestyle="-")
plt.title('MLS Full Stratospheric Moisture Tendency \n(Predefined Control Volume - 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylim([-1.5e-10,1.5e-10])
plt.ylabel(r'$\frac{\partial Q_s}{\partial t} (\frac{kg}{m^{2} \cdot s})$', fontsize=14)
plt.grid(True, linestyle=':')
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/MLS/MLS_dQsdt_Above100hPa.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")