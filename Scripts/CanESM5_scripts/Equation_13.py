import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cftime
 
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

ua_ds  = xr.open_mfdataset(ua_files, chunks=chunks).sortby('plev').isel(time=slice(0,364))
va_ds  = xr.open_mfdataset(va_files, chunks=chunks).sortby('plev').isel(time=slice(0,364))
hus_ds = xr.open_mfdataset(hus_files, chunks=chunks).sortby('plev').isel(time=slice(0,364))
wap_ds = xr.open_mfdataset(wap_files, chunks=chunks).sortby('plev').isel(time=slice(0,364))

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
Residual_ts = Q_s_ts + Div_h_ts + Flux_v_ts
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
plt.title('CanAM5 Amip Stratospheric Moisture Tendency from Physical Parameterizations \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
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
         label=r'Physical parameterizations ($S^{phys}_{q} + S^{diff}_{q}$)',
         color='g', alpha=0.8, linestyle='-')
 
plt.title('Stratospheric Water Vapor Budget Components (predefined control volume 100hPa)', fontsize=14, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'Moisture Flux Component [ $\frac{kg}{m^{2} \cdot s}$]', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(fontsize=11, loc='upper right')
final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/All_components_fluxes.png"
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")