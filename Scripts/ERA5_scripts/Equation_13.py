import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

 
#Settings and Constants
g = 9.81        # Gravity (m/s^2)
R = 6371000.0      # Earth's radius (meters)
p_c = 100    # Control surface of 100 hPa

chunks = {'valid_time': 120, 'lat': 64, 'lon': 128}
data_dir = "/home/karengarcia/downloads-karengarcia/ERA5/Daily/"

#File Paths
ua_files  = sorted(glob.glob(os.path.join(data_dir, "u/ERA5_Daily_Above100hPa_u_2014_124x64.nc")))
# va_files  = sorted(glob.glob(os.path.join(data_dir, "v/ERA5_Daily_Above100hPa_v_2014_124x64.nc")))
hus_files = sorted(glob.glob(os.path.join(data_dir, "q/ERA5_Daily_Above100hPa_q_2014_124x64.nc")))
wap_files = sorted(glob.glob(os.path.join(data_dir, "w/ERA5_Daily_Above100hPa_w_2014_124x64.nc")))
ua_ds  = xr.open_mfdataset(ua_files, chunks=chunks).sortby('pressure_level')
# va_ds  = xr.open_mfdataset(va_files, chunks=chunks).sortby('pressure_level')
hus_ds = xr.open_mfdataset(hus_files, chunks=chunks).sortby('pressure_level')
wap_ds = xr.open_mfdataset(wap_files, chunks=chunks).sortby('pressure_level')

 
u    = ua_ds['u'].sel(pressure_level=slice(0, p_c))
# v    = va_ds['v'].sel(pressure_level=slice(0, p_c))
q     = hus_ds['q'].sel(pressure_level=slice(0, p_c))
omega = wap_ds['w'].sel(pressure_level=slice(0, p_c))
# print(omega)
# #Values at 100hPa
q_pc     = hus_ds['q'].sel(pressure_level=p_c, method='nearest')
omega_pc = wap_ds['w'].sel(pressure_level=p_c, method='nearest')

print("Term 1: Full stratospheric moisture tendency")
time_seconds = (q.valid_time - q.valid_time[0]).dt.total_seconds()
dq_dt = q.assign_coords(valid_time=time_seconds).differentiate('valid_time')
dq_dt = dq_dt.assign_coords(valid_time=q.valid_time)

Q_s_spatial = (1 / g) * dq_dt.integrate('pressure_level')
# Moisture transport divergence
print("Term 2: Moisture Transport Divergence")
uq = q * u
# vq = q * v

cos_lat = np.cos(np.radians(q.lat))
duq_dlon = uq.differentiate('lon') * (180.0 / np.pi)
# dvq_dlat = vq.differentiate('lat') * (180.0 / np.pi)
# Spherical divergence conversion

# div_h = (1 / (R * cos_lat)) * duq_dlon + (1 / R) * dvq_dlat
# Div_h_spatial = - (1 / g) * div_h.integrate('pressure_level')

#Vertical flux across 100hPa Isobar
print("Term 3: Vertical Flux Across 100hPa")
Flux_v_spatial = - (q_pc * omega_pc) / g

#Spatial Averaging
weights = np.cos(np.radians(q.lat))
weights.name = "weights"
Q_s_ts    = Q_s_spatial.weighted(weights).mean(dim=('lat', 'lon')).compute()
# Div_h_ts = Div_h_spatial.weighted(weights).mean(dim=('lat', 'lon')).compute()
Flux_v_ts = Flux_v_spatial.weighted(weights).mean(dim=('lat', 'lon')).compute()

#Residual -> Sphys_q + Sdiff_q
# Residual_ts = Q_s_ts - Div_h_ts - Flux_v_ts
plot_time = Q_s_ts.valid_time

print("All plots")
plt.figure(figsize=(14, 7))
plt.plot(plot_time, Q_s_ts, color='black')
plt.title('ERA5 Full Stratospheric Moisture Tendency \n(Predefined Control Volume - 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'$\frac{\partial Q_s}{\partial t} (\frac{kg}{m^{2} \cdot s})$', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.5)
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/ERA5_dQsdt_Above100hPa.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

# plt.figure(figsize=(14, 7))
# plt.plot(plot_time, Residual_ts, color='black')
# plt.title('CanAM5 Amip Moisture Transport Divergence \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
# plt.title('CanAM5 Amip Stratospheric Moisture Tendency from Physical Parameterizations \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$-\nabla_h \cdot \mathbf{F}_{q,h} (\frac{kg}{m^{2} \cdot s})$', fontsize=14)
# plt.grid(True, linestyle=':', alpha=0.5)
# final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/Qphys_Qdiff_Above100hPa.png")
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# plt.figure(figsize=(14, 7))
# plt.plot(plot_time, Div_h_ts, color='black')
# plt.title('CanAM5 Amip Moisture Transport Divergence \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$S^{phys}_{q} + S^{diff}_{q}  (\frac{kg}{m^{2} \cdot s})$', fontsize=14)
# plt.grid(True, linestyle=':', alpha=0.5)
# final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/divFqh_Above100hPa.png")
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# plt.figure(figsize=(14, 7))
# plt.plot(plot_time, Flux_v_ts, color='black')
# plt.title('CanAM5 Amip Flux of Moisture across 100hPa', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'Vertical Boundary Flux ($\frac{q\omega}{g}$)', fontsize=14)
# plt.grid(True, linestyle=':', alpha=0.5)
# final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/Flux100hPa.png")
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plotting the three components on one plot
# plt.figure(figsize=(14, 7))
# plt.plot(plot_time, Q_s_ts,
#          label=r'Full stratospheric moisture tendency ($\frac{\partial Q_s}{\partial t}$)',
#          color='black', linewidth=2)

# plt.plot(plot_time, Div_h_ts,
#          label=r'Moisture transport divergence ($-\nabla_h \cdot \mathbf{F}_{q,h}$)',
#          color='crimson', alpha=0.8, linestyle='--')

# plt.plot(plot_time, Flux_v_ts,
#          label=r'Vertical Boundary Flux ($\frac{q\omega}{g}$ at 100hPa)',
#          color='royalblue', alpha=0.8, linestyle=':')

# plt.plot(plot_time, Residual_ts,
#          label=r'Physical parameterizations ($S^{phys}_{q} + S^{diff}_{q}$)',
#          color='g', alpha=0.8, linestyle='-')

# plt.title('Stratospheric Water Vapor Budget Components (predefined control volume 100hPa)', fontsize=14, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'Moisture Flux Component [ $\frac{kg}{m^{2} \cdot s}$]', fontsize=12)
# plt.grid(True, linestyle=':', alpha=0.5)
# plt.legend(fontsize=11, loc='upper right')
# final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/All_components_fluxes.png")
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")