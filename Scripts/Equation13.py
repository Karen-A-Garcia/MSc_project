import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
########################################################
######################### ERA5 ######################### 
########################################################
 
g = 9.81        # Gravity (m/s^2)
R = 6371000.0      # Earth's radius (meters)
p_c = 100    # Control surface of 100 hPa

chunks = {'valid_time': 120, 'lat': 64, 'lon': 128}
ERA_data_dir = "/home/karengarcia/downloads-karengarcia/ERA5/Daily/"
ERA_mdata_dir= "/home/karengarcia/downloads-karengarcia/ERA5/Monthly/"
#File Paths
ERA_ua_files  = sorted(glob.glob(os.path.join(ERA_data_dir, "u/ERA5_Daily_Above100hPa_u_2014_124x64.nc")))
ERA_va_files  = sorted(glob.glob(os.path.join(ERA_data_dir, "v/ERA5_Daily_Above100hPa_v_2014_124x64.nc")))
ERA_hus_files = sorted(glob.glob(os.path.join(ERA_data_dir, "q/ERA5_Daily_Above100hPa_q_2014_124x64.nc")))
ERA_wap_files = sorted(glob.glob(os.path.join(ERA_data_dir, "w/ERA5_Daily_Above100hPa_w_2014_124x64.nc")))
ERA_mua_files  = sorted(glob.glob(os.path.join(ERA_mdata_dir, "ERA5_monthly_u_1979_2019_128x64.nc")))
ERA_mva_files  = sorted(glob.glob(os.path.join(ERA_mdata_dir, "ERA5_monthly_v_1979_2019_128x64.nc")))
ERA_mhus_files = sorted(glob.glob(os.path.join(ERA_mdata_dir, "ERA5_monthly_q_1979_2019_128x64.nc")))
ERA_mwap_files = sorted(glob.glob(os.path.join(ERA_mdata_dir, "ERA5_monthly_w_1979_2019_128x64.nc")))

ERA_ua_ds  = xr.open_mfdataset(ERA_ua_files, chunks=chunks).sortby('pressure_level')
ERA_va_ds  = xr.open_mfdataset(ERA_va_files, chunks=chunks).sortby('pressure_level')
ERA_hus_ds = xr.open_mfdataset(ERA_hus_files, chunks=chunks).sortby('pressure_level')
ERA_wap_ds = xr.open_mfdataset(ERA_wap_files, chunks=chunks).sortby('pressure_level')
ERA_mua_ds  = xr.open_mfdataset(ERA_mua_files, chunks=chunks).sortby('pressure_level').isel(valid_time=slice(-12*6,(-12*5)+1))
ERA_mva_ds  = xr.open_mfdataset(ERA_mva_files, chunks=chunks).sortby('pressure_level').isel(valid_time=slice(-12*6,(-12*5)+1))
ERA_mhus_ds = xr.open_mfdataset(ERA_mhus_files, chunks=chunks).sortby('pressure_level').isel(valid_time=slice(-12*6,(-12*5)+1))
ERA_mwap_ds = xr.open_mfdataset(ERA_mwap_files, chunks=chunks).sortby('pressure_level').isel(valid_time=slice(-12*6,(-12*5)+1))

 
ERA_u     = ERA_ua_ds['u'].sel(pressure_level=slice(0, p_c))
ERA_v     = ERA_va_ds['v'].sel(pressure_level=slice(0, p_c))
ERA_q     = ERA_hus_ds['q'].sel(pressure_level=slice(0, p_c))
ERA_omega = ERA_wap_ds['w'].sel(pressure_level=slice(0, p_c))
ERA_mu    = ERA_mua_ds['u'].sel(pressure_level=slice(0, p_c))
ERA_mv    = ERA_mva_ds['v'].sel(pressure_level=slice(0, p_c))
ERA_mq    = ERA_mhus_ds['q'].sel(pressure_level=slice(0, p_c))
ERA_momega= ERA_mwap_ds['w'].sel(pressure_level=slice(0, p_c))

# # Values at 100hPa
ERA_q_pc     = ERA_hus_ds['q'].sel(pressure_level=p_c, method='nearest')
ERA_omega_pc = ERA_wap_ds['w'].sel(pressure_level=p_c, method='nearest')
ERA_mq_pc     = ERA_mhus_ds['q'].sel(pressure_level=p_c, method='nearest')
ERA_momega_pc = ERA_mwap_ds['w'].sel(pressure_level=p_c, method='nearest')

print("ERA Term 1: Full stratospheric moisture tendency")
ERA_time_seconds = (ERA_q.valid_time - ERA_q.valid_time[0]).dt.total_seconds()
ERA_dq_dt = ERA_q.assign_coords(valid_time=ERA_time_seconds).differentiate('valid_time')
ERA_dq_dt = ERA_dq_dt.assign_coords(valid_time=ERA_q.valid_time)

ERA_mtime_seconds = (ERA_mq.valid_time - ERA_mq.valid_time[0]).dt.total_seconds()
ERA_mdq_dt = ERA_mq.assign_coords(valid_time=ERA_mtime_seconds).differentiate('valid_time')
ERA_mdq_dt = ERA_mdq_dt.assign_coords(valid_time=ERA_mq.valid_time)

ERA_Q_s_spatial = (1 / g) * ERA_dq_dt.integrate('pressure_level')
ERA_mQ_s_spatial = (1 / g) * ERA_mdq_dt.integrate('pressure_level')
# Moisture transport divergence
print("ERA Term 2: Moisture Transport Divergence")
ERA_uq = ERA_q * ERA_u
ERA_vq = ERA_q * ERA_v
ERA_muq = ERA_mq * ERA_mu
ERA_mvq = ERA_mq * ERA_mv

ERA_cos_lat = np.cos(np.radians(ERA_q.lat))
ERA_mcos_lat = np.cos(np.radians(ERA_mq.lat))

ERA_duq_dlon = ERA_uq.differentiate('lon') * (180.0 / np.pi)
ERA_dvq_dlat = ERA_vq.differentiate('lat') * (180.0 / np.pi)
ERA_mduq_dlon = ERA_muq.differentiate('lon') * (180.0 / np.pi)
ERA_mdvq_dlat = ERA_mvq.differentiate('lat') * (180.0 / np.pi)
# Spherical divergence conversion

ERA_div_h = (1 / (R * ERA_cos_lat)) * ERA_duq_dlon + (1 / R) * ERA_dvq_dlat
ERA_mdiv_h = (1 / (R * ERA_mcos_lat)) * ERA_mduq_dlon + (1 / R) * ERA_mdvq_dlat
ERA_Div_h_spatial = - (1 / g) * ERA_div_h.integrate('pressure_level')
ERA_mDiv_h_spatial = - (1 / g) * ERA_mdiv_h.integrate('pressure_level')

#Vertical flux across 100hPa Isobar
print("ERA Term 3: Vertical Flux Across 100hPa")
ERA_Flux_v_spatial = - (ERA_q_pc * ERA_omega_pc) / g
ERA_mFlux_v_spatial = - (ERA_mq_pc * ERA_momega_pc) / g

#Spatial Averaging
ERA_weights = np.cos(np.radians(ERA_q.lat))
ERA_mweights = np.cos(np.radians(ERA_mq.lat))

ERA_weights.name = "weights"
ERA_mweights.name = "weights"

ERA_Q_s_ts    = ERA_Q_s_spatial.weighted(ERA_weights).mean(dim=('lat', 'lon')).compute()
ERA_mQ_s_ts    = ERA_mQ_s_spatial.weighted(ERA_mweights).mean(dim=('lat', 'lon')).compute()

ERA_Div_h_ts = ERA_Div_h_spatial.weighted(ERA_weights).mean(dim=('lat', 'lon')).compute()
ERA_mDiv_h_ts = ERA_mDiv_h_spatial.weighted(ERA_mweights).mean(dim=('lat', 'lon')).compute()

ERA_Flux_v_ts = ERA_Flux_v_spatial.weighted(ERA_weights).mean(dim=('lat', 'lon')).compute()
ERA_mFlux_v_ts = ERA_mFlux_v_spatial.weighted(ERA_mweights).mean(dim=('lat', 'lon')).compute()

#Residual -> Sphys_q + Sdiff_q
ERA_Residual_ts = ERA_Q_s_ts - ERA_Div_h_ts - ERA_Flux_v_ts
ERA_mResidual_ts = ERA_mQ_s_ts - ERA_mDiv_h_ts - ERA_mFlux_v_ts
ERA_plot_time = ERA_Q_s_ts.valid_time
ERA_mplot_time = ERA_mQ_s_ts.valid_time

# print("ERA plots")
plt.figure(figsize=(14, 7))
plt.plot(ERA_plot_time, ERA_Q_s_ts,  color='black', linestyle="--")
plt.plot(ERA_mplot_time, ERA_mQ_s_ts,  color='black', linestyle="-")
plt.title('ERA5 Full Stratospheric Moisture Tendency \n(Predefined Control Volume - 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'$\frac{\partial Q_s}{\partial t} (\frac{kg}{m^{2} \cdot s})$', fontsize=14)
plt.grid(True, linestyle=':')
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/ERA5_dQsdt_Above100hPa.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

plt.figure(figsize=(14, 7))
plt.plot(ERA_plot_time, ERA_Residual_ts,  color='black', linestyle="--")
plt.plot(ERA_mplot_time, ERA_mResidual_ts,  color='black', linestyle="-")
plt.ylim([-1.5e-10,1.5e-10])
plt.title('ERA5 Amip Stratospheric Moisture Tendency from Residual \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'$-\nabla_h \cdot \mathbf{F}_{q,h} (\frac{kg}{m^{2} \cdot s})$', fontsize=14)
plt.grid(True, linestyle=':')
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/ERA5_Qphys_Qdiff_Above100hPa.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

plt.figure(figsize=(14, 7))
plt.plot(ERA_plot_time, ERA_Div_h_ts, color='black', linestyle="--")
plt.plot(ERA_mplot_time, ERA_mDiv_h_ts, color='black', linestyle="-")
plt.ylim([-1e-12,1e-12])
plt.title('ERA5 Amip Moisture Transport Divergence \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'$S^{phys}_{q} + S^{diff}_{q}  (\frac{kg}{m^{2} \cdot s})$', fontsize=14)
plt.grid(True, linestyle=':')
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/ERA5_divFqh_Above100hPa.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

plt.figure(figsize=(14, 7))
plt.plot(ERA_plot_time, ERA_Flux_v_ts, color='black', linestyle="--")
plt.plot(ERA_plot_time, ERA_Flux_v_ts, color='black', linestyle="-")
plt.title('ERA5 Amip Flux of Moisture across 100hPa', fontsize=16, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'Vertical Boundary Flux ($\frac{q\omega}{g}$)', fontsize=14)
plt.grid(True, linestyle=':')
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/ERA5_Flux100hPa.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")

# Plotting the three components on one plot
plt.figure(figsize=(14, 7))
plt.plot(ERA_plot_time, ERA_Q_s_ts,
         label=r'Full stratospheric moisture tendency ($\frac{\partial Q_s}{\partial t}$)',
         color='black', linestyle = "--",linewidth=2)
plt.plot(ERA_mplot_time, ERA_mQ_s_ts,
         color='black', linestyle = "-")

plt.plot(ERA_plot_time, ERA_Div_h_ts,
         label=r'Moisture transport divergence ($-\nabla_h \cdot \mathbf{F}_{q,h}$)',
         color='crimson', alpha=0.8, linestyle='--')
plt.plot(ERA_mplot_time, ERA_mDiv_h_ts, 
         color='crimson', linestyle='-')

plt.plot(ERA_plot_time, ERA_Flux_v_ts,
         label=r'Vertical Boundary Flux ($\frac{q\omega}{g}$ at 100hPa)',
         color='royalblue', alpha=0.8, linestyle='--')
plt.plot(ERA_mplot_time, ERA_mFlux_v_ts,
        color='royalblue', linestyle='-')

plt.plot(ERA_plot_time, ERA_Residual_ts,
         label=r'Residual ($S^{phys}_{q} + S^{diff}_{q}$)',
         color='g', alpha=0.8, linestyle='--')
plt.plot(ERA_mplot_time, ERA_mResidual_ts,
         color='g', linestyle='-')

plt.title('ERA5 Stratospheric Water Vapor Budget Components \n (predefined control volume 100hPa)', fontsize=14, fontweight='bold')
plt.xlabel('Date (YYYY-MM)', fontsize=12)
plt.ylabel(r'Moisture Flux Component [ $\frac{kg}{m^{2} \cdot s}$]', fontsize=12)
plt.grid(True, linestyle=':')
plt.legend(fontsize=11, loc='upper right')
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/ERA5_All_components_fluxes.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}") 

# ########################################################
# ######################## CanAM5 ######################## 
# ########################################################


# # Settings and Constants
# g = 9.81           # Gravity (m/s^2)
# R = 6371000.0      # Earth's radius (meters)
# p_c = 100 * 100    # Control surface of 100 hPa (converted to Pa, i.e., 10000 Pa)

# chunks = {'time': 120, 'lat': 64, 'lon': 128}
# CAN_data_dir = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/"

# # File Paths
# CAN_ua_files  = sorted(glob.glob(os.path.join(CAN_data_dir, "ua/ua_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))
# CAN_va_files  = sorted(glob.glob(os.path.join(CAN_data_dir, "va/va_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))
# CAN_hus_files = sorted(glob.glob(os.path.join(CAN_data_dir, "hus/hus_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))
# CAN_wap_files = sorted(glob.glob(os.path.join(CAN_data_dir, "wap/wap_day_CanESM5_amip_r1i1p2f1_gn_2011*.nc")))

# CAN_ua_ds  = xr.open_mfdataset(CAN_ua_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))
# CAN_va_ds  = xr.open_mfdataset(CAN_va_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))
# CAN_hus_ds = xr.open_mfdataset(CAN_hus_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))
# CAN_wap_ds = xr.open_mfdataset(CAN_wap_files, chunks=chunks).sortby('plev').isel(time=slice(-365,None))

# CAN_ua    = CAN_ua_ds['ua'].sel(plev=slice(0, p_c))
# CAN_va    = CAN_va_ds['va'].sel(plev=slice(0, p_c))
# CAN_q     = CAN_hus_ds['hus'].sel(plev=slice(0, p_c))
# CAN_omega = CAN_wap_ds['wap'].sel(plev=slice(0, p_c))
# CAN_q_pc     = CAN_hus_ds['hus'].sel(plev=p_c, method='nearest')
# CAN_omega_pc = CAN_wap_ds['wap'].sel(plev=p_c, method='nearest')


# # Term 1: Full stratospheric moisture tendency (dQ/dt)
# print("CAN Term 1: Full stratospheric moisture tendency")
# CAN_time_seconds = (CAN_q.time - CAN_q.time[0]).dt.total_seconds()
# CAN_dq_dt = CAN_q.assign_coords(time=CAN_time_seconds).differentiate('time')
# CAN_dq_dt = CAN_dq_dt.assign_coords(time=CAN_q.time)

# CAN_Q_s_spatial = (1 / g) * CAN_dq_dt.integrate('plev')

# # Term 2: Horizontal Moisture Transport Divergence
# print("CAN Term 2: Moisture Transport Divergence")
# CAN_uq = CAN_q * CAN_ua
# CAN_vq = CAN_q * CAN_va

# CAN_cos_lat = np.cos(np.radians(CAN_q.lat))

# # Differentiate by degrees and convert to radians via (180 / pi)
# CAN_duq_dlon = CAN_uq.differentiate('lon') * (180.0 / np.pi)
# CAN_d_vq_coslat_dlat = (CAN_vq * CAN_cos_lat).differentiate('lat') * (180.0 / np.pi)

# # Spherical coordinate divergence: 
# CAN_div_h = (1 / (R * CAN_cos_lat)) * (CAN_duq_dlon + CAN_d_vq_coslat_dlat)
# CAN_Div_h_spatial = -(1 / g) * CAN_div_h.integrate('plev')


# # Term 3: Vertical Flux Across 100hPa Isobar
# print("CAN Term 3: Vertical Flux Across 100hPa")
# CAN_Flux_v_spatial = -(CAN_q_pc * CAN_omega_pc) / g


# # Spatial Averaging
# CAN_weights = np.cos(np.radians(CAN_q.lat))
# CAN_weights.name = "weights"

# CAN_Q_s_ts    = CAN_Q_s_spatial.weighted(CAN_weights).mean(dim=('lat', 'lon')).compute()
# CAN_Div_h_ts  = CAN_Div_h_spatial.weighted(CAN_weights).mean(dim=('lat', 'lon')).compute()
# CAN_Flux_v_ts = CAN_Flux_v_spatial.weighted(CAN_weights).mean(dim=('lat', 'lon')).compute()


# # Residual -> Sphys_q + Sdiff_q
# # Tendency + Divergence_Loss + Vertical_Loss = Residual Sources
# CAN_Residual_ts = CAN_Q_s_ts - CAN_Div_h_ts - CAN_Flux_v_ts
# CAN_plot_time = CAN_Q_s_ts.indexes['time'].to_datetimeindex()


# # Plotting and Figures
# print("CanAM5 plots")

# # Plot 1: Tendency
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Q_s_ts, color='black')
# plt.title('CanAM5 Amip Full Stratospheric Moisture Tendency \n(Predefined Control Volume - 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$\frac{\partial Q_s}{\partial t} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/dQsdt_Above100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 2: Horizontal Divergence
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Div_h_ts, color='black')
# plt.title('CanAM5 Amip Moisture Transport Divergence \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$\nabla_h \cdot \mathbf{F}_{q,h} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/divFqh_Above100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 3: Residual Parameterizations (FIXED file name matching title)
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Residual_ts, color='black')
# plt.title('CanAM5 Amip Stratospheric Moisture Tendency from Residual \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$S^{phys}_{q} + S^{diff}_{q} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/Qphys_Qdiff_Above100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 4: Vertical Boundary Flux
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Flux_v_ts, color='black')
# plt.title('CanAM5 Amip Flux of Moisture across 100hPa', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'Vertical Boundary Flux ($\frac{kg}{m^{2} \cdot s}$)', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/Flux100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 5: Combined Plot
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Q_s_ts,
#          label=r'Full stratospheric moisture tendency ($\frac{\partial Q_s}{\partial t}$)',
#          color='black', linewidth=2)
# plt.plot(CAN_plot_time, CAN_Div_h_ts,
#          label=r'Moisture transport divergence ($\nabla_h \cdot \mathbf{F}_{q,h}$)',
#          color='crimson', alpha=0.8, linestyle='--')
# plt.plot(CAN_plot_time, CAN_Flux_v_ts,
#          label=r'Vertical Boundary Flux Outward ($\frac{q\omega}{g}$ at 100hPa)',
#          color='royalblue', alpha=0.8, linestyle=':')
# plt.plot(CAN_plot_time, CAN_Residual_ts,
#          label=r'Residual ($S^{phys}_{q} + S^{diff}_{q}$)',
#          color='g', alpha=0.8, linestyle='-')
 
# plt.title('CanAM5 Stratospheric Water Vapor Budget Components\n(predefined control volume 100hPa)', fontsize=14, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'Moisture Flux Component [ $\frac{kg}{m^{2} \cdot s}$]', fontsize=12)
# plt.grid(True, linestyle=':')
# plt.legend(fontsize=11, loc='upper right')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/All_components_fluxes.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")


# # #########################################################
# # ##################### CanAM5 - ERA5 #####################
# # #########################################################


# # Plotting and Figures
# print("CanAM5 - ERA5 plots")

# # Plot 1: Tendency
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Q_s_ts.values-ERA_Q_s_ts.values, color='black')
# plt.title('CanAM5 - ERA5 Full Stratospheric Moisture Tendency \n(Predefined Control Volume - 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$\frac{\partial Q_s}{\partial t} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/diff_dQsdt_Above100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 2: Horizontal Divergence
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Div_h_ts.values-ERA_Div_h_ts.values, color='black')
# plt.title('CanAM5 - ERA5 Moisture Transport Divergence \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$\nabla_h \cdot \mathbf{F}_{q,h} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/diff_divFqh_Above100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 3: Residual Parameterizations (FIXED file name matching title)
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Residual_ts.values-ERA_Residual_ts.values, color='black')
# plt.title('CanAM5 - ERA5 Stratospheric Moisture Tendency from Residual \n (Predefined Control Volume 100hPa)', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'$S^{phys}_{q} + S^{diff}_{q} \ (\frac{kg}{m^{2} \cdot s})$', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/diff_Qphys_Qdiff_Above100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 4: Vertical Boundary Flux
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Flux_v_ts.values-ERA_Flux_v_ts.values, color='black')
# plt.title('CanAM5 - ERA5 Flux of Moisture across 100hPa', fontsize=16, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'Vertical Boundary Flux ($\frac{kg}{m^{2} \cdot s}$)', fontsize=13)
# plt.grid(True, linestyle=':')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/diff_Flux100hPa.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")

# # Plot 5: Combined Plot
# plt.figure(figsize=(14, 7))
# plt.plot(CAN_plot_time, CAN_Q_s_ts.values-ERA_Q_s_ts.values,
#          label=r'Full stratospheric moisture tendency ($\frac{\partial Q_s}{\partial t}$)',
#          color='black', linewidth=2)
# plt.plot(CAN_plot_time, CAN_Div_h_ts.values-ERA_Div_h_ts.values,
#          label=r'Moisture transport divergence ($\nabla_h \cdot \mathbf{F}_{q,h}$)',
#          color='crimson', alpha=0.8, linestyle='--')
# plt.plot(CAN_plot_time, CAN_Flux_v_ts.values-ERA_Flux_v_ts.values,
#          label=r'Vertical Boundary Flux Outward ($\frac{q\omega}{g}$ at 100hPa)',
#          color='royalblue', alpha=0.8, linestyle=':')
# plt.plot(CAN_plot_time, CAN_Residual_ts.values-ERA_Residual_ts.values,
#          label=r'Residual ($S^{phys}_{q} + S^{diff}_{q}$)',
#          color='g', alpha=0.8, linestyle='-')
 
# plt.title('CanAM5 - ERA5 Stratospheric Water Vapor Budget Components\n(predefined control volume 100hPa)', fontsize=14, fontweight='bold')
# plt.xlabel('Date (YYYY-MM)', fontsize=12)
# plt.ylabel(r'Moisture Flux Component [ $\frac{kg}{m^{2} \cdot s}$]', fontsize=12)
# plt.grid(True, linestyle=':')
# plt.legend(fontsize=11, loc='upper right')
# final_plot_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/diff_All_components_fluxes.png"
# plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"Figure saved to: {final_plot_path}")