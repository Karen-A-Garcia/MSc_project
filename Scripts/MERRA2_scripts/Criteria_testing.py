import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
year = 2014
chunks = {"time": 250}
g = 9.81                      # m/s^2
R_dry_air = 287.05            # J/(kg K)
target_lapserate = 2.0        # K/km
upper_bound = 50 * 100        # 50 hPa -> Pa
lower_bound = 400 * 100       # 400 hPa -> Pa
max_height_cap = 70 * 100     # 70 hPa -> Pa

input_dir = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly"
mst_Np = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_mst_Np.2014_6hourly_CanAM5_grid.nc"
mst_Ne = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_mst_Ne.2014_6hourly_CanAM5_grid.nc"
asm_Nv = "/home/karengarcia/downloads-karengarcia/MERRA-2/Regridded_6hourly/MERRA2_400.tavg3_3d_asm_Nv.2014_6hourly_CanAM5_grid.nc"

temp = ((xr.open_dataset(asm_Nv, chunks=chunks))['T']).sortby("lev", ascending=False)
press= ((xr.open_dataset(asm_Nv, chunks=chunks))['PL']).sortby("lev", ascending=False)
QI = ((xr.open_dataset(asm_Nv, chunks=chunks))['QI']).sortby("lev", ascending=False)
QL = ((xr.open_dataset(asm_Nv, chunks=chunks))['QL']).sortby("lev", ascending=False)
cmfmc_edges = ((xr.open_dataset(mst_Ne, chunks=chunks))['CMFMC']).sortby("lev", ascending=False)
mst_Np_PFLCU = ((xr.open_dataset(mst_Np, chunks=chunks))["PFLCU"]).sortby("lev", ascending=False)
mst_Np_PFICU = ((xr.open_dataset(mst_Np, chunks=chunks))["PFICU"]).sortby("lev", ascending=False)
precip = mst_Np_PFLCU.isel(lev=slice(0,5)).sum(dim='lev') + mst_Np_PFICU.isel(lev=slice(0,5)).sum(dim='lev')
mass_flux = (cmfmc_edges.isel(lev=slice(0, -1)) + 
             cmfmc_edges.isel(lev=slice(1, None)).values) / 2
mass_flux = mass_flux.assign_coords(lev=temp.lev)

#Masking the pressure between 400hPa and 50hPa so that I can restrict the tropopause location

mask = (press >= upper_bound) & (press <= lower_bound)
temp_masked = temp.where(mask)
press_masked = press.where(mask)

# #taking the change in temp and pressure along the lev dimension
dT = temp_masked.diff('lev')
dP = press_masked.diff('lev')

T_mid = (temp_masked.isel(lev=slice(0,-1)) + temp_masked.isel(lev=slice(1,None)).values) / 2
P_mid = (press_masked.isel(lev=slice(0,-1)) + press_masked.isel(lev=slice(1,None)).values) / 2

# # Hydrostatic dz
dz = -(R_dry_air * T_mid) / (g * P_mid) * dP
lapse_rate = - (dT / dz) * 1000

# # LAPSE RATE TROPOPAUSE (LRT)
tropopause_mask     = lapse_rate <= target_lapserate
lrt_index = tropopause_mask.argmax(dim="lev").compute()
lrt_pressure = press.isel(lev=lrt_index)

# # COLD POINT TROPOPAUSE (CPT)
cpt_index = temp_masked.T.argmin(dim="lev").compute()
cpt_pressure = press.isel(lev=cpt_index)

index_diff = np.abs(cpt_index - lrt_index)
# # Condition: If CPT is significantly different from LRT and meets height cap
use_cpt_condition = (index_diff >= 3)

# # Final Tropopause Pressure and Index selection
true_tropopause_p = xr.where(use_cpt_condition, cpt_pressure, lrt_pressure)
final_tp_index = xr.where(use_cpt_condition, cpt_index, lrt_index)
print("Tropopause calculation successful")

# #### Overshooting Calculation ####
cloud_total         = QI + QL
nlev                = temp.sizes["lev"]
#print("nlev:", nlev)
level_indices       = xr.DataArray(np.arange(nlev),
                            dims = ["lev"],
                            coords ={"lev": temp.lev})

above_tp = press <= true_tropopause_p
#print("Above tropopause:", above_tp.values)
above_tp = above_tp.broadcast_like(cloud_total)
ice_above_trop = (cloud_total.where(above_tp)).sum(dim="lev")
ice_above_trop = ice_above_trop.to_dataset(name='Ice_above_tp')
ice_path = f'/home/karengarcia/criteria_testing/MERRA_ice_over_tp.nc'
ice_above_trop.to_netcdf(ice_path)

#Saving mass flux over the tropopause 
dmcu_above_trop = (mass_flux.where(above_tp)).sum(dim="lev")
dmcu_above_trop = dmcu_above_trop.to_dataset(name='Mass_flux_above_tp')
mf_path = f'/home/karengarcia/criteria_testing/MERRA_flux_over_tp.nc'
dmcu_above_trop.to_netcdf(mf_path)

precip_path = f'/home/karengarcia/criteria_testing/MERRA_precip.nc'
precip.to_netcdf(precip_path)

# # print("Convective mass flux above trop:", dmcu_above_trop.values)
# #print("Ice above tropopause:", ice_above_trop.values)
# #print("Max:",ice_above_trop.max().values, "kg/kg")
# #print("Min:",ice_above_trop.min().values, "kg/kg")

# ################ Three criteria: ################
# # 1. Total cloud (sum of qi and ql) above the tropopause is bigger than zero
# # 2. Precipitation threshold must be met (4mm/day)
# # 3. Precipitation threshold must be met (8mm/day)
# # 4. Cumulative mass flux above the tropopause is bigger than zero
# # 5. Total cloud (sum of cic and clw) above the tropopause is bigger than 10^-5 kg/kg

# #Ice thresholds
# thresholds = np.array([1e-10,1e-09,1e-08,1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01])
# for ice in thresholds:
#     above_tp = (ice_above_trop >= ice)
#     above_tp = above_tp.astype('int8')
#     above_tp = above_tp.to_dataset(name='Ice_above_tp')
#     above_tp = above_tp.assign_coords({"lon": QI.lon,
#                                     "lat": QI.lat})
#     for v in above_tp.variables:
#         above_tp[v].encoding = {}
#     output_path = f'/home/karengarcia/criteria_testing/Ice_thresholds/MERRA_above_trop_{str(ice)}_{str(year)}.nc'
#     above_tp.to_netcdf(output_path)
#     print(f"File saved to", output_path)
    
#     CMF_above_tp = (dmcu_above_trop >= ice)
#     CMF_above_tp = CMF_above_tp.astype('int8')
#     CMF_above_tp = CMF_above_tp.to_dataset(name='Mass_flux_above_tp')
#     CMF_above_tp = CMF_above_tp.assign_coords({"lon": QI.lon,
#                                     "lat": QI.lat})
#     for v in CMF_above_tp.variables:
#         CMF_above_tp[v].encoding = {}
#     output_path = f'/home/karengarcia/criteria_testing/Mass_flux_thresholds/MERRA_above_trop_{str(ice)}_{str(year)}.nc'
#     CMF_above_tp.to_netcdf(output_path)
#     print(f"File saved to", output_path)


# # #option 1
# # ice_threshold = 0
# # ice_above_tp = ((ice_above_trop > 0))
# # ice_above_tp = ice_above_tp.astype('int8')
# # ice_above_tp = ice_above_tp.to_dataset(name='Option_1')
# # ice_above_tp = ice_above_tp.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})
# # for v in ice_above_tp.variables:
# #     ice_above_tp[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_1/MERRA_overshoot_option1_{str(year)}.nc'
# # ice_above_tp.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #option 2
# # precip1 = 4
# # prc4mm = precip*86400 >= precip1 #Convert (meter/hour to mm/day)
# # prc4mm = prc4mm.astype('int8')
# # prc4mm = prc4mm.to_dataset(name='Option_2')
# # prc4mm = prc4mm.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})

# # for v in prc4mm.variables:
# #     prc4mm[v].encoding = {}

# # output_path = f'/home/karengarcia/criteria_testing/Option_2/MERRA_overshoot_option2_{str(year)}.nc'
# # prc4mm.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #option 3
# # precip2 = 8
# # prc8mm = precip*86400 >= precip2 #Convert (meter/hour to mm/day)
# # prc8mm = prc8mm.astype('int8')
# # prc8mm = prc8mm.to_dataset(name='Option_3')
# # prc8mm = prc8mm.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})

# # for v in prc8mm.variables:
# #     prc8mm[v].encoding = {}

# # output_path = f'/home/karengarcia/criteria_testing/Option_3/MERRA_overshoot_option3_{str(year)}.nc'
# # prc8mm.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #option 12
# # above_tp_and_prc4 = (ice_above_trop > 0) & \
# #                 (precip*86400 >= precip1) #Convert (meter/hour to mm/day)

# # above_tp_and_prc4 = above_tp_and_prc4.astype('int8')
# # above_tp_and_prc4 = above_tp_and_prc4.to_dataset(name='Option_12')
# # above_tp_and_prc4 = above_tp_and_prc4.assign_coords({"lon": QI.lon,
# #                                                 "lat": QI.lat})

# # for v in above_tp_and_prc4.variables:
# #     above_tp_and_prc4[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_12/MERRA_overshoot_option12_{str(year)}.nc'
# # above_tp_and_prc4.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #option 13
# # above_tp_and_prc8 = (ice_above_trop > 0) & \
# #                 (precip*86400 >= precip2) #Convert (meter/hour to mm/day)

# # above_tp_and_prc8 = above_tp_and_prc8.astype('int8')
# # above_tp_and_prc8 = above_tp_and_prc8.to_dataset(name='Option_13')
# # above_tp_and_prc8 = above_tp_and_prc8.assign_coords({"lon": QI.lon,
# #                                                 "lat": QI.lat})

# # for v in above_tp_and_prc8.variables:
# #     above_tp_and_prc8[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_13/MERRA_overshoot_option13_{str(year)}.nc'
# # above_tp_and_prc8.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #option 5
# # above_tp = ((ice_above_trop >= 1e-5))
# # above_tp = above_tp.astype('int8')
# # above_tp = above_tp.to_dataset(name='Option_5')
# # above_tp = above_tp.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})
# # for v in above_tp.variables:
# #     above_tp[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_5/MERRA_overshoot_option5_{str(year)}.nc'
# # above_tp.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #option 52 
# # above_tp5_and_prc4 = (ice_above_trop >=1e-5) & \
# #                 (precip*86400 >= precip1) #Convert (meter/hour to mm/day)

# # above_tp5_and_prc4 = above_tp5_and_prc4.astype('int8')
# # above_tp5_and_prc4 = above_tp5_and_prc4.to_dataset(name='Option_52')
# # above_tp5_and_prc4 = above_tp5_and_prc4.assign_coords({"lon": QI.lon,
# #                                                 "lat": QI.lat})

# # for v in above_tp5_and_prc4.variables:
# #     above_tp5_and_prc4[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_52/MERRA_overshoot_option52_{str(year)}.nc'
# # above_tp5_and_prc4.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #option 53
# # above_tp5_and_prc8 = (ice_above_trop >= 1e-5) & \
# #                 (precip*86400 >= precip2) #Convert (meter/hour to mm/day)

# # above_tp5_and_prc8 = above_tp5_and_prc8.astype('int8')
# # above_tp5_and_prc8 = above_tp5_and_prc8.to_dataset(name='Option_53')
# # above_tp5_and_prc8 = above_tp5_and_prc8.assign_coords({"lon": QI.lon,
# #                                                 "lat": QI.lat})

# # for v in above_tp5_and_prc8.variables:
# #     above_tp5_and_prc8[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_53/MERRA_overshoot_option53_{str(year)}.nc'
# # above_tp5_and_prc8.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #Option 4
# # CMF_above_tp = (dmcu_above_trop > 0)
# # CMF_above_tp = CMF_above_tp.astype('int8')
# # CMF_above_tp = CMF_above_tp.to_dataset(name='Option_4')
# # CMF_above_tp = CMF_above_tp.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})
# # for v in CMF_above_tp.variables:
# #     CMF_above_tp[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_4/MERRA_overshoot_option4_{str(year)}.nc'
# # CMF_above_tp.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #Option 1 and 4
# # CMF_above_tp = (dmcu_above_trop > 0) & (ice_above_trop > 0)
# # CMF_above_tp = CMF_above_tp.astype('int8')
# # CMF_above_tp = CMF_above_tp.to_dataset(name='Option_14')
# # CMF_above_tp = CMF_above_tp.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})
# # for v in CMF_above_tp.variables:
# #     CMF_above_tp[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_14/MERRA_overshoot_option14_{str(year)}.nc'
# # CMF_above_tp.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #Option 2 and 4 
# # CMF_above_tp = (precip*86400 >= precip1) & (dmcu_above_trop > 0) 
# # CMF_above_tp = CMF_above_tp.astype('int8')
# # CMF_above_tp = CMF_above_tp.to_dataset(name='Option_24')
# # CMF_above_tp = CMF_above_tp.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})
# # for v in CMF_above_tp.variables:
# #     CMF_above_tp[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_24/MERRA_overshoot_option24_{str(year)}.nc'
# # CMF_above_tp.to_netcdf(output_path)
# # print(f"File saved to", output_path)

# # #Option 3 and 4 
# # CMF_above_tp = (precip*86400 >= precip2) & (dmcu_above_trop > 0)
# # CMF_above_tp = CMF_above_tp.astype('int8')
# # CMF_above_tp = CMF_above_tp.to_dataset(name='Option_34')
# # CMF_above_tp = CMF_above_tp.assign_coords({"lon": QI.lon,
# #                                 "lat": QI.lat})
# # for v in CMF_above_tp.variables:
# #     CMF_above_tp[v].encoding = {}
# # output_path = f'/home/karengarcia/criteria_testing/Option_34/MERRA_overshoot_option34_{str(year)}.nc'
# # CMF_above_tp.to_netcdf(output_path)
# # print(f"File saved to", output_path)