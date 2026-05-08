import xarray as xr 

ds = xr.open_mfdataset('/home/karengarcia/downloads-karengarcia/MERRA-2/tavg1_2d_flx_Nx/MERRA2_400.tavg1_2d_flx_Nx.201406*.nc4', data_vars=['PRECCON'])
ds.to_netcdf('/home/karengarcia/downloads-karengarcia/MERRA-2/MERRA2_400.tavg1_2d_flx_Nx_201406_PRECCON.nc') 
