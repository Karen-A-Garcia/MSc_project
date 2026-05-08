import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ua_file = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Monthly/ua_Amon_CanESM5_historical_r1i1p2f1_gn_185001-201412.nc"

ua = xr.open_dataset(ua_file)

# Step 1: zonal mean (average over longitude)
ua_zonal = ua["ua"].mean(dim="lon")

ua_eq = ua_zonal.sel(lat=slice(-15, 15)).mean(dim="lat") 

plt.figure(figsize=(12, 6))
# Create plot and store handle
cf = ua_eq.plot.contourf(
    x="time",
    y="plev",
    cmap="RdBu_r",
    levels=np.linspace(-10,10,20),
    extend="both"
)
plt.gca().invert_yaxis()

plt.title("QBO CanAM5")
plt.xlabel("Time")
plt.ylabel("Pressure (Pa)")
plt.tight_layout()
outpng = f"/home/karengarcia/MSc_project_backup/CanAM5_QBO.png"
plt.savefig(outpng, dpi=300, bbox_inches='tight')

