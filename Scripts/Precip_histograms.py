import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#####################################
# ERA5
#####################################
ds = xr.open_dataset("/home/karengarcia/downloads-karengarcia/ERA5/Hourly/convective_precipitation/2014/Regridded/ERA5_convective_precipitation_2014_6hourly_CanAM5.nc")
# print(ds)
data_era = ds['cp'].values*4000 #m/6hours to mm/day
# print(data_era)
data_era = data_era[np.isfinite(data_era)]
data_era = data_era[data_era > 0]

bins_era = np.logspace(np.log10(data_era.min()), np.log10(data_era.max()), 50)
p10_era = np.percentile(data_era, 10)

plt.figure(figsize=(16,6))
plt.hist(data_era, bins=bins_era, color="skyblue", edgecolor='black', alpha=0.9)
plt.axvline(p10_era, color="red", linestyle="--",
            linewidth=2, label=f"10th percentile = {p10_era:.3e} mm/day")

plt.legend()
plt.xscale("log")
plt.yscale("log")
# plt.ylim([0, 3e6])
plt.xlim([1e-8, 1e2])
plt.xlabel("Convective Precipitation (mm/day)")
plt.ylabel("Occurrences per year (Count)")
plt.title("ERA5 Histogram of Convective Precipitation")

plt.savefig("/home/karengarcia/MSc_project/Figures/Criteria_testing/Histograms/ERA5_precip_histogram.png",
            dpi=300, bbox_inches='tight')
plt.close()
print("/home/karengarcia/MSc_project/Figures/Criteria_testing/Histograms/ERA5_precip_histogram.png")


# #####################################
# # MERRA-2
# #####################################
ds2 = xr.open_dataset("/home/karengarcia/criteria_testing/MERRA_precip.nc")
# print(ds2)
data_merra = ds2["__xarray_dataarray_variable__"].values*86400
data_merra = data_merra[np.isfinite(data_merra)]
data_merra = data_merra[data_merra > 0]

bins_merra = np.logspace(np.log10(data_merra.min()), np.log10(data_merra.max()), 50)
p10_merra = np.percentile(data_merra, 10)

plt.figure(figsize=(16,6))
plt.hist(data_merra, bins=bins_merra, color="skyblue", edgecolor='black', alpha=0.9)
plt.axvline(p10_merra, color="red", linestyle="--",
            linewidth=2, label=f"10th percentile = {p10_merra:.3e} mm/day")

plt.legend()
plt.xscale("log")
plt.yscale("log")
# plt.ylim([0, 3e6])

plt.xlabel("Convective Precipitation (mm/day)")
plt.ylabel("Occurrences per year (Count)")
plt.title("MERRA-2 Histogram of Convective Precipitation")

plt.savefig("/home/karengarcia/MSc_project/Figures/Criteria_testing/Histograms/MERRA_precip_histogram.png",
            dpi=300, bbox_inches='tight')
plt.close()
print("/home/karengarcia/MSc_project/Figures/Criteria_testing/Histograms/MERRA_precip_histogram.png")

