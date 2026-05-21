import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#####################################
# ERA5
#####################################
ds = xr.open_dataset("/home/karengarcia/criteria_testing/ERA5_ice_over_tp.nc")
data_era = ds["Ice_above_tp"].values
data_era = data_era[np.isfinite(data_era)]
data_era = data_era[data_era > 0]

bins_era = np.logspace(np.log10(data_era.min()), np.log10(data_era.max()), 100)
p10_era = np.percentile(data_era, 10)

plt.figure(figsize=(16,6))
plt.hist(data_era, bins=bins_era, color="skyblue", edgecolor='black', alpha=0.9)
plt.axvline(p10_era, color="red", linestyle="--",
            linewidth=2, label=f"10th percentile = {p10_era:.3e} kg/kg")

plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim([0, 1e6])

plt.xlabel("Cloud Ice Mass + Cloud Liquid Mass (kg/kg)")
plt.ylabel("Occurrences per year (Count)")
plt.title("ERA5 Histogram of Combined Cloud Ice and Cloud Liquid Mass Over the Tropopause")

plt.savefig("/home/karengarcia/criteria_testing/ERA5_ice_histogram.png",
            dpi=300, bbox_inches='tight')


#####################################
# MERRA-2
#####################################
ds = xr.open_dataset("/home/karengarcia/criteria_testing/MERRA_ice_over_tp.nc")
data_merra = ds["Ice_above_tp"].values
data_merra = data_merra[np.isfinite(data_merra)]
data_merra = data_merra[data_merra > 0]

bins_merra = np.logspace(np.log10(data_merra.min()), np.log10(data_merra.max()), 100)
p10_merra = np.percentile(data_merra, 10)

plt.figure(figsize=(16,6))
plt.hist(data_merra, bins=bins_merra, color="skyblue", edgecolor='black', alpha=0.9)
plt.axvline(p10_merra, color="red", linestyle="--",
            linewidth=2, label=f"10th percentile = {p10_merra:.3e} kg/kg")

plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim([0, 1e6])

plt.xlabel("Cloud Ice Mass + Cloud Liquid Mass (kg/kg)")
plt.ylabel("Occurrences per year (Count)")
plt.title("MERRA-2 Histogram of Combined Cloud Ice and Cloud Liquid Mass Over the Tropopause")

plt.savefig("/home/karengarcia/criteria_testing/MERRA_ice_histogram.png",
            dpi=300, bbox_inches='tight')


#####################################
# OVERLAY (key fix)
#####################################
combined = np.concatenate([data_era, data_merra])
overlay_bins = np.logspace(np.log10(combined.min()), np.log10(combined.max()), 100)

plt.figure(figsize=(16,6))

plt.hist(data_era, bins=overlay_bins, alpha=0.5, label="ERA5")
plt.hist(data_merra, bins=overlay_bins, alpha=0.5, label="MERRA-2")

plt.axvline(p10_era, color="blue", linestyle="--",
            label=f"ERA5 10th pct = {p10_era:.3e}")

plt.axvline(p10_merra, color="orange", linestyle="--",
            label=f"MERRA-2 10th pct = {p10_merra:.3e}")

plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim([0, 1e6])

plt.xlabel("Cloud Ice Mass + Cloud Liquid Mass (kg/kg)")
plt.ylabel("Occurrences per year (Count)")
plt.title("ERA5 vs MERRA-2 Overlay")

plt.savefig("/home/karengarcia/criteria_testing/ERA5_vs_MERRA_ice_histogram.png",
            dpi=300, bbox_inches='tight')

print("Done.")