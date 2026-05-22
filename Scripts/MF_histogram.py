import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset("/home/karengarcia/criteria_testing/MERRA_flux_over_tp.nc")
data_merra = ds["Mass_flux_above_tp"].values
data_merra = data_merra[np.isfinite(data_merra)]
data_merra = data_merra[data_merra > 0]

bins_merra = np.logspace(np.log10(data_merra.min()), np.log10(data_merra.max()), 100)
p10_merra = np.percentile(data_merra, 10)

plt.figure(figsize=(16,6))
plt.hist(data_merra, bins=bins_merra, color="skyblue", edgecolor='black', alpha=0.9)
plt.axvline(p10_merra, color="red", linestyle="--",
            linewidth=2, label=f"10th percentile = {p10_merra:.3e} kg/m$^{2}$s")

plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylim([0, 2e4])

plt.xlabel(f"Mass flux  (kg/m$^{2}$s)")
plt.ylabel("Occurrences per year (Count)")
plt.title("MERRA-2 Histogram of Mass Flux Over the Tropopause")

plt.savefig("/home/karengarcia/criteria_testing/MERRA_mf_histogram.png",
            dpi=300, bbox_inches='tight')