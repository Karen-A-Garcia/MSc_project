import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

file_path = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Daily/hus/hus_day_CanESM5_amip_r1i1p2f1_gn_20110101-20141231.nc"
ds = xr.open_dataset(file_path)

hus = ds["hus"]  # Specific humidity
plev = ds["plev"]  # Pressure levels in Pa
daylnt = 86400.0  # Seconds in a day
eps1 = 0.62195  # Ratio of mol weight of water to dry air
qreff = 6.8e-6 * eps1
tauscal = 100.0
dtadv = 900.0

p = plev.broadcast_like(hus)

# Calculate relaxation factors (fact1 and fact2)
ptop1 = 150.0
x1 = (np.maximum(p, ptop1) / ptop1) ** 1.50
fact1 = 1.0 / (tauscal * daylnt * x1)

pbot2 = 20.0
ptop2 = 0.1
x2 = (
    6000.0 * (p / pbot2) ** 4
    + 0.4 * (np.maximum(p, ptop2) / ptop2) ** 1.5
    + 2.6
)
fact2 = xr.where(p < pbot2, 1.0 / (x2 * daylnt), 0.0)

qmra = hus / (1.0 - hus)  # Final mass mixing ratio
qmr = (qmra * (1.0 + (fact1 + fact2) * dtadv)) - (
    fact1 * dtadv * qreff
)  # Initial mass mixing ratio

hus_old = qmr / (1.0 + qmr)
h2o_tendency = (hus - hus_old) / dtadv

# Methane mass mixing ratio tendency (s^-1)
delta_qmr = qmra - qmr
ch4_tendency = (-0.5 * (16.04 / 18.02) * delta_qmr) / dtadv

# Apply the stratospheric mask (only active at or above 50 hPa)
h2o_tendency = xr.where(p <= 5000.0, h2o_tendency, 0.0)
ch4_tendency = xr.where(p <= 5000.0, ch4_tendency, 0.0)

# 5. Compute the Cosine-Latitude Weighted Global Average
weights = np.cos(np.deg2rad(ds["lat"]))
weights.name = "weights"

h2o_global = h2o_tendency.weighted(weights).mean(dim=["lat", "lon"])
ch4_global = ch4_tendency.weighted(weights).mean(dim=["lat", "lon"])

# Average vertically across only the stratospheric levels (plev <= 5000 Pa)
h2o_strat_ts = h2o_global.where(h2o_global.plev <= 5000.0, 
                                drop=True).mean(
                                    dim="plev")
                                
ch4_strat_ts = ch4_global.where(ch4_global.plev <= 5000.0, 
                                drop=True).mean(
                                    dim="plev")

# 6. Plotting both on a single plot with twin Y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary Y-axis (Left): Stratospheric Water Vapor Tendency
h2o_strat_ts.plot(ax=ax1, color="blue", linewidth=1.5, label=r"$\mathregular{H_2O}$ Gain")
ax1.set_title(
    "Globally & Stratospherically Averaged Atmospheric Tendencies",
    fontsize=14,
    fontweight="bold",
)
ax1.set_ylabel(
    r"$\mathregular{H_2O}$ Tendency ($\mathregular{kg kg^{-1}s^{-1}}$)",
    color="blue",
    fontsize=12,
)
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, linestyle="--", alpha=0.5)

# Create a secondary Y-axis (Right) that shares the same x-axis
ax2 = ax1.twinx()
ch4_strat_ts.plot(ax=ax2, color="red", linewidth=1.5, label=r"$\mathregular{CH_4}$ Loss")
ax2.set_ylabel(
    r"$\mathregular{CH_4}$ Oxidation Rate ($\mathregular{kg kg^{-1}s^{-1}}$)",
    color="red",
    fontsize=12,
)
ax2.tick_params(axis="y", labelcolor="red")

# Clean up X-axis label
ax1.set_xlabel("Time", fontsize=12)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
figure_path = "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/global_methane_h2o_tendency_timeseries.png"
plt.savefig(figure_path, dpi=300)
print("Plot successfully saved to:", figure_path)