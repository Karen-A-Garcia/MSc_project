import numpy as np
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

g = 9.81
R_EARTH = 6371000.0


def pressure_layer_thickness(levels, pressure_factor=1):
    """
    This function computes the pressure layer thicknesses (Pa)
    """
    p = levels.values * pressure_factor

    edges = np.concatenate([
        [p[0] - (p[1] - p[0]) / 2],
        (p[:-1] + p[1:]) / 2,
        [p[-1] + (p[-1] - p[-2]) / 2]
    ])

    return np.abs(np.diff(edges))


def column_water_vapor(q, level_dim, pressure_factor=1):
    """
    
    This function integrates specific humidity vertically above control level.
    Returns kg m^-2.
    """

    dp = xr.DataArray(
        pressure_layer_thickness(q[level_dim], pressure_factor),
        coords=[q[level_dim]],
        dims=[level_dim]
    )

    q_layer = q.isel({level_dim: slice(0, -1)})
    dp = dp.isel({level_dim: slice(0, -1)})

    return (q_layer * dp).sum(dim=level_dim) / g


def grid_cell_area(lat, lon):
    """
    Area of each grid cell.
    """

    dlon = np.deg2rad(lon.diff("lon").values[0])
    dlat = np.deg2rad(abs(lat.diff("lat").values[0]))

    return (R_EARTH**2* np.cos(np.deg2rad(lat))* dlon* dlat)


def integrated_masses(W, lat, lon):
    """
    Global, tropical, and extratropical masses.
    """

    area = grid_cell_area(lat, lon)

    tropics = (lat >= -30) & (lat <= 30)
    extratropics = ~tropics

    global_mass = (
        W * area
    ).sum(dim=["lat", "lon"]).compute()

    tropics_mass = (
        W.where(tropics) * area
    ).sum(dim=["lat", "lon"]).compute()

    extratropics_mass = (
        W.where(extratropics) * area
    ).sum(dim=["lat", "lon"]).compute()

    return global_mass, tropics_mass, extratropics_mass


def annual_mean(series, time_dim):
    """
    Annual mean time series in for water vapour in the stratosphere in Teragrams (x10^12g).
    """

    annual = (series / 1e9).groupby(f"{time_dim}.year").mean()

    annual_time = [np.datetime64(f"{yr}-07-01")
                   for yr in annual.year.values]

    return annual, annual_time


def compute_dQs_dt(mass, start_date, end_date):
    """
    Compute dQs/dt in Tg month^-1.
    """

    mass = mass.copy()

    if "valid_time" in mass.dims:
        mass = mass.rename({"valid_time": "time"})

    if hasattr(mass.indexes["time"], "to_datetimeindex"):
        mass["time"] = mass.indexes["time"].to_datetimeindex()

    mass = mass.sel(time=slice(start_date, end_date))
    mass = mass / 1e9
    time_sec = (mass.time - mass.time[0]
                ).dt.total_seconds()

    tendency = (mass.assign_coords(time=time_sec)
                .differentiate("time")
                * (3600 * 24 * 30.44))

    return tendency.assign_coords(time=mass.time)


# CANAM5
CAN_DIR = "/home/karengarcia/downloads-karengarcia/ESGF_downloads/Monthly"
CAN_ds = xr.open_mfdataset(sorted(glob.glob(os.path.join(CAN_DIR,
                        "hus_Amon_CanESM5_amip_r1i1p2f1_gn_*.nc"))),chunks={"time": 120})
CAN_ds = CAN_ds.sel(plev=slice(10000, 0)).isel(time=slice(-12*10, None))
CAN_W = column_water_vapor(CAN_ds["hus"],"plev")

(CAN_global,CAN_tropics,CAN_extra) = integrated_masses(CAN_W,
                                                       CAN_ds.lat,
                                                       CAN_ds.lon)


# ERA5
ERA_DIR = "/home/karengarcia/downloads-karengarcia/ERA5/Monthly"
ERA_ds = xr.open_mfdataset(sorted(glob.glob(os.path.join(ERA_DIR,
                        "ERA5_monthly_q_1979_2019_128x64.nc"))),chunks={"valid_time": 120})
ERA_ds = ERA_ds.sel(pressure_level=slice(100, 0)).isel(valid_time=slice(-12*15, -12*5))
ERA_W = column_water_vapor(ERA_ds["q"],
                           "pressure_level",
                           pressure_factor=100)

(ERA_global,
 ERA_tropics,
 ERA_extra) = integrated_masses(ERA_W,
                                ERA_ds.lat,
                                ERA_ds.lon)

# MLS

MLS_DIR = "/home/karengarcia/downloads-karengarcia/MLS_data/v06"

MLS_ds = xr.open_mfdataset(sorted(glob.glob(os.path.join(MLS_DIR,
                        "MLS-Aura_L3MB-H2O_v05*20*.nc")
                                            )),
    group="H2O PressureGrid",
    combine="by_coords")

MLS_H2O = MLS_ds["value"].fillna(0)
mmr_wtda = 18.015 / 28.96

MLS_q = (MLS_H2O * mmr_wtda) / (1 + MLS_H2O * mmr_wtda)
MLS_q = MLS_q.sel(lev=slice(100, 0)).isel(time=slice(None, -12*3))
MLS_W = column_water_vapor(MLS_q,
                           "lev",
                           pressure_factor=100)

(MLS_global,
 MLS_tropics,
 MLS_extra) = integrated_masses(MLS_W,
                                MLS_q.lat,
                                MLS_q.lon)


# ANNUAL MEANS
CAN_ann, CAN_ann_time = annual_mean(CAN_global,
    "time")

ERA_ann, ERA_ann_time = annual_mean(ERA_global,
    "valid_time")

MLS_ann, MLS_ann_time = annual_mean(MLS_global,
    "time")

# dQs/dt
START = "2005-01-01"
END = "2014-12-31"

CAN_dQs = compute_dQs_dt(CAN_global,
                         START,
                         END)

ERA_dQs = compute_dQs_dt(ERA_global,
                         START,
                         END)

MLS_dQs = compute_dQs_dt(MLS_global,
                         START,
                         END)



plt.figure(figsize=(15, 6))
plt.plot(MLS_global.time, MLS_global / 1e9,
    color="black",label="MLS")

plt.plot(CAN_global.time,CAN_global / 1e9,
    color="red",linestyle="--", label="CanAM5")

plt.plot(ERA_global.valid_time,ERA_global / 1e9,
        color="blue",linestyle=":",label="ERA5")

plt.ylabel("Mass (Tg)")
plt.xlabel("Time")
plt.title(
    "Integrated Stratospheric Water Vapour Above 100 hPa"
)
plt.legend()
plt.tight_layout()
final_plot_path = ("/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/New_script_test.png")
plt.savefig(final_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {final_plot_path}")    


def seasonal_zonal_mean(q, time_dim):
    """
    Compute seasonal climatological zonal mean.

    Returns:
        seasonal(lat, level)
    """
    return (
        q.groupby(f"{time_dim}.season")
        .mean(time_dim)
        .mean("lon")
    )
    
CAN_seasonal = seasonal_zonal_mean(CAN_ds["hus"], "time")
ERA_seasonal = seasonal_zonal_mean(ERA_ds["q"], "valid_time")
MLS_seasonal = seasonal_zonal_mean(MLS_q, "time")

def plot_seasonal_pressure_latitude(
    seasonal_data,
    lat,
    plev,
    title,
    output_file,
    levels=None,
    ymin=None,
    ymax=None):
    """
    Seasonal pressure-latitude cross sections.
    """

    seasons = ["DJF", "MAM", "JJA", "SON"]

    if levels is None:
        levels = np.linspace(
            float(seasonal_data.min()),
            float(seasonal_data.max()),
            21
        )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        sharex=True,
        sharey=True
    )

    for ax, season in zip(axes.flat, seasons):

        season_data = seasonal_data.sel(season=season)

        cf = ax.contourf(
            lat,
            plev,
            season_data,
            levels=levels,
            cmap="viridis",
            extend="both"
        )

        ax.set_yscale("log")

        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(float(np.max(plev)),
                        float(np.min(plev)))

        # Pressure decreases upward
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.set_ylim(float(np.max(plev)),
                    float(np.min(plev)))

        ax.set_title(season)

    # Common labels
    fig.supxlabel("Latitude")
    fig.supylabel("Pressure (hPa)")

    # Leave room for colorbar
    fig.subplots_adjust(
        right=0.88,
        wspace=0.15,
        hspace=0.20
    )

    # Colorbar outside panel grid
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])

    fig.colorbar(
        cf,
        cax=cax,
        label="Specific humidity (kg kg$^{-1}$)"
    )

    fig.suptitle(title)

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Plot saved to:", output_file)
    
plot_seasonal_pressure_latitude(CAN_seasonal,
                                CAN_ds.lat,
                                CAN_ds.plev / 100,   # Pa -> hPa
                                "CanAM5 Seasonal Mean Water Vapour",
    "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/CANAM5_seasonal_lat_pressure.png")

plot_seasonal_pressure_latitude(ERA_seasonal,
                                ERA_ds.lat,
                                ERA_ds.pressure_level,
                                "ERA5 Seasonal Mean Water Vapour",
    "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/ERA5_seasonal_lat_pressure.png")

plot_seasonal_pressure_latitude(
    MLS_seasonal.sel(lev=slice(100,1)),
    MLS_q.lat,
    MLS_q.lev.sel(lev=slice(100,1)),
    "MLS Seasonal Mean Water Vapour",
    "/home/karengarcia/MSc_project/Figures/Water_budget/Control_volume/MLS_seasonal_lat_pressure.png",
    ymin=100,
    ymax=1
)