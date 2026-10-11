import os
import calendar
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==========================================================
# USER SETTINGS
# ==========================================================

YEARS = range(1979, 2026)

MONTHS = list(range(1, 13))
HOURS = ["00", "03", "06", "09", "12", "15", "18", "21"]

# Domain
LAT_MIN = -15
LAT_MAX = 30

LON_MIN = 40
LON_MAX = 110

QUIVER_STRIDE = 5

# Output
OUTFILE = "/home/sagim/SLP_Wind_Climatology.png"
NETCDF_FILE = "/home/sagim/SLP_Climatology1.nc"
# ERA5 paths
BASE_2D = "/work/meteogroup/era5_1h_2d"
BASE_B  = "/work/meteogroup/era5"


# ==========================================================
# HELPERS
# ==========================================================

def open_ds(path):
    return xr.open_dataset(path, decode_times=False)


def subset(ds):

    lat_name = "lat"
    lon_name = "lon"

    lat = ds[lat_name]

    if lat[0] < lat[-1]:
        ds = ds.sel(
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX)
        )
    else:
        ds = ds.sel(
            lat=slice(LAT_MAX, LAT_MIN),
            lon=slice(LON_MIN, LON_MAX)
        )

    return ds


def make_paths(year, month, day, hour):

    ymd = f"{year}{month:02d}{day:02d}"

    sfc_path = f"{BASE_2D}/{year}/{month:02d}/SFC{ymd}_{hour}"
    b_path   = f"{BASE_B}/{year}/{month:02d}/B{ymd}_{hour}"

    return sfc_path, b_path

# ==========================================================
# MAIN
# ==========================================================

lat = None
lon = None

# Will hold the 12 monthly climatologies
slp_monthly = []
u_monthly = []
v_monthly = []
slp_monthly_anom = []

for month in MONTHS:

    print(f"\nProcessing month {month:02d}")

    # Reset accumulators for THIS MONTH ONLY
    slp_sum = None
    u_sum = None
    v_sum = None

    slp_count = None
    u_count = None
    v_count = None

    count = 0

    for year in YEARS:

        print(f"   Year {year}")

        ndays = calendar.monthrange(year, month)[1]

        for day in range(1, ndays + 1):

            for hour in HOURS:

                sfc_path, b_path = make_paths(
                    year,
                    month,
                    day,
                    hour
                )

                if not os.path.exists(sfc_path):
                    continue

                if not os.path.exists(b_path):
                    continue

                try:

                    ds_sfc = subset(open_ds(sfc_path))
                    ds_b = subset(open_ds(b_path))

                    slp = ds_sfc["SLP"].squeeze().values.astype(float)
                    u10 = ds_b["U10M"].squeeze().values.astype(float)
                    v10 = ds_b["V10M"].squeeze().values.astype(float)

                    slp[~np.isfinite(slp)] = np.nan
                    u10[~np.isfinite(u10)] = np.nan
                    v10[~np.isfinite(v10)] = np.nan
                    # Remove unrealistic values from occasional corrupt files
                    slp[np.abs(slp) > 2000] = np.nan
                    u10[np.abs(u10) > 300] = np.nan
                    v10[np.abs(v10) > 300] = np.nan

                    if slp_sum is None:

                        slp_sum = np.zeros_like(slp)
                        u_sum = np.zeros_like(u10)
                        v_sum = np.zeros_like(v10)

                        slp_count = np.zeros_like(slp)
                        u_count = np.zeros_like(u10)
                        v_count = np.zeros_like(v10)

                        lat = ds_sfc["lat"].values
                        lon = ds_sfc["lon"].values

                    valid = np.isfinite(slp)
                    slp_sum[valid] += slp[valid]
                    slp_count[valid] += 1

                    valid = np.isfinite(u10)
                    u_sum[valid] += u10[valid]
                    u_count[valid] += 1

                    valid = np.isfinite(v10)
                    v_sum[valid] += v10[valid]
                    v_count[valid] += 1

                    ds_sfc.close()
                    ds_b.close()

                    count += 1

                except Exception as e:

                    print(
                        f"{year}-{month:02d}-{day:02d} {hour}: {e}"
                    )

    print(f"   Processed timesteps: {count}")

    # ------------------------------------------------------
    # Compute monthly climatology
    # ------------------------------------------------------

    slp_mean = slp_sum / slp_count
    u_mean = u_sum / u_count
    v_mean = v_sum / v_count

    slp_mean[slp_count == 0] = np.nan
    u_mean[u_count == 0] = np.nan
    v_mean[v_count == 0] = np.nan
    # SLP anomaly relative to standard atmosphere (1013 hPa)
    slp_anom = slp_mean - 1013.0

    # Save this month's climatology
    slp_monthly.append(slp_mean)
    slp_monthly_anom.append(slp_anom)
    u_monthly.append(u_mean)
    v_monthly.append(v_mean)


# ==========================================================
# SAVE NETCDF
# ==========================================================

month_vals = np.arange(1, 13)

ds_out = xr.Dataset(

    coords={
        "month": ("month", month_vals),
        "lat": ("lat", lat),
        "lon": ("lon", lon),
    }

)

ds_out["SLP_mean"] = (
    ("month", "lat", "lon"),
    np.stack(slp_monthly, axis=0)
)
ds_out["SLP_anomaly"] = (
    ("month", "lat", "lon"),
    np.stack(slp_monthly_anom, axis=0)
)

ds_out["U10_mean"] = (
    ("month", "lat", "lon"),
    np.stack(u_monthly, axis=0)
)

ds_out["V10_mean"] = (
    ("month", "lat", "lon"),
    np.stack(v_monthly, axis=0)
)

ds_out["SLP_mean"].attrs["units"] = "hPa"
ds_out["U10_mean"].attrs["units"] = "m s-1"
ds_out["V10_mean"].attrs["units"] = "m s-1"
ds_out["SLP_anomaly"].attrs["units"] = "hPa"
ds_out["SLP_anomaly"].attrs["description"] = (
    "Sea level pressure anomaly relative to 1013 hPa"
)
ds_out.attrs["years"] = f"{YEARS.start}-{YEARS.stop-1}"
ds_out.attrs["hours"] = ",".join(HOURS)

encoding = {
    var: {"zlib": True, "complevel": 4}
    for var in ds_out.data_vars
}

ds_out.to_netcdf(
    NETCDF_FILE,
    encoding=encoding
)

print("Saved:", NETCDF_FILE)

# ==========================================================
# PLOT MONTHLY CLIMATOLOGY
# ==========================================================

MONTH_NAMES = [
    "Jan","Feb","Mar","Apr",
    "May","Jun","Jul","Aug",
    "Sep","Oct","Nov","Dec"
]


def setup_map(ax):

    ax.set_extent(
        [LON_MIN, LON_MAX,
         LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    gl = ax.gridlines(draw_labels=True)

    gl.top_labels = False
    gl.right_labels = False


# ----------------------------------------------------------
# COMMON COLOR LIMITS
# ----------------------------------------------------------

all_slp = np.stack(slp_monthly)

vmin = np.floor(np.nanmin(all_slp) / 2) * 2
vmax = np.ceil(np.nanmax(all_slp) / 2) * 2

fill_levels = np.arange(vmin, vmax + 1, 1)
line_levels = np.arange(vmin, vmax + 2, 2)


# ==========================================================
# FIGURE 1
# SHADED SLP
# ==========================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(18,12),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

axes = axes.ravel()

mappable = None

for month in range(12):

    ax = axes[month]

    setup_map(ax)

    slp = slp_monthly[month]
    u = u_monthly[month]
    v = v_monthly[month]

    mesh = ax.contourf(
        lon,
        lat,
        slp,
        levels=fill_levels,
        cmap="RdBu_r",
        extend="both",
        transform=ccrs.PlateCarree()
    )

    mappable = mesh

    ii = slice(None,None,QUIVER_STRIDE)
    jj = slice(None,None,QUIVER_STRIDE)

    q = ax.quiver(
        lon[jj],
        lat[ii],
        u[ii,jj],
        v[ii,jj],
        transform=ccrs.PlateCarree(),
        scale=250
    )

    ax.set_title(MONTH_NAMES[month])

fig.colorbar(
    mappable,
    ax=axes,
    orientation="horizontal",
    fraction=0.05,
    pad=0.06,
    label="SLP (hPa)"
)

ax.quiverkey(
    q,
    X=0.88,
    Y=-0.10,
    U=2,
    label="2 m s$^{-1}$",
    labelpos="E"
)

plt.suptitle(
    f"Monthly Mean SLP (Shaded) + 10-m Winds\n"
    f"{YEARS.start}-{YEARS.stop-1}",
    fontsize=16
)

plt.savefig(
    "/home/sagim/SLP_Shading_12Months.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# FIGURE 2
# CONTOUR SLP
# ==========================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(18,12),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

axes = axes.ravel()

for month in range(12):

    ax = axes[month]

    setup_map(ax)

    slp = slp_monthly[month]
    u = u_monthly[month]
    v = v_monthly[month]

    cs = ax.contour(
        lon,
        lat,
        slp,
        levels=line_levels,
        colors="black",
        linewidths=1.2,
        transform=ccrs.PlateCarree()
    )

    ax.clabel(
        cs,
        fontsize=7,
        fmt="%.0f"
    )

    ii = slice(None,None,QUIVER_STRIDE)
    jj = slice(None,None,QUIVER_STRIDE)

    q = ax.quiver(
        lon[jj],
        lat[ii],
        u[ii,jj],
        v[ii,jj],
        transform=ccrs.PlateCarree(),
        scale=250
    )

    ax.set_title(MONTH_NAMES[month])

ax.quiverkey(
    q,
    X=0.88,
    Y=-0.10,
    U=2,
    label="2 m s$^{-1}$",
    labelpos="E"
)

plt.suptitle(
    f"Monthly Mean SLP (Contours) + 10-m Winds\n"
    f"{YEARS.start}-{YEARS.stop-1}",
    fontsize=16
)

plt.savefig(
    "/home/sagim/SLP_Contours_12Months.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print("/home/sagim/SLP_Shading_12Months.png")
print("/home/sagim/SLP_Contours_12Months.png")