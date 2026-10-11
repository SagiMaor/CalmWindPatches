# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:15:30 2026

@author: sagim
"""

# -*- coding: utf-8 -*-

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize

# =====================================================
# USER SETTINGS
# =====================================================

DB = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\ClimChange\PatchDatabase.sqlite"
)

OUT_DIR = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\ClimChange\ERA5monthly"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

OUTFILE = os.path.join(
    OUT_DIR,
    "ERA5_IO_MonthlyClimatology.png"
)
DPI = 300

CMAP = "inferno"

IO_EXTENT = [40,100,-15,25]

LSM_PATH = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\ClimChange\landseacover.nc"
)

BUFFER_KM = 100.0

# =====================================================
# ERA5 GRID
# =====================================================

lat = np.arange(-90, 90.5, 0.5)

lon = np.arange(-180, 180, 0.5)
Ny = len(lat)
Nx = len(lon)

# =====================================================
# LOAD DATABASE
# =====================================================

print("Loading database...")

conn = sqlite3.connect(DB)

patches = pd.read_sql(
    """
    SELECT
        patch_id,
        time,
        area
    FROM patches
    """,
    conn
)

pixels = pd.read_sql(
    "SELECT patch_id,row,col FROM patch_pixels",
    conn
)

conn.close()

print("Grouping pixels...")

pixel_groups = pixels.groupby("patch_id")

patches["time"] = pd.to_datetime(
    patches["time"]
)

patches["month"] = patches["time"].dt.month

patches["area"] = pd.to_numeric(
    patches["area"],
    errors="coerce"
)

# =====================================================
# DOMAIN AREA FUNCTIONS
# =====================================================

def wrap_lon(lon):

    return (
        (lon + 180) % 360
    ) - 180


def compute_cell_area(lat, lon):

    R = 6371.0

    lat_r = np.deg2rad(lat)

    dlat = np.abs(
        lat_r[1] - lat_r[0]
    )

    dlon = np.deg2rad(
        np.abs(lon[1] - lon[0])
    )

    area_lat = (
        (R ** 2)
        * dlon
        * (
            np.sin(lat_r + dlat / 2)
            -
            np.sin(lat_r - dlat / 2)
        )
    )

    return np.abs(
        np.repeat(
            area_lat[:, None],
            len(lon),
            axis=1
        )
    )


def calculate_io_domain_area():

    ds = xr.open_dataset(
        LSM_PATH
    )

    da = list(
        ds.data_vars.values()
    )[0]

    while da.ndim > 2:

        da = da.isel(
            {da.dims[0]: 0}
        )

    land = (
        da.values > 0.5
    )

    lsm_lat = ds.lat.values
    lsm_lon = ds.lon.values

    ds.close()

    # -------------------------------------------------
    # 100-km land buffer
    # Same exclusion used for patch detection
    # -------------------------------------------------

    R = 6371.0

    lon_w = wrap_lon(
        lsm_lon
    )

    lat2d, lon2d = np.meshgrid(
        lsm_lat,
        lon_w,
        indexing="ij"
    )

    x = (
        R
        * np.deg2rad(lon2d)
        * np.cos(np.deg2rad(lat2d))
    )

    y = (
        R
        * np.deg2rad(lat2d)
    )

    land_idx = np.where(
        land
    )

    tree = cKDTree(
        np.column_stack(
            (
                x[land_idx],
                y[land_idx]
            )
        )
    )

    pts = np.column_stack(
        (
            x.ravel(),
            y.ravel()
        )
    )

    dist, _ = tree.query(
        pts,
        k=1
    )

    dist = dist.reshape(
        lat2d.shape
    )

    keep_mask = (
        (~land)
        &
        (dist >= BUFFER_KM)
    )

    # -------------------------------------------------
    # Indian Ocean study domain
    # -------------------------------------------------

    lon_min, lon_max, lat_min, lat_max = IO_EXTENT

    io_mask = (
        (lat2d >= lat_min)
        &
        (lat2d <= lat_max)
        &
        (lon2d >= lon_min)
        &
        (lon2d <= lon_max)
    )

    domain_mask = (
        keep_mask
        &
        io_mask
    )

    # -------------------------------------------------
    # Physical grid-cell areas
    # -------------------------------------------------

    area_2d = compute_cell_area(
        lsm_lat,
        lsm_lon
    )

    domain_area = np.nansum(
        area_2d[domain_mask]
    )

    print(
        f"ERA5 IO ocean domain area: "
        f"{domain_area:,.0f} km²"
    )

    return domain_area

def calculate_monthly_coverage(
        patches,
        domain_area):

    # -------------------------------------------------
    # Use only valid patch areas
    # -------------------------------------------------

    area_df = patches[
        np.isfinite(patches["area"])
        &
        (patches["area"] > 0)
    ].copy()

    # -------------------------------------------------
    # Total calm-patch area at every ERA5 time step
    # -------------------------------------------------

    three_hourly_area = (
        area_df
        .groupby("time")["area"]
        .sum()
        .sort_index()
    )

    # -------------------------------------------------
    # Complete ERA5 3-hourly timeline
    #
    # Time steps containing no calm patches must
    # contribute zero area.
    # -------------------------------------------------

    full_time = pd.date_range(
        start=patches["time"].min().floor("D"),
        end=patches["time"].max().floor("D")
            + pd.Timedelta(hours=21),
        freq="3h"
    )

    three_hourly_area = three_hourly_area.reindex(
        full_time,
        fill_value=0.0
    )

    # -------------------------------------------------
    # Daily mean patch area
    # Mean of the eight 3-hourly analyses
    # -------------------------------------------------

    daily_area = (
        three_hourly_area
        .resample("D")
        .mean()
    )

    # -------------------------------------------------
    # Monthly mean daily patch area
    # -------------------------------------------------

    monthly_area = (
        daily_area
        .resample("MS")
        .mean()
    )

    # -------------------------------------------------
    # Fraction of ocean domain occupied
    # -------------------------------------------------

    monthly_coverage = (
        100.0
        * monthly_area
        / domain_area
    )

    # -------------------------------------------------
    # Monthly climatology
    # -------------------------------------------------

    monthly_climatology = (
        monthly_coverage
        .groupby(
            monthly_coverage.index.month
        )
        .mean()
    )

    return monthly_climatology

def plot_monthly_coverage(
        monthly_climatology,
        outfile):

    months = np.arange(
        1,
        13
    )

    month_labels = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    # Same source width as monthly-map figure
    fig, ax = plt.subplots(
        figsize=(21, 3.2),
        dpi=DPI
    )

    ax.plot(
        months,
        monthly_climatology.values,
        marker="o",
        linewidth=2.5,
        markersize=7
    )

    ax.set_ylabel(
        "Ocean Coverage (%)",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xticks(
        months
    )

    ax.set_xticklabels(
        month_labels,
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlim(
        0.7,
        12.3
    )

    ax.tick_params(
        axis="y",
        labelsize=14
    )

    for tick in ax.get_yticklabels():
        tick.set_fontweight(
            "bold"
        )

    ax.grid(
        axis="y",
        alpha=0.20,
        linestyle="--",
        linewidth=0.8
    )

    ax.set_axisbelow(
        True
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=DPI,
        facecolor="white"
    )

    plt.close(fig)

    print(
        f"[SAVE] {outfile}"
    )

# =====================================================
# BUILD MONTHLY DENSITY
# =====================================================

monthly = {}

all_values = []

for month in range(1,13):

    density = np.zeros(
        (Ny,Nx),
        dtype=np.uint32
    )

    subset = patches[
        patches.month == month
    ]

    print(
        f"Month {month}:",
        len(subset),
        "patches"
    )

    for pid in subset.patch_id:

        if pid not in pixel_groups.groups:
            continue

        pts = pixel_groups.get_group(pid)

        density[
            pts.row.values,
            pts.col.values
        ] += 1

    monthly[month] = density

    vals = density[density>0]

    if len(vals):
        all_values.extend(vals)

# =====================================================
# COLOR SCALE
# =====================================================

vmax = int(
    np.ceil(
        np.percentile(
            all_values,
            99
        )
    )
)

print(
    f"99th percentile: "
    f"{np.percentile(all_values, 99):.1f}"
)

print(
    f"Maximum: "
    f"{np.max(all_values)}"
)

norm = Normalize(
    vmin=1,
    vmax=vmax
)
# =====================================================
# FIGURE
# =====================================================

proj = ccrs.PlateCarree()

fig, axes = plt.subplots(
    3,
    4,
    figsize=(21, 12),
    subplot_kw={
        "projection": proj
    }
)

plt.subplots_adjust(
    left=0.03,
    right=0.99,
    top=0.965,
    bottom=0.12,
    wspace=0.010,
    hspace=0.10
)

axes = axes.ravel()

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

# =====================================================
# DRAW MAPS
# =====================================================

for month in range(12):

    ax = axes[month]

    # ----------------------------------------------
    # Extent
    # ----------------------------------------------

    ax.set_extent(
        IO_EXTENT,
        crs=proj
    )

    # ----------------------------------------------
    # Background
    # ----------------------------------------------

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.85"
    )

    ax.coastlines(
        linewidth=0.5
    )

    # ----------------------------------------------
    # Gridlines
    # ----------------------------------------------

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        linestyle=":",
        color="gray",
        alpha=0.45
    )

    gl.xlabel_style = {
        "size": 13,
        "weight": "bold"
    }

    gl.ylabel_style = {
        "size": 13,
        "weight": "bold"
    }

    gl.top_labels = False
    gl.right_labels = False

    # Latitude labels only on left column
    if month not in [0, 4, 8]:
        gl.left_labels = False

    # Longitude labels only on bottom row
    if month < 8:
        gl.bottom_labels = False

    # ----------------------------------------------
    # Density
    # ----------------------------------------------

    masked = np.ma.masked_where(
        monthly[month + 1] == 0,
        monthly[month + 1]
    )

    mesh = ax.pcolormesh(
        lon,
        lat,
        masked,
        shading="auto",
        cmap=CMAP,
        norm=norm,
        transform=proj
    )

    # ----------------------------------------------
    # Month title
    # ----------------------------------------------

    ax.set_title(
        month_names[month],
        fontsize=16,
        fontweight="bold",
        pad=6
    )

# =====================================================
# SHARED COLORBAR
# =====================================================

cbar = fig.colorbar(
    mesh,
    ax=axes,
    orientation="horizontal",
    shrink=0.60,
    pad=0.035,
    aspect=60,
    fraction=0.025
)

cbar.set_label(
    "Event Count",
    fontsize=13,
    fontweight="bold"
)

cbar.ax.tick_params(
    labelsize=13,
    width=1,
    length=4
)

for tick in cbar.ax.get_xticklabels():
    tick.set_fontweight("bold")

# =====================================================
# SAVE
# =====================================================

plt.savefig(
    OUTFILE,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print(
    f"[SAVE] {OUTFILE}"
)


# =====================================================
# MONTHLY OCEAN COVERAGE
# =====================================================

domain_area = calculate_io_domain_area()

monthly_climatology = calculate_monthly_coverage(
    patches,
    domain_area
)

print()
print("Monthly ERA5 ocean coverage:")

for month, value in monthly_climatology.items():

    print(
        f"{month:02d}: "
        f"{value:.3f}%"
    )

coverage_file = os.path.join(
    OUT_DIR,
    "ERA5_IO_MonthlyCoverage.png"
)

plot_monthly_coverage(
    monthly_climatology,
    coverage_file
)