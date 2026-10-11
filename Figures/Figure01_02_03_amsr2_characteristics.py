# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:27:20 2026

@author: sagim
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:13:57 2026

@author: sagim
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.gridspec import GridSpec

# =====================================================
# USER SETTINGS
# =====================================================

MIN_LIFETIME = 1

# =====================================================
# TITLE LABEL
# =====================================================

if MIN_LIFETIME == 1:
    LIFETIME_LABEL = "Lifetime ≥ 1 Pass"
else:
    LIFETIME_LABEL = f"Lifetime ≥ {MIN_LIFETIME} Passes"

IO_EXTENT = [40, 100, -15, 25]
CP_EXTENT = [-115, -80, 5, 25]
EP_EXTENT = [-120, -80, -5, 5]
WA_EXTENT = [-20, -10, 0, 15]
PWP_EXTENT = [130, 180, -5, 5]

GRID_FILE = r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\AMSR2\2016\RSS_AMSR2_ocean_L3_daily_2016-01-02_v08.2.nc"


STATIONARY_CSV = r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\SavedData\PatchLifetime_0.92\StationaryEvents_GLOBAL.csv"

STATIONARY_NPZ = r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\SavedData\PatchLifetime_0.92\StationaryEventFootprints_GLOBAL.npz"

DPI = 300
CMAP = "inferno"
# =====================================================
# OUTPUT
# =====================================================

PLOT_DIR = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\SavedData\LifeTime Plots2"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)
# =====================================================
# LOAD GRID
# =====================================================

ds = xr.open_dataset(GRID_FILE)

lats = ds["lat"].values
lons = ds["lon"].values

# =====================================================
# LAND MASK FOR DOMAIN-AREA CALCULATION
# =====================================================

land_da = ds["land_mask"]

# land_mask contains a pass dimension in the AMSR2 files.
# The geographical land mask should be the same for both passes,
# so use the first pass.
if "pass" in land_da.dims:
    land_da = land_da.isel(**{"pass": 0})

land_mask = land_da.values

ds.close()

Ny = len(lats)
Nx = len(lons)

lons_wrapped = ((lons + 180) % 360) - 180
sort_idx = np.argsort(lons_wrapped)

lons_wrapped = lons_wrapped[sort_idx]

# =====================================================
# LOAD EVENTS
# =====================================================

stationary_df = pd.read_csv(STATIONARY_CSV)
stationary_df = stationary_df[
    stationary_df["lifetime_passes"] >= MIN_LIFETIME
].copy()

stationary_npz = np.load(STATIONARY_NPZ)


stationary_df["month"] = pd.to_datetime(
    stationary_df["start_stamp"].str[:10]
).dt.month

# =====================================================
# INDIAN OCEAN STATIONARY EVENTS
# =====================================================

stationary_io = stationary_df[
    (stationary_df["core_lat"] >= IO_EXTENT[2]) &
    (stationary_df["core_lat"] <= IO_EXTENT[3]) &
    (stationary_df["core_lon"] >= IO_EXTENT[0]) &
    (stationary_df["core_lon"] <= IO_EXTENT[1])
].copy()

stationary_cp = stationary_df[
    (stationary_df["core_lat"] >= 5) &
    (stationary_df["core_lat"] <= 25) &
    (stationary_df["core_lon"] >= -110) &
    (stationary_df["core_lon"] <= -80)
].copy()

stationary_ep = stationary_df[
    (stationary_df["core_lat"] >= -5) &
    (stationary_df["core_lat"] <= 5) &
    (stationary_df["core_lon"] >= -120) &
    (stationary_df["core_lon"] <= -80)
].copy()

stationary_wa = stationary_df[
    (stationary_df["core_lat"] >= 0) &
    (stationary_df["core_lat"] <= 15) &
    (stationary_df["core_lon"] >= -20) &
    (stationary_df["core_lon"] <= 0)
].copy()

stationary_pwp = stationary_df[
    (stationary_df["core_lat"] >= PWP_EXTENT[2]) &
    (stationary_df["core_lat"] <= PWP_EXTENT[3]) &
    (stationary_df["core_lon"] >= PWP_EXTENT[0]) &
    (stationary_df["core_lon"] <= PWP_EXTENT[1])
].copy()

REGIONS = ["ALL", "Arabian", "Bengal", "Equatorial"]

REGION_BOUNDS = {
    "Arabian": {
        "lon_min": 50,
        "lon_max": 78,
        "lat_min": 5,
        "lat_max": 25,
    },
    "Bengal": {
        "lon_min": 78,
        "lon_max": 100,
        "lat_min": 5,
        "lat_max": 25,
    },
    "Equatorial": {
        "lon_min": 40,
        "lon_max": 100,
        "lat_min": -10,
        "lat_max": 5,
    },
}

stationary_regions = {
    "ALL": stationary_io
}

for region in ["Arabian", "Bengal", "Equatorial"]:

    b = REGION_BOUNDS[region]

    stationary_regions[region] = stationary_io[
        (stationary_io["core_lon"] >= b["lon_min"]) &
        (stationary_io["core_lon"] <= b["lon_max"]) &
        (stationary_io["core_lat"] >= b["lat_min"]) &
        (stationary_io["core_lat"] <= b["lat_max"])
    ].copy()
    


def build_density(
        df,
        npz,
        key_y,
        key_x):

    density = np.zeros(
        (Ny, Nx),
        dtype=np.uint32
    )

    for eid in df["event_id"]:

        ky = key_y.format(eid=eid)
        kx = key_x.format(eid=eid)

        if ky not in npz:
            continue

        ys = npz[ky]
        xs = npz[kx]

        density[ys, xs] += 1

    return density
def add_region_boxes(ax):

    REGIONS = [
        IO_EXTENT,
        # CP_EXTENT,
        # EP_EXTENT,
        # WA_EXTENT,
        # PWP_EXTENT
    ]

    for extent in REGIONS:

        xmin, xmax, ymin, ymax = extent

        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=20
        )

        ax.add_patch(rect)
def plot_total(
        density,
        title, outfile,
        extent=None):

    proj = ccrs.PlateCarree()

    fig = plt.figure(
        figsize=(13,4.8),
        dpi=DPI
    )

    ax = plt.axes(
        projection=proj
    )

    if extent is None:
        ax.set_extent([-180,180,-60,60], crs=proj)
    else:
        ax.set_extent(
            extent,
            crs=proj
        )

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.85"
    )

    ax.coastlines()
    if extent is None:
        add_region_boxes(ax)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        alpha=0.4
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

    masked = np.ma.masked_where(
        density[:, sort_idx] == 0,
        density[:, sort_idx]
    )
    vals = masked.compressed()

    vmax = int(np.ceil(np.percentile(vals, 99)))
    print(f"99th percentile: {np.percentile(vals, 99):.1f}")
    print(f"Maximum: {vals.max()}")

    # -------------------------------------------------
    # Continuous colorbar for very low lifetime threshold
    # -------------------------------------------------
    
    if MIN_LIFETIME <= 1:
    
        norm = Normalize(
            vmin=1,
            vmax=vmax
        )
    
        bounds = None
    
    # -------------------------------------------------
    # Discrete colorbar for longer lifetimes
    # -------------------------------------------------
    
    else:
    
        bounds = np.arange(1, vmax + 2)
    
        norm = BoundaryNorm(
            bounds,
            ncolors=plt.get_cmap(CMAP).N,
            clip=True
        )
    
    im = ax.pcolormesh(
        lons_wrapped,
        lats,
        masked,
        shading="auto",
        cmap=CMAP,
        norm=norm,
        transform=proj
    )

    cax = fig.add_axes([
        0.28,   # left
        0.03,  # bottom
        0.44,   # width
        0.022   # height
    ])

    if MIN_LIFETIME <= 1:

        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal"
        )
    
    else:
    
        cb = fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            ticks=bounds
        )
    
    cb.ax.tick_params(
        labelsize=13,
        length=4,
        width=1
    )


    cb.set_label(
        "Event Count",
        fontsize=13,
        fontweight="bold",
        labelpad=2
    )
    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=4
    )

    plt.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight"
    )
    
    plt.close(fig)
    
    print(
        f"[SAVE] {outfile}"
    )
    
def plot_monthly(
        df,
        npz,
        key_y,
        key_x,
        title,
        outfile,
        extent=None):

    # =====================================================
    # Build monthly density maps
    # =====================================================

    monthly = {}

    all_values = []

    for month in range(1,13):

        subset = df[
            df["month"] == month
        ]

        density = build_density(
            subset,
            npz,
            key_y,
            key_x
        )

        monthly[month] = density
               
        vals = density[
            density > 0
        ]

        if len(vals):

            all_values.extend(vals)

    vmax = int(
        np.ceil(
            np.percentile(
                all_values,
                99
            )
        )
    )

    if MIN_LIFETIME <= 1:

        norm = Normalize(
            vmin=1,
            vmax=vmax
        )

        bounds = None

    else:

        bounds = np.arange(
            1,
            vmax + 2
        )

        norm = BoundaryNorm(
            bounds,
            ncolors=plt.get_cmap(CMAP).N,
            clip=True
        )

    # =====================================================
    # Figure
    # =====================================================

    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(

        3,

        4,

        figsize=(21,12),

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

        "Jan","Feb","Mar","Apr",

        "May","Jun","Jul","Aug",

        "Sep","Oct","Nov","Dec"

    ]
    # =====================================================
    # DRAW MAPS
    # =====================================================
    
    from cartopy.mpl.gridliner import (
        LONGITUDE_FORMATTER,
        LATITUDE_FORMATTER
    )
    
    for month in range(12):
    
        ax = axes[month]
    
        # ----------------------------------------------
        # Extent
        # ----------------------------------------------
    
        if extent is None:
    
            ax.set_extent(
                [-180,180,-60,60],
                crs=proj
            )
    
        else:
    
            ax.set_extent(
                extent,
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
            "size":13,
            "weight":"bold"
        }
    
        gl.ylabel_style = {
            "size":13,
            "weight":"bold"
        }
    
        gl.top_labels = False
        gl.right_labels = False
    
        # only left column
    
        if month not in [0,4,8]:
            gl.left_labels = False
    
        # only bottom row
    
        if month < 8:
            gl.bottom_labels = False
    
        # ----------------------------------------------
        # Density
        # ----------------------------------------------
    
        masked = np.ma.masked_where(
    
            monthly[month+1][:,sort_idx] == 0,
    
            monthly[month+1][:,sort_idx]
    
        )
    
        mesh = ax.pcolormesh(
    
            lons_wrapped,
    
            lats,
    
            masked,
    
            shading="auto",
    
            cmap=CMAP,
    
            norm=norm,
    
            transform=proj
    
        )
    
        # ----------------------------------------------
        # Title
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
        
        shrink=0.72,
        
        pad=0.045,
        
        aspect=55,
        
        fraction=0.030
        
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
    # TITLE
    # =====================================================
        
    if title != "":
        
        fig.suptitle(
        
            title,
        
            fontsize=16,
        
            fontweight="bold"
        
        )
        
    # =====================================================
    # SAVE
    # =====================================================
        
    plt.savefig(
        
        outfile,
        
        dpi=DPI,
        
        bbox_inches="tight",
        
        facecolor="white"
        
    )
        
    plt.close(fig)

def plot_monthly_core_area_comparison(
        regions,
        outfile):

    months = [
        "Jan","Feb","Mar","Apr",
        "May","Jun","Jul","Aug",
        "Sep","Oct","Nov","Dec"
    ]

    fig, ax = plt.subplots(
        figsize=(10,5),
        dpi=DPI
    )

    for label, df in regions.items():
        monthly_area = []

        for month in range(1, 13):
        
            subset = df[df["month"] == month]
        
            area = 0.0
        
            for _, row in subset.iterrows():
        
                lat = row["core_lat"]
        
                pixel_area = (
                    0.25 * 111.32 *
                    0.25 * 111.32 *
                    np.cos(np.deg2rad(lat))
                )
        
                area += row["core_pixels"] * pixel_area
        
            monthly_area.append(area)
        
        monthly_area = np.array(monthly_area)
        
        ax.plot(
            months,
            monthly_area,
            marker="o",
            linewidth=2,
            label=label
        )

    ax.set_ylabel(
        "Total Area (km²)",
        fontsize=12,
        fontweight="bold"
    )
    
    ax.set_title(
        "Monthly Total Area",
        fontsize=16,
        fontweight="bold"
    )
    
    ax.grid(
        alpha=0.3,
        linestyle="--"
    )
    
    ax.legend(
        frameon=False,
        fontsize=8
    )
    
    ax.tick_params(
        axis="x",
        rotation=30
    )
    
    ax.tick_params(
        axis="y",
        labelsize=8
    )

    plt.tight_layout(rect=[0,0,1,0.95])

    plt.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVE] {outfile}"
    )
    
def plot_monthly_area_bar(
        df,
        region_name,
        outfile):

    months = [
        "Jan","Feb","Mar","Apr",
        "May","Jun","Jul","Aug",
        "Sep","Oct","Nov","Dec"
    ]

    monthly_area = []

    for month in range(1, 13):

        subset = df[df["month"] == month]

        area = 0.0

        for _, row in subset.iterrows():

            lat = row["core_lat"]

            pixel_area = (
                0.25 * 111.32 *
                0.25 * 111.32 *
                np.cos(np.deg2rad(lat))
            )

            area += row["core_pixels"] * pixel_area

        monthly_area.append(area)

    plt.figure(
        figsize=(8,4),
        dpi=DPI
    )

    plt.bar(
        months,
        monthly_area,
        edgecolor="black"
    )

    plt.ylabel(
        "Total Calm-Patch Area (km²)",
        fontsize=13,
        fontweight="bold",
        color="black"
    )

    plt.xlabel(
        "Month",
        fontsize=13,
        fontweight="bold",
        color="black"
    )

    plt.title(
        # f"{region_name}\n{LIFETIME_LABEL}",
        "",
        fontsize=13,
        fontweight="bold"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
        linestyle="--"
    )
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVE] {outfile}")
    
def calculate_monthly_relative_area(
        df,
        region_name,
        extent):

    months = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    # =====================================================
    # DOMAIN AREA
    # Ocean area inside region, excluding land
    # =====================================================

    lon_min, lon_max, lat_min, lat_max = extent

    lon_wrapped = ((lons + 180) % 360) - 180

    lat_sel = (
        (lats >= lat_min) &
        (lats <= lat_max)
    )

    lon_sel = (
        (lon_wrapped >= lon_min) &
        (lon_wrapped <= lon_max)
    )

    domain_area = 0.0

    for iy in np.where(lat_sel)[0]:

        lat = lats[iy]

        pixel_area = (
            0.25 * 111.32 *
            0.25 * 111.32 *
            np.cos(np.deg2rad(lat))
        )

        for ix in np.where(lon_sel)[0]:

            if land_mask[iy, ix] == 0:
                domain_area += pixel_area

    print(
        f"[INFO] {region_name} ocean domain area = "
        f"{domain_area:,.0f} km²"
    )

    # =====================================================
    # EVENT AREA
    # =====================================================

    work = df.copy()

    work["event_area_km2"] = (
        work["core_pixels"] *
        (0.25 * 111.32) *
        (0.25 * 111.32) *
        np.cos(np.deg2rad(work["core_lat"]))
    )

    # =====================================================
    # RECONSTRUCT EVERY PASS DURING EACH EVENT
    # =====================================================

    pass_records = []

    for _, row in work.iterrows():

        start_text = row["start_stamp"]
        end_text = row["end_stamp"]

        start_date = pd.to_datetime(
            start_text[:10]
        )

        end_date = pd.to_datetime(
            end_text[:10]
        )

        start_pass = (
            "Day"
            if "Day" in start_text
            else "Night"
        )

        end_pass = (
            "Day"
            if "Day" in end_text
            else "Night"
        )

        start_time = (
            start_date +
            pd.Timedelta(
                hours=0 if start_pass == "Night" else 12
            )
        )

        end_time = (
            end_date +
            pd.Timedelta(
                hours=0 if end_pass == "Night" else 12
            )
        )

        active_passes = pd.date_range(
            start=start_time,
            end=end_time,
            freq="12h"
        )

        for pass_time in active_passes:

            pass_records.append({
                "datetime": pass_time,
                "area_km2": row["event_area_km2"]
            })

    pass_df = pd.DataFrame(pass_records)

    # =====================================================
    # TOTAL PATCH AREA PER PASS
    # =====================================================

    pass_totals = (
        pass_df
        .groupby("datetime")["area_km2"]
        .sum()
    )

    # =====================================================
    # COMPLETE PASS TIMELINE
    # Missing passes contribute zero area
    # =====================================================

    first_date = pd.to_datetime(
        stationary_df["start_stamp"].str[:10]
    ).min()

    last_date = pd.to_datetime(
        stationary_df["end_stamp"].str[:10]
    ).max()

    full_pass_index = pd.date_range(
        start=first_date.normalize(),
        end=last_date.normalize() + pd.Timedelta(hours=12),
        freq="12h"
    )

    pass_totals = pass_totals.reindex(
        full_pass_index,
        fill_value=0.0
    )

    # =====================================================
    # DAILY MEAN AREA
    # Mean of Day + Night
    # =====================================================

    daily_area = (
        pass_totals
        .groupby(pass_totals.index.normalize())
        .mean()
    )

    # =====================================================
    # MONTHLY CLIMATOLOGY
    # =====================================================

    monthly_relative_area = []

    for month in range(1, 13):

        month_daily = daily_area[
            daily_area.index.month == month
        ]

        mean_daily_area = month_daily.mean()

        relative_area = (
            100.0 *
            mean_daily_area /
            domain_area
        )

        monthly_relative_area.append(
            relative_area
        )

        print(
            f"[INFO] {months[month-1]} | "
            f"Days = {len(month_daily)} | "
            f"Mean daily area = "
            f"{mean_daily_area:,.0f} km² | "
            f"Coverage = "
            f"{relative_area:.2f}%"
        )

    return np.asarray(monthly_relative_area)
def plot_monthly_relative_area_bar(
        df,
        region_name,
        extent,
        outfile):

    months = [
        "Jan","Feb","Mar","Apr",
        "May","Jun","Jul","Aug",
        "Sep","Oct","Nov","Dec"
    ]

    # =====================================================
    # DOMAIN AREA
    # Ocean area inside region, excluding land
    # =====================================================

    lon_min, lon_max, lat_min, lat_max = extent

    lon_wrapped = ((lons + 180) % 360) - 180

    lat_sel = (
        (lats >= lat_min) &
        (lats <= lat_max)
    )

    lon_sel = (
        (lon_wrapped >= lon_min) &
        (lon_wrapped <= lon_max)
    )

    domain_area = 0.0

    for iy in np.where(lat_sel)[0]:

        lat = lats[iy]

        pixel_area = (
            0.25 * 111.32 *
            0.25 * 111.32 *
            np.cos(np.deg2rad(lat))
        )

        for ix in np.where(lon_sel)[0]:

            if land_mask[iy, ix] == 0:
                domain_area += pixel_area

    print(
        f"[INFO] {region_name} ocean domain area = "
        f"{domain_area:,.0f} km²"
    )

    # =====================================================
    # EVENT AREA
    # =====================================================

    work = df.copy()

    work["event_area_km2"] = (
        work["core_pixels"] *
        (0.25 * 111.32) *
        (0.25 * 111.32) *
        np.cos(np.deg2rad(work["core_lat"]))
    )

    # =====================================================
    # RECONSTRUCT EVERY PASS DURING EACH EVENT
    # =====================================================

    pass_records = []

    for _, row in work.iterrows():

        start_text = row["start_stamp"]
        end_text = row["end_stamp"]

        start_date = pd.to_datetime(
            start_text[:10]
        )

        end_date = pd.to_datetime(
            end_text[:10]
        )

        start_pass = (
            "Day"
            if "Day" in start_text
            else "Night"
        )

        end_pass = (
            "Day"
            if "Day" in end_text
            else "Night"
        )

        # -------------------------------------------------
        # Convert date + pass to a 12-hour time index
        #
        # Generator order:
        # Night -> Day -> Night -> Day ...
        # -------------------------------------------------

        start_time = (
            start_date +
            pd.Timedelta(
                hours=0 if start_pass == "Night" else 12
            )
        )

        end_time = (
            end_date +
            pd.Timedelta(
                hours=0 if end_pass == "Night" else 12
            )
        )

        active_passes = pd.date_range(
            start=start_time,
            end=end_time,
            freq="12h"
        )

        for pass_time in active_passes:

            pass_records.append({
                "datetime": pass_time,
                "date": pass_time.normalize(),
                "month": pass_time.month,
                "area_km2": row["event_area_km2"]
            })

    pass_df = pd.DataFrame(pass_records)

    # =====================================================
    # TOTAL PATCH AREA PER PASS
    # =====================================================

    pass_totals = (
        pass_df
        .groupby("datetime")["area_km2"]
        .sum()
    )

    # =====================================================
    # COMPLETE PASS TIMELINE
    #
    # This is important:
    # passes with ZERO qualifying events must still
    # contribute zero area to the monthly average.
    # =====================================================

    first_date = pd.to_datetime(
        stationary_df["start_stamp"].str[:10]
    ).min()

    last_date = pd.to_datetime(
        stationary_df["end_stamp"].str[:10]
    ).max()

    full_pass_index = pd.date_range(
        start=first_date.normalize(),
        end=last_date.normalize() + pd.Timedelta(hours=12),
        freq="12h"
    )

    pass_totals = pass_totals.reindex(
        full_pass_index,
        fill_value=0.0
    )

    # =====================================================
    # DAILY MEAN AREA
    #
    # Average Day + Night for each day.
    # =====================================================

    daily_area = (
        pass_totals
        .groupby(pass_totals.index.normalize())
        .mean()
    )

    # =====================================================
    # MONTHLY CLIMATOLOGICAL DAILY MEAN
    #
    # Example:
    # mean of ALL January days from ALL years.
    # =====================================================

    monthly_relative_area = []
    monthly_mean_area = []

    for month in range(1, 13):

        month_daily = daily_area[
            daily_area.index.month == month
        ]

        mean_daily_area = month_daily.mean()

        relative_area = (
            100.0 *
            mean_daily_area /
            domain_area
        )

        monthly_mean_area.append(
            mean_daily_area
        )

        monthly_relative_area.append(
            relative_area
        )

        print(
            f"[INFO] {months[month-1]} | "
            f"Days = {len(month_daily)} | "
            f"Mean daily area = "
            f"{mean_daily_area:,.0f} km² | "
            f"Coverage = "
            f"{relative_area:.2f}%"
        )

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(
        figsize=(21,3.2),
        dpi=DPI
    )

    plt.plot(
        months,
        monthly_relative_area,
        marker="o",
        linewidth=2,
        markersize=7
    )

    plt.ylabel(
        "Ocean Coverage(%)",
        fontsize=16,
        fontweight="bold",
    )

    plt.title(
        "",
        fontsize=13,
        fontweight="bold"
    )

    plt.grid(
        axis="y",
        alpha=0.2,
        linestyle="--",
        linewidth=0.8
    )

    plt.xticks(
        fontsize=16,
        fontweight="bold"
    )

    plt.yticks(
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=DPI,
        facecolor="white"
    )

    plt.close()

    print(f"[SAVE] {outfile}")

def plot_mean_patch_size(
        regions,
        outfile):

    labels = []
    mean_sizes = []

    for name, df in regions.items():

        labels.append(name)

        areas = []

        for _, row in df.iterrows():

            pixel_area = (
                0.25 * 111.32 *
                0.25 * 111.32 *
                np.cos(np.deg2rad(row["core_lat"]))
            )

            areas.append(
                row["core_pixels"] *
                pixel_area
            )

        mean_sizes.append(np.mean(areas))

    plt.figure(figsize=(8,5), dpi=DPI)

    plt.bar(labels, mean_sizes)

    plt.ylabel(
        "Mean Patch Area (km²)",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Mean Patch Size",
        fontsize=15,
        fontweight="bold"
    )

    plt.xticks(rotation=20)

    plt.grid(
        axis="y",
        alpha=0.3,
        linestyle="--"
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVE] {outfile}")
def plot_total_events(
        regions,
        outfile):

    labels = []
    counts = []

    for name, df in regions.items():

        labels.append(name)
        counts.append(len(df))

    plt.figure(figsize=(8,5), dpi=DPI)

    plt.bar(labels, counts)

    plt.ylabel("Number of Events",
               fontsize=12,
               fontweight="bold")

    plt.title(
        f"Total Events ({LIFETIME_LABEL})",
        fontsize=15,
        fontweight="bold"
    )

    plt.xticks(rotation=20)

    plt.grid(
        axis="y",
        alpha=0.3,
        linestyle="--"
    )

    plt.tight_layout()

    plt.savefig(outfile,
                dpi=DPI,
                bbox_inches="tight")

    plt.close()

    print(f"[SAVE] {outfile}")
def plot_monthly_region_fraction(
        regions,
        extents,
        outfile):

    months = [
        "Jan","Feb","Mar","Apr",
        "May","Jun","Jul","Aug",
        "Sep","Oct","Nov","Dec"
    ]

    plt.figure(figsize=(10,5), dpi=DPI)

    for name, df in regions.items():

        lon1, lon2, lat1, lat2 = extents[name]

        mean_lat = 0.5 * (lat1 + lat2)

        width = (lon2-lon1) * 111.32 * np.cos(np.deg2rad(mean_lat))
        height = (lat2-lat1) * 111.32

        region_area = width * height

        monthly_fraction = []

        for month in range(1,13):

            subset = df[df["month"] == month]

            total_area = 0

            for _, row in subset.iterrows():

                pixel_area = (
                    0.25 * 111.32 *
                    0.25 * 111.32 *
                    np.cos(np.deg2rad(row["core_lat"]))
                )

                total_area += row["core_pixels"] * pixel_area

            monthly_fraction.append(
                100 * total_area / region_area
            )

        plt.plot(
            months,
            monthly_fraction,
            marker="o",
            linewidth=2,
            label=name
        )

    plt.ylabel("Occupied Area (%)",
               fontsize=12,
               fontweight="bold")

    plt.title(
        "Monthly Fraction of Region Occupied",
        fontsize=15,
        fontweight="bold"
    )

    plt.grid(alpha=0.3, linestyle="--")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(outfile, dpi=DPI, bbox_inches="tight")

    plt.close()

    print(f"[SAVE] {outfile}")
def plot_monthly_relative_area_timeseries(
        df,
        region_name,
        extent,
        outfile):

    # =====================================================
    # DOMAIN AREA
    # Ocean area inside region, excluding land
    # =====================================================

    lon_min, lon_max, lat_min, lat_max = extent

    lon_wrapped = ((lons + 180) % 360) - 180

    lat_sel = (
        (lats >= lat_min) &
        (lats <= lat_max)
    )

    lon_sel = (
        (lon_wrapped >= lon_min) &
        (lon_wrapped <= lon_max)
    )

    domain_area = 0.0

    for iy in np.where(lat_sel)[0]:

        lat = lats[iy]

        pixel_area = (
            0.25 * 111.32 *
            0.25 * 111.32 *
            np.cos(np.deg2rad(lat))
        )

        for ix in np.where(lon_sel)[0]:

            if land_mask[iy, ix] == 0:
                domain_area += pixel_area

    print(
        f"[INFO] {region_name} ocean domain area = "
        f"{domain_area:,.0f} km²"
    )

    # =====================================================
    # EVENT AREA
    # =====================================================

    work = df.copy()

    work["event_area_km2"] = (
        work["core_pixels"] *
        (0.25 * 111.32) *
        (0.25 * 111.32) *
        np.cos(np.deg2rad(work["core_lat"]))
    )

    # =====================================================
    # RECONSTRUCT EVERY ACTIVE PASS
    # =====================================================

    pass_records = []

    for _, row in work.iterrows():

        start_text = row["start_stamp"]
        end_text = row["end_stamp"]

        start_date = pd.to_datetime(
            start_text[:10]
        )

        end_date = pd.to_datetime(
            end_text[:10]
        )

        start_pass = (
            "Day"
            if "Day" in start_text
            else "Night"
        )

        end_pass = (
            "Day"
            if "Day" in end_text
            else "Night"
        )

        start_time = (
            start_date +
            pd.Timedelta(
                hours=0 if start_pass == "Night" else 12
            )
        )

        end_time = (
            end_date +
            pd.Timedelta(
                hours=0 if end_pass == "Night" else 12
            )
        )

        active_passes = pd.date_range(
            start=start_time,
            end=end_time,
            freq="12h"
        )

        for pass_time in active_passes:

            pass_records.append({
                "datetime": pass_time,
                "area_km2": row["event_area_km2"]
            })

    pass_df = pd.DataFrame(
        pass_records
    )

    # =====================================================
    # TOTAL AREA PER PASS
    # =====================================================

    pass_totals = (
        pass_df
        .groupby("datetime")["area_km2"]
        .sum()
    )

    # =====================================================
    # COMPLETE TIMELINE
    #
    # Missing passes = zero coverage
    # =====================================================

    first_date = pd.to_datetime(
        stationary_df["start_stamp"].str[:10]
    ).min()

    last_date = pd.to_datetime(
        stationary_df["end_stamp"].str[:10]
    ).max()

    full_pass_index = pd.date_range(
        start=first_date.normalize(),
        end=last_date.normalize() + pd.Timedelta(hours=12),
        freq="12h"
    )

    pass_totals = pass_totals.reindex(
        full_pass_index,
        fill_value=0.0
    )

    # =====================================================
    # DAILY MEAN
    #
    # Mean of Day + Night
    # =====================================================

    daily_area = (
        pass_totals
        .groupby(
            pass_totals.index.normalize()
        )
        .mean()
    )

    # =====================================================
    # MONTHLY TIME SERIES
    #
    # Mean daily area for EACH individual month
    # =====================================================

    monthly_area = (
        daily_area
        .resample("MS")
        .mean()
    )

    monthly_coverage = (
        100.0 *
        monthly_area /
        domain_area
    )

    # =====================================================
    # PRINT BASIC INFORMATION
    # =====================================================

    print(
        f"[INFO] Time series: "
        f"{monthly_coverage.index.min():%Y-%m} "
        f"to "
        f"{monthly_coverage.index.max():%Y-%m}"
    )

    print(
        f"[INFO] Mean coverage = "
        f"{monthly_coverage.mean():.2f}%"
    )

    print(
        f"[INFO] Maximum coverage = "
        f"{monthly_coverage.max():.2f}%"
    )

    # =====================================================
    # PLOT
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(14,5),
        dpi=DPI
    )

    ax.plot(
        monthly_coverage.index,
        monthly_coverage.values,
        linewidth=2
    )

    ax.set_ylabel(
        "Mean Daily Calm-Patch Coverage (%)",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_title(
        "",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        alpha=0.3,
        linestyle="--"
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVE] {outfile}"
    )
def plot_figure3_patch_characteristics(
        df,
        outfile):

    # =====================================================
    # PANEL (a): LIFETIME DISTRIBUTION
    # =====================================================

    lifetime = (
        df["lifetime_passes"]
        .astype(int)
        .values
    )

    lifetime_values = np.arange(
        lifetime.min(),
        lifetime.max() + 1
    )

    lifetime_counts = np.array([
        np.sum(lifetime == value)
        for value in lifetime_values
    ])

    # =====================================================
    # PANEL (b): PATCH AREA ECDF
    # =====================================================

    areas = []

    for _, row in df.iterrows():

        lat = row["core_lat"]

        pixel_area = (
            0.25 * 111.32 *
            0.25 * 111.32 *
            np.cos(np.deg2rad(lat))
        )

        patch_area = (
            row["core_pixels"] *
            pixel_area
        )

        areas.append(patch_area)

    areas = np.asarray(
        areas,
        dtype=float
    )

    areas = areas[
        np.isfinite(areas) &
        (areas > 0)
    ]

    areas_sorted = np.sort(areas)

    cumulative = (
        np.arange(1, len(areas_sorted) + 1)
        / len(areas_sorted)
        * 100.0
    )

    q25_area = np.percentile(
        areas_sorted,
        25
    )

    median_area = np.percentile(
        areas_sorted,
        50
    )

    q75_area = np.percentile(
        areas_sorted,
        75
    )

    # =====================================================
    # FIGURE
    # =====================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        dpi=DPI
    )

    ax1, ax2 = axes

    # =====================================================
    # PANEL (a): LIFETIME
    # =====================================================

    ax1.bar(
        lifetime_values,
        lifetime_counts,
        width=0.85,
        edgecolor="black",
        linewidth=0.8
    )

    ax1.set_yscale("log")

    ax1.set_xlabel(
        "Lifetime (passes)",
        fontsize=13,
        fontweight="bold"
    )

    ax1.set_ylabel(
        "Number of Events",
        fontsize=13,
        fontweight="bold"
    )

    ax1.set_xticks(
        lifetime_values
    )

    ax1.tick_params(
        axis="both",
        labelsize=11
    )

    for tick in ax1.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax1.get_yticklabels():
        tick.set_fontweight("bold")

    ax1.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.25
    )

    ax1.set_axisbelow(True)

    # =====================================================
    # PANEL LABEL (a)
    # =====================================================

    ax1.text(
        0.0,
        1.06,
        "(a)",
        transform=ax1.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    # =====================================================
    # PANEL (b): ECDF
    # =====================================================

    ax2.plot(
        areas_sorted,
        cumulative,
        linewidth=2
    )

    # Q25
    ax2.axvline(
        q25_area,
        linestyle=":",
        linewidth=1.5,
        color="gray",
        label=f"Q25 = {q25_area:,.0f} km²"
    )

    # Median
    ax2.axvline(
        median_area,
        linestyle="--",
        linewidth=2.2,
        color="black",
        label=f"Median = {median_area:,.0f} km²"
    )

    # Q75
    ax2.axvline(
        q75_area,
        linestyle=":",
        linewidth=1.5,
        color="gray",
        label=f"Q75 = {q75_area:,.0f} km²"
    )

    for percentile in [25, 50, 75]:

        ax2.axhline(
            percentile,
            linestyle=":",
            linewidth=1,
            color="gray",
            alpha=0.55
        )

    ax2.set_xscale("log")

    ax2.set_ylim(
        0,
        100
    )

    ax2.set_xlabel(
        "Patch Area (km²)",
        fontsize=13,
        fontweight="bold"
    )

    ax2.set_ylabel(
        "Cumulative Fraction of Patches (%)",
        fontsize=13,
        fontweight="bold"
    )

    ax2.tick_params(
        axis="both",
        labelsize=11
    )

    for tick in ax2.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax2.get_yticklabels():
        tick.set_fontweight("bold")

    ax2.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.25
    )

    ax2.set_axisbelow(True)

    ax2.legend(
        frameon=False,
        fontsize=10,
        loc="lower right"
    )

    # =====================================================
    # PANEL LABEL (b)
    # =====================================================

    ax2.text(
        0.0,
        1.06,
        "(b)",
        transform=ax2.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    # =====================================================
    # LAYOUT
    # =====================================================

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.17,
        top=0.88,
        wspace=0.28
    )

    # =====================================================
    # SAVE
    # =====================================================

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
# STATIONARY
# =====================================================

stationary_global = build_density(
    stationary_df,
    stationary_npz,
    "event_{eid}_core_y",
    "event_{eid}_core_x"
)

# plot_total(
#     stationary_global,
#     # "Global Distribution of AMSR2 Calm-Patch Events (2012–2023)",
#     "",
#     os.path.join(
#         PLOT_DIR,
#         f"Stationary_Global_Total_K{LIFETIME_LABEL}.png"
#     )
# )

# plot_monthly(
#     stationary_io,
#     stationary_npz,
#     "event_{eid}_core_y",
#     "event_{eid}_core_x",
#     "",
#     os.path.join(
#         PLOT_DIR,
#         f"Stationary_IO_Monthly_K{LIFETIME_LABEL}.png"
#     ),
#     extent=IO_EXTENT
# )

# plot_monthly_relative_area_bar(
#     stationary_io,
#     "Indian Ocean",
#     IO_EXTENT,
#     os.path.join(
#         PLOT_DIR,
#         "MonthlyRelativeArea_IndianOcean.png"
#     )
# )

# plot_figure3_patch_characteristics(
#     stationary_io,
#     os.path.join(
#         PLOT_DIR,
#         "Figure3_AMSR2_PatchCharacteristics.png"
#     )
# )





# plot_monthly_relative_area_timeseries(
#     stationary_io,
#     "Indian Ocean",
#     IO_EXTENT,
#     os.path.join(
#         PLOT_DIR,
#         "MonthlyRelativeArea_TimeSeries_IndianOcean.png"
#     )
# )



# plot_total(
#     stationary_global,
#     f"Patch Density Indian Ocean | {LIFETIME_LABEL}",
#     os.path.join(
#         PLOT_DIR,
#         f"Stationary_IO_Total_K{LIFETIME_LABEL}.png"
#     ),
#     extent=IO_EXTENT
# )



# # =====================================================
# # STATIONARY MONTHLY METRICS
# # =====================================================
# plot_monthly_core_area_comparison(
#     {
#         "Indian Ocean": stationary_io,
#         "Central America": stationary_cp,
#         "Equatorial Pacific": stationary_ep,
#         "West Africa": stationary_wa,
#         "Pacific Warm Pool": stationary_pwp
#     },
#     os.path.join(
#         PLOT_DIR,
#         f"Monthly_CoreArea_Comparison_K{LIFETIME_LABEL}.png"
#     )
# )



# plot_monthly_area_bar(
#     stationary_io,
#     "Indian Ocean",
#     os.path.join(
#         PLOT_DIR,
#         "MonthlyArea_IndianOcean.png"
#     )
# )

# plot_monthly_area_bar(
#     stationary_cp,
#     "Central Pacific",
#     os.path.join(
#         PLOT_DIR,
#         "MonthlyArea_CentralPacific.png"
#     )
# )

# plot_monthly_area_bar(
#     stationary_ep,
#     "Equatorial Pacific",
#     os.path.join(
#         PLOT_DIR,
#         "MonthlyArea_EquatorialPacific.png"
#     )
# )

# plot_monthly_area_bar(
#     stationary_wa,
#     "West Africa",
#     os.path.join(
#         PLOT_DIR,
#         "MonthlyArea_WestAfrica.png"
#     )
# )
# for region, df in stationary_regions.items():

#     plot_monthly_area_bar(
#         df,
#         region,
#         os.path.join(
#             PLOT_DIR,
#             f"MonthlyArea_{region}.png"
#         )
#     )
# REGIONS = {
#     "Indian Ocean": stationary_io,
#     "Central America": stationary_cp,
#     "Equatorial Pacific": stationary_ep,
#     "West Africa": stationary_wa,
#     "Pacific Warm Pool": stationary_pwp
# }

# EXTENTS = {
#     "Indian Ocean": IO_EXTENT,
#     "Central America": CP_EXTENT,
#     "Equatorial Pacific": EP_EXTENT,
#     "West Africa": WA_EXTENT,
#     "Pacific Warm Pool": PWP_EXTENT
# }

# plot_monthly_region_fraction(
#     REGIONS,
#     EXTENTS,
#     os.path.join(
#         PLOT_DIR,
#         f"Monthly_RegionFraction_K{LIFETIME_LABEL}.png"
#     )
# )

# plot_total_events(
#     REGIONS,
#     os.path.join(
#         PLOT_DIR,
#         f"TotalEvents_K{LIFETIME_LABEL}.png"
#     )
# )

# plot_mean_patch_size(
#     REGIONS,
#     os.path.join(
#         PLOT_DIR,
#         f"MeanPatchSize_K{LIFETIME_LABEL}.png"
#     )
# )