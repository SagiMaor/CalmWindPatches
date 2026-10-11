import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================================
# USER SETTINGS
# ==========================================================

NC_FILE = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\ClimChange\SLP_Climatology1.nc"
)

OUT_DIR = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\SavedData\SLP"
)

# Create the folder if it doesn't already exist
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DPI = 300

# ----------------------------------------------------------
# Wind plotting
# ----------------------------------------------------------

QUIVER_STRIDE = 5
QUIVER_SCALE = 230
QUIVER_WIDTH = 0.0032


# ----------------------------------------------------------
# Shading
# ----------------------------------------------------------

CMAP = "RdBu_r"

# ----------------------------------------------------------
# Month names
# ----------------------------------------------------------

MONTH_NAMES = [
    "Jan","Feb","Mar","Apr",
    "May","Jun","Jul","Aug",
    "Sep","Oct","Nov","Dec"
]

# ==========================================================
# LOAD NETCDF
# ==========================================================

print("Opening climatology...")

ds = xr.open_dataset(NC_FILE,engine="netcdf4")

slp = ds["SLP_mean"].values
u10 = ds["U10_mean"].values
v10 = ds["V10_mean"].values
lat = ds["lat"].values
lon = ds["lon"].values

print("Done.")

# ==========================================================
# COLOR LEVELS
# ==========================================================

vmin = np.floor(np.nanmin(slp))
vmax = np.ceil(np.nanmax(slp))

fill_levels = np.arange(
    vmin,
    vmax + 0.5,
    0.5
)


# ==========================================================
# BOX DEFINITIONS
# ==========================================================

NORTH_LAT_MIN = 15
NORTH_LAT_MAX = 30

SOUTH_LAT_MIN = -15
SOUTH_LAT_MAX = 0

# Use the complete longitude range of the climatology
BOX_LON_MIN = lon.min()
BOX_LON_MAX = lon.max()


# MAP SETUP
# ==========================================================

def setup_map(ax):

    ax.set_extent(
        [
            lon.min(),
            lon.max(),
            lat.min(),
            lat.max()
        ],
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        linewidth=0.5,
        zorder=5
    )

    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        linewidth=0.35,
        zorder=5
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        linestyle=":",
        color="gray",
        alpha=0.45
    )


    gl.xlabel_style = {"size":13, "weight": "bold"}
    gl.ylabel_style = {"size":13, "weight": "bold"}
    
    gl.top_labels = False
    gl.right_labels = False
    
    gl.xlabel_style = {
        "size": 13,
        "weight": "bold"
    }
    
    gl.ylabel_style = {
        "size": 13,
        "weight": "bold"
    }
    
    return gl
        

# ==========================================================
# QUIVER INDICES
# ==========================================================

ii = slice(None, None, QUIVER_STRIDE)
jj = slice(None, None, QUIVER_STRIDE)

# ==========================================================
# FIGURE
# Monthly Mean SLP  + Winds
# ==========================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(20, 12),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

plt.subplots_adjust(
    left=0.03,
    right=0.99,
    top=0.97,
    bottom=0.12,
    wspace=0.015,
    hspace=0.12
)
axes = axes.ravel()

for month in range(12):

    ax = axes[month]

    gl = setup_map(ax)

    # Only left column gets latitude labels
    if month not in [0, 4, 8]:
        gl.left_labels = False
    
    # Only bottom row gets longitude labels
    if month < 8:
        gl.bottom_labels = False

    # ------------------------------------------------------
    # Filled SLP
    # ------------------------------------------------------

    mesh = ax.contourf(
        lon,
        lat,
        slp[month],
        levels=fill_levels,
        cmap=CMAP,
        extend="both",
        antialiased=True,
        transform=ccrs.PlateCarree(),
        zorder=0
    )

    
    # ------------------------------------------------------
    # Show SLP averaging regions on January panel only
    # ------------------------------------------------------
    
    if month == 0:
    
        north_box = Rectangle(
            (BOX_LON_MIN, NORTH_LAT_MIN),
            BOX_LON_MAX - BOX_LON_MIN,
            NORTH_LAT_MAX - NORTH_LAT_MIN,
            facecolor="red",
            edgecolor="darkred",
            alpha=0.12,
            linewidth=2.0,
            transform=ccrs.PlateCarree(),
            zorder=4
        )
    
        south_box = Rectangle(
            (BOX_LON_MIN, SOUTH_LAT_MIN),
            BOX_LON_MAX - BOX_LON_MIN,
            SOUTH_LAT_MAX - SOUTH_LAT_MIN,
            facecolor="blue",
            edgecolor="darkblue",
            alpha=0.12,
            linewidth=2.0,
            transform=ccrs.PlateCarree(),
            zorder=4
        )
    
        ax.add_patch(north_box)
        ax.add_patch(south_box)
    
    
    # ------------------------------------------------------
    # Wind vectors — ALL MONTHS
    # ------------------------------------------------------
    
    q = ax.quiver(
        lon[jj],
        lat[ii],
        u10[month][ii, jj],
        v10[month][ii, jj],
        transform=ccrs.PlateCarree(),
        scale=QUIVER_SCALE,
        width=QUIVER_WIDTH,
        headwidth=3.8,
        headlength=4.8,
        headaxislength=4.5,
        pivot="middle",
        color="black",
        zorder=8
    )
    
    
    # ------------------------------------------------------
    # Month title — ALL MONTHS
    # ------------------------------------------------------
    
    ax.set_title(
        MONTH_NAMES[month],
        fontsize=16,
        pad=6,
        fontweight="bold"
    )

# ==========================================================
# Shared colorbar
# ==========================================================

cbar = fig.colorbar(
    mesh,
    ax=axes,
    orientation="horizontal",
    shrink=0.65,
    pad=0.045,
    aspect=50,
    fraction=0.035
)

cbar.set_label(
    "Sea Level Pressure (hPa)",
    fontsize=13,
    fontweight="bold"
)

cbar.ax.tick_params(labelsize=10)
for tick in cbar.ax.get_xticklabels():
    tick.set_fontweight("bold")

# ==========================================================
# Quiver key
# ==========================================================

axes[-1].quiverkey(
    q,
    X=0.93,
    Y=-0.18,
    U=2,
    label="2 m s$^{-1}$",
    labelpos="E",
    fontproperties={"size":10}
)

# ==========================================================
# Save
# ==========================================================

outfile = os.path.join(
    OUT_DIR,
    "SLP_Shading_Winds.png"
)

plt.savefig(
    outfile,
    dpi=FIG_DPI,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================================
# FIGURE 8
# Monthly Mean Wind Speed + North-South SLP Difference
# with SLP box map
# ==========================================================


print("Building monthly summary...")


# ==========================================================
# MEAN WIND SPEED
# ==========================================================

wind_speed = np.sqrt(
    u10**2 +
    v10**2
)

monthly_wind = np.nanmean(
    wind_speed,
    axis=(1, 2)
)


# ==========================================================
# NORTH-SOUTH SLP DIFFERENCE
# ==========================================================

north_mask = (
    (lat >= NORTH_LAT_MIN)
    &
    (lat <= NORTH_LAT_MAX)
)

south_mask = (
    (lat >= SOUTH_LAT_MIN)
    &
    (lat <= SOUTH_LAT_MAX)
)


# Mean SLP in northern box
north_slp = np.nanmean(
    slp[:, north_mask, :],
    axis=(1, 2)
)


# Mean SLP in southern box
south_slp = np.nanmean(
    slp[:, south_mask, :],
    axis=(1, 2)
)


# North minus South
monthly_slp = (
    north_slp
    -
    south_slp
)


# ==========================================================
# PRINT VALUES
# ==========================================================

print()
print("North SLP:")
print(north_slp)

print()
print("South SLP:")
print(south_slp)

print()
print("North - South:")
print(monthly_slp)



# ==========================================================
# FIGURE 1
# MONTHLY WIND SPEED + SLP DIFFERENCE
# ==========================================================

months = np.arange(1, 13)

fig, ax1 = plt.subplots(
    figsize=(8, 4.5),
    dpi=FIG_DPI
)


# ==========================================================
# TRANSITION-MONTH SHADING
# March-April and October-November
# ==========================================================

ax1.axvspan(
    2.5,
    4.5,
    color="gray",
    alpha=0.12,
    zorder=0
)

ax1.axvspan(
    9.5,
    11.5,
    color="gray",
    alpha=0.12,
    zorder=0
)


# ==========================================================
# WIND SPEED
# ==========================================================

ax1.plot(
    months,
    monthly_wind,
    "-o",
    color="black",
    linewidth=2.5,
    markersize=6,
    zorder=3
)

ax1.set_ylabel(
    "Mean Wind Speed (m s$^{-1}$)",
    fontsize=12,
    fontweight="bold",
    color="black"
)



ax1.set_xticks(months)

ax1.set_xticklabels(
    MONTH_NAMES,
    fontsize=11,
    fontweight="bold"
)

ax1.tick_params(
    axis="y",
    colors="black",
    labelsize=11
)

for t in ax1.get_yticklabels():
    t.set_fontweight("bold")

ax1.grid(
    axis="y",
    alpha=0.3,
    linestyle="--"
)


# ==========================================================
# SLP DIFFERENCE
# ==========================================================

ax2 = ax1.twinx()

ax2.plot(
    months,
    monthly_slp,
    "-s",
    color="darkred",
    linewidth=2.5,
    markersize=5,
    zorder=4
)

ax2.axhline(
    0,
    color="darkred",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8
)

ax2.set_ylabel(
    "North - South SLP (hPa)",
    fontsize=12,
    fontweight="bold",
    color="darkred"
)

ax2.tick_params(
    axis="y",
    colors="darkred",
    labelsize=11
)

for t in ax2.get_yticklabels():
    t.set_fontweight("bold")


# ==========================================================
# TITLE
# ==========================================================


ax1.set_xlim(
    0.5,
    12.5
)


# ==========================================================
# SAVE FIGURE 1
# ==========================================================

plt.tight_layout()

outfile = os.path.join(
    OUT_DIR,
    "Monthly_Wind_SLP.png"
)

plt.savefig(
    outfile,
    dpi=FIG_DPI,
    bbox_inches="tight"
)

plt.show()
plt.close()

