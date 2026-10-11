import os

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

from scipy.stats import linregress

# ==================================================
# USER SETTINGS
# ==================================================

CALM_THRESHOLD = 1

AMSR_FILE = (
    r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\SavedData\AMSR2_wind_daynight.nc"
)

ERA5_FILE = (
    r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\SavedData\ERA5__SLT_Strips__ClosestSLT__W0p5h__lt3ms.nc"
)
# ERA5_FILE = (
#     r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\SavedData\ERA5_SLT_10MIN.nc"
# )

OUT_DIR = (
    r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\SavedData\Validation_AMSR_vs_ERA5_10min"
)

TEST_MODE = False
TEST_N_DAYS = 365


os.makedirs(
    OUT_DIR,
    exist_ok=True
)

plt.rcParams.update({

    "font.size": 16,

    "axes.titlesize": 20,
    "axes.titleweight": "bold",

    "axes.labelsize": 18,
    "axes.labelweight": "bold",

    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    "legend.fontsize": 14,

    "lines.linewidth": 2.5
})
print("Opening datasets...")

ds_a = xr.open_dataset(
    AMSR_FILE
)

ds_e = xr.open_dataset(
    ERA5_FILE
)

    
amsr_day = ds_a["wind_speed"].isel(**{"pass": 0})

amsr_night = ds_a["wind_speed"].isel(**{"pass": 1})

era5_day = ds_e["wind_asc"]

era5_night = ds_e["wind_desc"]

if TEST_MODE:

    amsr_day = amsr_day.isel(
        time=slice(0, TEST_N_DAYS)
    )

    amsr_night = amsr_night.isel(
        time=slice(0, TEST_N_DAYS)
    )

    era5_day = era5_day.isel(
        time=slice(0, TEST_N_DAYS)
    )

    era5_night = era5_night.isel(
        time=slice(0, TEST_N_DAYS)
    )
    


INDIAN_LAT_MIN = -15
INDIAN_LAT_MAX = 30

INDIAN_LON_MIN = 40
INDIAN_LON_MAX = 110

lat = ds_a.lat.values
lon = ds_a.lon.values

lon2d, lat2d = np.meshgrid(
    lon,
    lat
)

indian_mask = (
    (lat2d >= INDIAN_LAT_MIN)
    &
    (lat2d <= INDIAN_LAT_MAX)
    &
    (lon2d >= INDIAN_LON_MIN)
    &
    (lon2d <= INDIAN_LON_MAX)
)
# ==================================================
# STRIP DOMAIN
# ==================================================

print("Building strip-domain mask...")

strip_domain = np.any(
    np.isfinite(
        era5_day.values
    ),
    axis=0
)

strip_domain |= np.any(
    np.isfinite(
        era5_night.values
    ),
    axis=0
)

comparison_mask = (
    indian_mask
    &
    strip_domain
)
print(
    "Comparison pixels:",
    comparison_mask.sum()
)

print(
    "Fraction of globe:",
    comparison_mask.sum() /
    comparison_mask.size
)

# ==================================================
# COMPRESSED PIXEL INDEX
# ==================================================

mask_idx = np.where(
    comparison_mask
)
print("Loading strip pixels into memory...")

amsr_day_all = amsr_day.values[:, mask_idx[0], mask_idx[1]]
amsr_night_all = amsr_night.values[:, mask_idx[0], mask_idx[1]]

era5_day_all = era5_day.values[:, mask_idx[0], mask_idx[1]]
era5_night_all = era5_night.values[:, mask_idx[0], mask_idx[1]]

print("Done loading.")


# ==================================================
# MONTHLY CALM COUNTS + FRACTIONS
# ==================================================

def monthly_counts(
        amsr_all,
        era5_all,
        times,
        threshold):

    monthly_dates = []

    monthly_amsr_counts = []
    monthly_era5_counts = []

    monthly_amsr_frac = []
    monthly_era5_frac = []

    groups = {}

    for i, t in enumerate(times):

        key = (t.year, t.month)

        groups.setdefault(
            key,
            []
        ).append(i)

    for (yr, mo), inds in groups.items():

        calm_a = 0
        calm_e = 0

        valid_a = 0
        valid_e = 0

        for it in inds:

            amsr = amsr_all[it]

            era5 = era5_all[it]

            valid = (
                np.isfinite(amsr)
                &
                np.isfinite(era5)

            )

            calm_a += np.sum(
                (amsr < threshold)
                &
                valid
            )

            calm_e += np.sum(
                (era5 < threshold)
                &
                valid
            )

            valid_a += np.sum(valid)
            valid_e += np.sum(valid)

        monthly_dates.append(
            pd.Timestamp(
                year=yr,
                month=mo,
                day=15
            )
        )

        monthly_amsr_counts.append(calm_a)
        monthly_era5_counts.append(calm_e)

        monthly_amsr_frac.append(
            calm_a / valid_a
            if valid_a > 0 else np.nan
        )

        monthly_era5_frac.append(
            calm_e / valid_e
            if valid_e > 0 else np.nan
        )

    return (
        np.array(monthly_dates),

        np.array(monthly_amsr_counts),
        np.array(monthly_era5_counts),

        np.array(monthly_amsr_frac),
        np.array(monthly_era5_frac)
    )
# ==================================================
# BUILD MONTHLY SERIES
# ==================================================
times = pd.to_datetime(
    amsr_day.time.values
)
print("Building monthly counts...")
dates_d, amsr_d, era5_d, frac_a_d, frac_e_d = monthly_counts(
    amsr_day_all,
    era5_day_all,
    times,
    CALM_THRESHOLD
)

dates_n, amsr_n, era5_n, frac_a_n, frac_e_n = monthly_counts(
    amsr_night_all,
    era5_night_all,
    times,
    CALM_THRESHOLD
)

monthly_dates = dates_d

monthly_amsr = amsr_d + amsr_n
monthly_era5 = era5_d + era5_n

monthly_frac_amsr = (
    frac_a_d +
    frac_a_n
) / 2

monthly_frac_era5 = (
    frac_e_d +
    frac_e_n
) / 2

# ==================================================
# BUILD DAILY SERIES
# ==================================================

print("Building daily counts...")

daily_dates = times

daily_amsr = []
daily_era5 = []

daily_frac_amsr = []
daily_frac_era5 = []

for it in range(
    amsr_day_all.shape[0]
):

    calm_a = 0
    calm_e = 0

    valid_total = 0

    for amsr_all, era5_all in [

        (amsr_day_all, era5_day_all),

        (amsr_night_all, era5_night_all)

    ]:

        amsr = amsr_all[it]
        era5 = era5_all[it]

        valid = (
            np.isfinite(amsr)
            &
            np.isfinite(era5)
        )

        calm_a += np.sum(
            (amsr < CALM_THRESHOLD)
            &
            valid
        )

        calm_e += np.sum(
            (era5 < CALM_THRESHOLD)
            &
            valid
        )

        valid_total += np.sum(valid)

    daily_amsr.append(calm_a)
    daily_era5.append(calm_e)

    daily_frac_amsr.append(
        calm_a / valid_total
    )

    daily_frac_era5.append(
        calm_e / valid_total
    )

daily_amsr = np.array(daily_amsr)
daily_era5 = np.array(daily_era5)

daily_frac_amsr = np.array(daily_frac_amsr)
daily_frac_era5 = np.array(daily_frac_era5)
# ==================================================
# FIGURE 4
# ==================================================

print("Creating validation figure...")

# -----------------------------------------
# Statistics
# -----------------------------------------

r = np.corrcoef(
    monthly_amsr,
    monthly_era5
)[0, 1]

slope, intercept, rvalue, pval, stderr = linregress(
    monthly_amsr,
    monthly_era5
)

# -----------------------------------------
# Normalize
# -----------------------------------------

amsr_norm = (
    monthly_amsr /
    np.mean(monthly_amsr)
)

era5_norm = (
    monthly_era5 /
    np.mean(monthly_era5)
)

# ==================================================
# FIGURE
# ==================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(16, 5),
    dpi=300,
    gridspec_kw={
        "width_ratios": [2.0, 1.1]
    }
)

# ==================================================
# (a) TIME SERIES
# ==================================================

ax1.plot(
    monthly_dates,
    amsr_norm,
    label="AMSR2",
    linewidth=2.2
)

ax1.plot(
    monthly_dates,
    era5_norm,
    label="ERA5",
    linewidth=2.2
)

ax1.set_ylabel(
    "Normalized Monthly Calm Pixel Count",
    fontsize=13,
    fontweight="bold"
)

ax1.set_xlabel(
    "Year",
    fontsize=13,
    fontweight="bold"
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
    linewidth=0.6,
    alpha=0.20
)

ax1.set_axisbelow(True)

ax1.legend(
    loc="upper right",
    frameon=False,
    fontsize=12
)

# -----------------------------------------
# Panel label (a)
# -----------------------------------------

ax1.text(
    0.0,
    1.05,
    "(a)",
    transform=ax1.transAxes,
    fontsize=13,
    fontweight="bold",
    ha="left",
    va="bottom"
)

# ==================================================
# (b) SCATTER
# ==================================================

ax2.scatter(
    monthly_amsr,
    monthly_era5,
    s=28,
    alpha=0.8
)

mx = max(
    monthly_amsr.max(),
    monthly_era5.max()
)

# -----------------------------------------
# 1:1 line
# -----------------------------------------

ax2.plot(
    [0, mx],
    [0, mx],
    "k--",
    linewidth=1.8,
    label="1:1"
)

# -----------------------------------------
# Regression line
# -----------------------------------------

xfit = np.array([
    0,
    mx
])

ax2.plot(
    xfit,
    intercept + slope * xfit,
    color="red",
    linewidth=2,
    label="Regression"
)

ax2.set_xlabel(
    "AMSR2 Monthly Calm Pixel Count",
    fontsize=13,
    fontweight="bold"
)

ax2.set_ylabel(
    "ERA5 Monthly Calm Pixel Count",
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
    linewidth=0.6,
    alpha=0.20
)

ax2.set_axisbelow(True)

# -----------------------------------------
# Statistics
# -----------------------------------------

p_exp = int(
    np.floor(
        np.log10(pval)
    )
)

p_coeff = (
    pval /
    (10 ** p_exp)
)

ax2.text(
    0.05,
    0.96,
    (
        rf"$R = {r:.3f}$"
        "\n"
        rf"Slope $= {slope:.3f}$"
        "\n"
        rf"$p = {p_coeff:.2f} \times 10^{{{p_exp}}}$"
    ),
    transform=ax2.transAxes,
    fontsize=12,
    va="top",
    ha="left"
)

ax2.legend(
    loc="lower right",
    frameon=False,
    fontsize=11
)

# -----------------------------------------
# Panel label (b)
# -----------------------------------------

ax2.text(
    0.0,
    1.05,
    "(b)",
    transform=ax2.transAxes,
    fontsize=13,
    fontweight="bold",
    ha="left",
    va="bottom"
)

# ==================================================
# LAYOUT
# ==================================================

fig.subplots_adjust(
    left=0.055,
    right=0.99,
    bottom=0.17,
    top=0.88,
    wspace=0.16
)

# ==================================================
# SAVE
# ==================================================

plt.savefig(
    os.path.join(
        OUT_DIR,
        "Figure4_AMSR2_ERA5_Validation.png"
    ),
    dpi=300,
    facecolor="white"
)

plt.close(fig)

# ==================================================
# DAILY SCATTER
# ==================================================

print("Creating daily scatter...")

r_day = np.corrcoef(
    daily_amsr,
    daily_era5
)[0,1]

slope_day, intercept_day, rvalue_day, pval_day, stderr_day = linregress(
    daily_amsr,
    daily_era5
)

r2_day = rvalue_day**2

plt.figure(figsize=(8,8))

plt.scatter(
    daily_amsr,
    daily_era5,
    s=10,
    alpha=0.4
)

mx = max(
    daily_amsr.max(),
    daily_era5.max()
)

plt.plot(
    [0,mx],
    [0,mx],
    "k--"
)

plt.xlabel(
    "AMSR Daily Calm Count"
)

plt.ylabel(
    "ERA5 Daily Calm Count"
)

plt.title(
    f"Daily Calm Counts (<{CALM_THRESHOLD} m/s)"
)

plt.text(
    0.05,
    0.95,
    (
        f"R = {r_day:.3f}\n"
        f"R² = {r2_day:.3f}\n"
        f"Slope = {slope_day:.3f}"
    ),
    transform=plt.gca().transAxes,
    va="top"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "DailyScatter.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# FIGURE 1B
# ==================================================

print("Creating fraction scatter...")

r_frac = np.corrcoef(
    monthly_frac_amsr,
    monthly_frac_era5
)[0,1]

slope_frac, intercept_frac, rvalue_frac, pval_frac, stderr_frac = linregress(
    monthly_frac_amsr,
    monthly_frac_era5
)
r2_frac = rvalue_frac**2

plt.figure(
    figsize=(8,8)
)

plt.scatter(
    monthly_frac_amsr,
    monthly_frac_era5,
    s=40
)

mx = max(
    monthly_frac_amsr.max(),
    monthly_frac_era5.max()
)

plt.plot(
    [0,mx],
    [0,mx],
    "k--"
)

plt.xlabel(
    "AMSR Monthly Calm Fraction"
)

plt.ylabel(
    "ERA5 Monthly Calm Fraction"
)

plt.title(
    f"Monthly Calm Fraction (<{CALM_THRESHOLD} m/s)"
)

plt.text(
    0.05,
    0.95,
    (
        f"R = {r_frac:.3f}\n"
        f"R² = {r2_frac:.3f}\n"
        f"Slope = {slope_frac:.3f}"
    ),
    transform=plt.gca().transAxes,
    va="top"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "MonthlyScatter_Fraction.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# SAVE NETCDF
# ==================================================

print("Saving NetCDF...")

ds_out = xr.Dataset(

    data_vars={

        "monthly_count_amsr":
            ("month", monthly_amsr),

        "monthly_count_era5":
            ("month", monthly_era5),

        "monthly_fraction_amsr":
            ("month", monthly_frac_amsr),

        "monthly_fraction_era5":
            ("month", monthly_frac_era5),
    },

    coords={

        "month":
            monthly_dates
    },

    attrs={

        "calm_threshold_ms":
            CALM_THRESHOLD,

        "comparison_pixels":
            int(comparison_mask.sum()),

        }
        )

nc_file = os.path.join(
    OUT_DIR,
    "ValidationData.nc"
)

ds_out.to_netcdf(
    nc_file
)

ds_out.close()

print(
    "Saved:",
    nc_file
)
ds_a.close()
ds_e.close()

print()
print("DONE")
print("Output:", OUT_DIR)