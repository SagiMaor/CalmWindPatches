# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:04:38 2026

@author: sagin
"""
# -*- coding: utf-8 -*-
"""
Trend analysis of ERA5 weak-wind regimes
Indian Ocean
Rain-free only

Author: Sagi Maor
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from scipy.stats import kendalltau
from scipy.stats.mstats import theilslopes

# ==========================================================
# PATHS
# ==========================================================

FILE = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\ClimChange"
    r"\ERA5_DailyWindBinFractions.nc"
)

OUT_DIR = (
    r"C:\Users\sagin\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\ClimChange"
    r"\WindTrend"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading...")

ds = xr.open_dataset(FILE)

time = pd.to_datetime(
    ds.time.values
)

bin_low = ds.bin_low.values
bin_high = ds.bin_high.values

bin_labels = [

    f"{int(a)}-{int(b)} m/s"

    for a, b in zip(
        bin_low,
        bin_high
    )

]

# Rain-free Indian Ocean

wind = ds.indian_rainfree_daily.values

# ==========================================================
# BUILD DATAFRAME
# ==========================================================

df = pd.DataFrame(

    wind,

    columns=bin_labels,

    index=time

)

df.index.name = "Date"

df["Year"] = df.index.year

print(df.head())

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def compute_trend(y):

    """
    Theil-Sen slope
    Kendall significance
    """

    y = np.asarray(y)

    t = np.arange(len(y))

    good = np.isfinite(y)

    y = y[good]
    t = t[good]

    slope, intercept, lo, hi = theilslopes(
        y,
        t
    )

    tau, p = kendalltau(
        t,
        y
    )

    return {

        "slope": slope,

        "intercept": intercept,

        "p": p,

    }


def annual_statistics(series):

    """
    Annual mean
    95% confidence interval
    """

    grouped = (

        series

        .groupby(df["Year"])

    )

    annual_mean = grouped.mean()

    annual_std = grouped.std()

    annual_n = grouped.count()

    ci95 = (

        1.96

        * annual_std

        / np.sqrt(annual_n)

    )

    return (

        annual_mean,

        ci95,

    )

# ==========================================================
# BUILD ANNUAL TIME SERIES
# ==========================================================

annual = {}

for b in bin_labels:

    mean, ci = annual_statistics(

        df[b]

    )

    annual[b] = {

        "years": mean.index.values,

        "mean": mean.values * 100,

        "ci95": ci.values * 100,

    }

print()

print("Finished annual statistics.")

for b in bin_labels:

    print(

        b,

        annual[b]["mean"].mean()

    )
# ==========================================================
# FIGURE
# Annual Trends
# ==========================================================

fig, ax = plt.subplots(

    figsize=(11,5)

)

b = "0-1 m/s"
years = annual[b]["years"]

y = annual[b]["mean"]

ci95 = annual[b]["ci95"]

trend = compute_trend(y)

t = np.arange(len(y))

fit = (

    trend["intercept"]

    +

    trend["slope"] * t

)

# -----------------------------------------
# Annual values
# -----------------------------------------

ax.plot(

    years,

    y,

    color="0.65",

    linewidth=2,

    zorder=2,

)

# -----------------------------------------
# 95% CI
# -----------------------------------------

ax.fill_between(

    years,

    y - ci95,

    y + ci95,

    color="0.80",

    alpha=0.35,

    zorder=1,

)

# -----------------------------------------
# Trend
# -----------------------------------------

ax.plot(

    years,

    fit,

    color="tab:red",

    linewidth=3,

    zorder=3,

)

# -----------------------------------------
# Relative trend
# -----------------------------------------

mean_val = np.mean(y)

relative = (

    trend["slope"]

    * 10

    / mean_val

    * 100

)

stats = (

    f"Slope = "

    f"{trend['slope']*10:.3f} "

    "% decade$^{-1}$\n"

    f"Relative = {relative:.2f}% decade$^{{-1}}$\n"

    f"p = {trend['p']:.2e}"

)

ax.text(

    0.02,

    0.97,

    stats,

    transform=ax.transAxes,

    va="top",

    fontsize=10,

    fontweight="bold",

    bbox=dict(

        facecolor="white",

        edgecolor="0.7",

        alpha=0.95,

        boxstyle="round,pad=0.3"

    )

)

# -----------------------------------------
# Formatting
# -----------------------------------------

ax.set_ylabel(

    "Ocean Area (%)",

    fontsize=12,

    fontweight="bold",

)

ax.grid(

    linestyle="--",

    alpha=0.3,

)

ax.tick_params(

    labelsize=11

)

for tick in ax.get_xticklabels():

    tick.set_fontweight("bold")

for tick in ax.get_yticklabels():

    tick.set_fontweight("bold")

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.set_xlabel(

    "Year",

    fontsize=13,

    fontweight="bold",

)

plt.suptitle(

    "Annual Mean Fraction of the Rain-Free Indian Ocean\n"
    "With 0-1 m/s winds",

    fontsize=18,

    fontweight="bold",

)

plt.tight_layout(
    rect=[0.03,0.03,1,0.92]
)
plt.savefig(

    os.path.join(

        OUT_DIR,

        "Annual_WindBin_Trends.png",

    ),

    dpi=300,

    bbox_inches="tight",

)

plt.show()