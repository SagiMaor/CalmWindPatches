# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:31:17 2026

@author: sagim
"""

# -*- coding: utf-8 -*-

# ============================================================
# Monthly Calm-Patch Area Time Series + Patch Area KDE
# ============================================================

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IO_EXTENT = [
    40, 100,
    -15, 25
]
# ============================================================
# PATHS
# ============================================================

BASE = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\0Wind\ClimChange"
)

PATCH_DB = os.path.join(
    BASE,
    "PatchDatabase.sqlite"
)

LIFETIME_DB = os.path.join(
    BASE,
    "Lifetime",
    "Lifetime.sqlite"
)

OUT = os.path.join(
    BASE,
    "ERA5_PatchCharacteristics"
)

os.makedirs(
    OUT,
    exist_ok=True
)

print(f"Output folder:\n{OUT}")


# ============================================================
# LOAD PATCH DATA
# ============================================================

print("\nLoading patch database...")

conn = sqlite3.connect(
    PATCH_DB
)

df = pd.read_sql(
    """
    SELECT
        patch_id,
        time,
        lat,
        lon,
        area
    FROM patches
    """,
    conn
)

conn.close()


# ============================================================
# CLEAN DATA
# ============================================================

df["time"] = pd.to_datetime(
    df["time"]
)

df["area"] = pd.to_numeric(
    df["area"],
    errors="coerce"
)

df = df[
    np.isfinite(df["area"]) &
    (df["area"] > 0)
].copy()

# ============================================================
# NORTHERN INDIAN OCEAN DOMAIN
# ============================================================

lon_min, lon_max, lat_min, lat_max = IO_EXTENT

df = df[
    (df["lon"] >= lon_min) &
    (df["lon"] <= lon_max) &
    (df["lat"] >= lat_min) &
    (df["lat"] <= lat_max)
].copy()

print(
    f"Patches inside study domain: "
    f"{len(df):,}"
)

# ============================================================
# 2. PATCH AREA ECDF
# ============================================================

areas = (
    df["area"]
    .dropna()
    .to_numpy(dtype=float)
)

areas = areas[
    np.isfinite(areas) &
    (areas > 0)
]

# ============================================================
# SORT AREAS
# ============================================================

areas_sorted = np.sort(
    areas
)

cumulative = (
    np.arange(1, len(areas_sorted) + 1)
    / len(areas_sorted)
    * 100.0
)

# ============================================================
# QUARTILES
# ============================================================

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

# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n==============================")
print("PATCH AREA DISTRIBUTION")
print("==============================")

print(
    f"N patches = {len(areas_sorted):,}"
)

print(
    f"Q25 = {q25_area:,.0f} km²"
)

print(
    f"Median = {median_area:,.0f} km²"
)

print(
    f"Q75 = {q75_area:,.0f} km²"
)

print(
    f"IQR = "
    f"{q25_area:,.0f} – "
    f"{q75_area:,.0f} km²"
)
# ============================================================
# 3. LIFETIME DISTRIBUTION
# ============================================================

print("\nLoading lifetime database...")

conn = sqlite3.connect(
    LIFETIME_DB
)

events = pd.read_sql(
    """
    SELECT
        event_id,
        event_step,
        lifetime_steps,
        lat,
        lon
    FROM event_patches
    """,
    conn
)
conn.close()

print(
    f"Loaded {len(events):,} lifetime patch records"
)


# ============================================================
# KEEP ONE RECORD PER EVENT
#
# event_patches contains multiple records for events that
# survive for multiple 3-hour steps.
#
# Keep the first record ("birth") of each event so every
# event contributes exactly once to the lifetime distribution.
# ============================================================

birth = (
    events
    .sort_values(
        ["event_id", "event_step"]
    )
    .groupby("event_id")
    .first()
    .reset_index()
)

print(
    f"Unique events: {len(birth):,}"
)

# ============================================================
# KEEP EVENTS BORN INSIDE THE STUDY DOMAIN
# ============================================================

lon_min, lon_max, lat_min, lat_max = IO_EXTENT

birth = birth[
    (birth["lon"] >= lon_min) &
    (birth["lon"] <= lon_max) &
    (birth["lat"] >= lat_min) &
    (birth["lat"] <= lat_max)
].copy()

print(
    f"Unique events born inside study domain: "
    f"{len(birth):,}"
)
# ============================================================
# CLEAN LIFETIME DATA
# ============================================================

lifetimes = pd.to_numeric(
    birth["lifetime_steps"],
    errors="coerce"
)

lifetimes = lifetimes[
    np.isfinite(lifetimes)
    &
    (lifetimes >= 1)
].astype(int)


# ============================================================
# LIFETIME COUNTS
#
# lifetime_steps = 1, 2, 3, ...
#
# Each step represents one 3-hour ERA5 interval.
# ============================================================

lifetime_counts = (
    lifetimes
    .value_counts()
    .sort_index()
)

lifetime_steps = (
    lifetime_counts.index.to_numpy()
)

lifetime_n = (
    lifetime_counts.to_numpy()
)


# ============================================================
# LIFETIME STATISTICS
# ============================================================

print("\n==============================")
print("EVENT LIFETIME DISTRIBUTION")
print("==============================")

print(
    f"N events = {len(lifetimes):,}"
)

print(
    f"Median lifetime = "
    f"{np.median(lifetimes):.1f} intervals "
    f"({np.median(lifetimes) * 3:.1f} h)"
)

print(
    f"Mean lifetime = "
    f"{np.mean(lifetimes):.2f} intervals "
    f"({np.mean(lifetimes) * 3:.2f} h)"
)

print(
    f"Maximum lifetime = "
    f"{lifetimes.max()} intervals "
    f"({lifetimes.max() * 3} h)"
)

print(
    f"One-step events = "
    f"{(lifetimes == 1).sum():,} "
    f"({100 * (lifetimes == 1).mean():.1f}%)"
)

print(
    f"Multi-step events = "
    f"{(lifetimes >= 2).sum():,} "
    f"({100 * (lifetimes >= 2).mean():.1f}%)"
)
# ============================================================
# FIGURE 6
#
# (a) Lifetime distribution
# (b) Patch-area ECDF
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
    dpi=300
)

ax1, ax2 = axes


# ============================================================
# PANEL (a) — LIFETIME DISTRIBUTION
# ============================================================

ax1.bar(
    lifetime_steps,
    lifetime_n,
    width=0.85,
    edgecolor="black",
    linewidth=0.8
)

ax1.set_yscale(
    "log"
)

ax1.set_xticks(
    [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
)

ax1.set_xticklabels(
    ["3", "12", "24", "36", "48", "60",
     "72", "84", "96", "108", "120"]
)

ax1.set_xlabel(
    "Lifetime (h)",
    fontsize=13,
    fontweight="bold"
)

ax1.set_ylabel(
    "Number of Events",
    fontsize=13,
    fontweight="bold"
)

ax1.set_xlim(
    0.3,
    lifetimes.max() + 0.7
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

ax1.set_axisbelow(
    True
)

# ------------------------------------------------------------
# Panel label (a)
# ------------------------------------------------------------

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


# ============================================================
# PANEL (b) — PATCH AREA ECDF
# ============================================================

ax2.plot(
    areas_sorted,
    cumulative,
    linewidth=2
)

# ------------------------------------------------------------
# Quartile lines
# ------------------------------------------------------------

ax2.axvline(
    q25_area,
    linestyle=":",
    linewidth=1.5,
    color="gray",
    label=f"Q25 = {q25_area:,.0f} km²"
)

ax2.axvline(
    median_area,
    linestyle="--",
    linewidth=2.2,
    color="black",
    label=f"Median = {median_area:,.0f} km²"
)

ax2.axvline(
    q75_area,
    linestyle=":",
    linewidth=1.5,
    color="gray",
    label=f"Q75 = {q75_area:,.0f} km²"
)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax2.set_xscale(
    "log"
)

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

ax2.set_axisbelow(
    True
)

ax2.legend(
    frameon=False,
    fontsize=10,
    loc="lower right"
)

# ------------------------------------------------------------
# Panel label (b)
# ------------------------------------------------------------

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


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.17,
    top=0.88,
    wspace=0.28
)


# ============================================================
# SAVE
# ============================================================

combined_file = os.path.join(
    OUT,
    "Figure6_ERA5_PatchCharacteristics.png"
)

plt.savefig(
    combined_file,
    dpi=300,
    facecolor="white"
)

plt.close(fig)

print(
    f"\n[SAVE] {combined_file}"
)