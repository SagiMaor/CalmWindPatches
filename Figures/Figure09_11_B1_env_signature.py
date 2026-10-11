import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PATHS
# ==========================================================

PATCH_DB = (
    r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\ClimChange\PatchDatabase.sqlite"
)

OUTPUT_DIR = (
    r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\ClimChange\RadialComparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# SETTINGS
# ==========================================================

COMPARISON_COLORS = {
    "first": "tab:blue",
    "second": "tab:orange",
}

RADIAL_DISTANCES = [
    0,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
]

VARIABLES = [
    "sst",
    "sshf",
    "slhf",
    "omega925",
]

REGION_BOUNDS = {

    "Arabian": {
        "lon_min":50,
        "lon_max":78,
        "lat_min":5,
        "lat_max":25,
    },

    "Bengal":{
        "lon_min":78,
        "lon_max":100,
        "lat_min":5,
        "lat_max":25,
    },

    "Equatorial":{
        "lon_min":40,
        "lon_max":100,
        "lat_min":-10,
        "lat_max":5,
    },

}

REGION_COLORS = {

    "Arabian":"red",
    "Bengal":"blue",
    "Equatorial":"green",
    "ALL":"black",

}

YLABELS = {

    "sst":"ΔSST (°C)",
    "sshf":"ΔSSHF (W m$^{-2}$)",
    "slhf":"ΔSLHF (W m$^{-2}$)",
    "omega925":r"$\Delta \omega$ (Pa s$^{-1}$)",

}
RAW_YLABELS = {

    "sst": "Mean Patch SST (°C)",

    "sshf": "Mean Patch SSHF (W m$^{-2}$)",

    "slhf": "Mean Patch SLHF (W m$^{-2}$)",

    "omega925": r"Mean Patch $\omega$ (Pa s$^{-1}$)",

}

TITLES = {

    "sst":"Sea Surface Temperature",
    "sshf":"Sensible Heat Flux",
    "slhf":"Latent Heat Flux",
    "omega925": "Vertical Velocity (925 hPa)",

}

# ==========================================================
# EARLY / LATE PERIODS
# ==========================================================

TIME_PERIODS = {

    "1979–1993": {
        "start": 1979,
        "end": 1993,
        "color": COMPARISON_COLORS["first"],
    },

    "2011–2025": {
        "start": 2011,
        "end": 2025,
        "color": COMPARISON_COLORS["second"],
    },

}

# ==========================================================
# LOAD
# ==========================================================

conn = sqlite3.connect(PATCH_DB)

patch_df = pd.read_sql(
    "SELECT * FROM patches",
    conn,
)

conn.close()

patch_df["time"] = pd.to_datetime(
    patch_df["time"]
)
# ==========================================================
# LOCAL SOLAR TIME
# ==========================================================

# Convert longitude to an approximate solar-time offset.
# 15 degrees longitude = 1 hour.

solar_offset_hours = (
    patch_df["lon"] / 15.0
)

patch_df["local_time"] = (
    patch_df["time"]
    +
    pd.to_timedelta(
        solar_offset_hours,
        unit="h"
    )
)

patch_df["local_hour"] = (
    patch_df["local_time"].dt.hour
    +
    patch_df["local_time"].dt.minute / 60.0
)

# ==========================================================
# DAY / NIGHT CLASSIFICATION
#
# Day   = 06:00–18:00 local solar time
# Night = 18:00–06:00 local solar time
# ==========================================================

patch_df["day_night"] = np.where(
    (patch_df["local_hour"] >= 6.0)
    &
    (patch_df["local_hour"] < 18.0),
    "Day",
    "Night"
)

# Local calendar date
patch_df["local_date"] = (
    patch_df["local_time"]
    .dt.floor("D")
)

# ==========================================================
# CHECK
# ==========================================================

print("\n==============================")
print("DAY / NIGHT INFORMATION")
print("==============================")

print(
    patch_df["day_night"]
    .value_counts()
)

print("\nMean local hour:")

print(
    patch_df
    .groupby("day_night")["local_hour"]
    .mean()
)
# ==========================================================
# FILTER REGION
# ==========================================================

def filter_region(df, region):

    if region == "ALL":
        return df.copy()

    r = REGION_BOUNDS[region]

    return df[
        (df.lon >= r["lon_min"]) &
        (df.lon <= r["lon_max"]) &
        (df.lat >= r["lat_min"]) &
        (df.lat <= r["lat_max"])
    ].copy()

# ==========================================================
# BUILD DAILY
# ==========================================================

def build_daily_dataframe(df):

    df = df.copy()

    df["date"] = df["time"].dt.floor("D")

    groups = df.groupby("date")

    daily = pd.DataFrame({
        "date":list(groups.groups.keys())
    })

    for var in VARIABLES:

        for dist in RADIAL_DISTANCES:

            values_daily = []

            for _, group in groups:

                values = group[f"{var}_{dist}"].to_numpy(float)
                areas = group["area"].to_numpy(float)

                good = np.isfinite(values)

                values = values[good]
                areas = areas[good]

                if len(values)==0:

                    values_daily.append(np.nan)

                else:

                    values_daily.append(
                        np.sum(values*areas) /
                        np.sum(areas)
                    )

            daily[f"{var}_{dist}_aw"] = values_daily

    return daily

# ==========================================================
# BUILD DAILY FOR A TIME PERIOD
# ==========================================================

def build_daily_period_dataframe(
        df,
        start_year,
        end_year):

    subset = df[
        (df["time"].dt.year >= start_year)
        &
        (df["time"].dt.year <= end_year)
    ].copy()

    return build_daily_dataframe(subset)

# ==========================================================
# BUILD DAILY DAY / NIGHT
# ==========================================================

def build_daily_daynight_dataframe(
        df,
        period):

    # ------------------------------------------------------
    # Select Day or Night patches
    # ------------------------------------------------------

    subset = df[
        df["day_night"] == period
    ].copy()

    # ------------------------------------------------------
    # Group using LOCAL date
    # ------------------------------------------------------

    groups = subset.groupby(
        "local_date"
    )

    daily = pd.DataFrame({
        "date": list(
            groups.groups.keys()
        )
    })

    # ------------------------------------------------------
    # Daily area-weighted anomalies
    # ------------------------------------------------------

    for var in VARIABLES:

        for dist in RADIAL_DISTANCES:

            values_daily = []

            for _, group in groups:

                values = group[
                    f"{var}_{dist}"
                ].to_numpy(float)

                areas = group[
                    "area"
                ].to_numpy(float)

                good = (
                    np.isfinite(values)
                    &
                    np.isfinite(areas)
                    &
                    (areas > 0)
                )

                values = values[good]
                areas = areas[good]

                if len(values) == 0:

                    values_daily.append(
                        np.nan
                    )

                else:

                    value_aw = (
                        np.sum(
                            values * areas
                        )
                        /
                        np.sum(
                            areas
                        )
                    )

                    values_daily.append(
                        value_aw
                    )

            daily[
                f"{var}_{dist}_aw"
            ] = values_daily

    return daily

# ==========================================================
# BUILD DAILY FOR EACH BASIN
# ==========================================================

region_daily = {}

for region in ["ALL","Arabian","Bengal","Equatorial"]:

    region_daily[region] = build_daily_dataframe(

        filter_region(
            patch_df,
            region,
        )

    )

# ==========================================================
# BUILD DAY / NIGHT DATASETS
# ==========================================================

region_daily_daynight = {}

for region in [
        "ALL",
        "Arabian",
        "Bengal",
        "Equatorial"]:

    region_df = filter_region(
        patch_df,
        region
    )

    region_daily_daynight[
        region
    ] = {}

    region_daily_daynight[
        region
    ]["Day"] = (
        build_daily_daynight_dataframe(
            region_df,
            "Day"
        )
    )

    region_daily_daynight[
        region
    ]["Night"] = (
        build_daily_daynight_dataframe(
            region_df,
            "Night"
        )
    )


print("\n==============================")
print("DAILY DAY / NIGHT COUNTS")
print("==============================")

print(
    "Day:",
    len(
        region_daily_daynight[
            "ALL"
        ]["Day"]
    )
)

print(
    "Night:",
    len(
        region_daily_daynight[
            "ALL"
        ]["Night"]
    )
)

# ==========================================================
# BUILD EARLY / LATE DATASETS
# ==========================================================

region_daily_period = {}

for region in [
        "ALL",
        "Arabian",
        "Bengal",
        "Equatorial"]:

    region_df = filter_region(
        patch_df,
        region
    )

    region_daily_period[region] = {}

    for period in TIME_PERIODS:

        info = TIME_PERIODS[period]

        region_daily_period[region][period] = (

            build_daily_period_dataframe(

                region_df,

                info["start"],

                info["end"]

            )

        )
# ==========================================================
# PLOT
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.5,7.2),
    sharex=True,
)

axes = axes.ravel()

panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]

for ax, var, panel_label in zip(
        axes,
        VARIABLES,
        panel_labels
):
    
    daily = region_daily["ALL"]

    mean = []
    p25 = []
    p75 = []

    for dist in RADIAL_DISTANCES:

            vals = -daily[f"{var}_{dist}_aw"].to_numpy(float)

            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:

                mean.append(np.nan)
                p25.append(np.nan)
                p75.append(np.nan)
            
            else:
            
                mean.append(np.mean(vals))
            
                p25.append(
                    np.percentile(vals, 25)
                )
            
                p75.append(
                    np.percentile(vals, 75)
                )

    line, = ax.plot(

            RADIAL_DISTANCES,
            mean,

            marker="o",
            linewidth=2,
            color="black",

        )

    ax.fill_between(

        RADIAL_DISTANCES,
    
        p25,
    
        p75,
    
        color="0.4",
        alpha=0.25,
    
    )
    

    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(

    TITLES[var],

        fontsize=14,
    
        fontweight="bold",
    
        pad=8,
    
    )
    ax.text(
        0.0,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    ax.set_ylabel(

        YLABELS[var],
    
        fontsize=12,
    
        fontweight="bold",
    
    )

    ax.set_xticks(RADIAL_DISTANCES)
    ax.tick_params(
        labelsize=11
    )
    
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
    
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    
    # ======================================================
    # RAW PATCH VALUE — SECONDARY Y AXIS
    # ======================================================
    
    raw_values = patch_df[
        f"{var}_patch"
    ].to_numpy(float)
    
    raw_values = raw_values[
        np.isfinite(raw_values)
    ]
    
    raw_mean = np.mean(
        raw_values
    )
    
    # SST is stored in Kelvin -> convert to Celsius
    if var == "sst":
        raw_mean -= 273.15
    
    # ------------------------------------------------------
    # Secondary axis
    # ------------------------------------------------------
    
    ax2 = ax.twinx()
    
    ax2.axhline(
        raw_mean,
        linestyle=":",
        linewidth=2,
        color="0.45",
    )
    
    if var == "sst":
        pad = 1.0
    
    elif var in ["sshf", "slhf"]:
        pad = 20.0
    
    elif var == "omega925":
        pad = 0.03
    
    ax2.set_ylim(
        raw_mean - pad,
        raw_mean + pad
    )
    
    ax2.set_ylabel(
        RAW_YLABELS[var],
        fontsize=11,
        fontweight="bold",
        color="0.4"
    )
    
    ax2.tick_params(
        axis="y",
        labelsize=10,
        colors="0.4"
    )
    
    for t in ax2.get_yticklabels():
        t.set_fontweight("bold")
    
    ax2.spines["top"].set_visible(False)

    print(
        f"[RAW] {var} | "
        f"Mean = {raw_mean:.3f}"
    )


fig.supxlabel(
    "Distance from Patch Edge (km)",
    fontsize=13,

    fontweight="bold",
    y=0.05

)

fig.suptitle(
    "",
    fontsize=18,
    fontweight="bold",
    y=0.95

)

plt.tight_layout(
    rect=[0.03,0.03,1,0.95]
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "AllBasins_RadialComparison.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ==========================================================
# DAY / NIGHT RADIAL COMPARISON
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.5, 7.2),
    sharex=True,
)

axes = axes.ravel()

# ==========================================================
# LINE SETTINGS
# ==========================================================

PERIOD_SETTINGS = {

    "Day": {
        "linestyle": "-",
        "marker": "o",
        "color": COMPARISON_COLORS["first"],
    },

    "Night": {
        "linestyle": "--",
        "marker": "s",
        "color": COMPARISON_COLORS["second"],
    },
}
# ==========================================================
# VARIABLES
# ==========================================================

panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]

for ax, var, panel_label in zip(
        axes,
        VARIABLES,
        panel_labels
):
    # ======================================================
    # SECONDARY AXIS FOR RAW PATCH VALUES
    # ======================================================
    
    ax2 = ax.twinx()
    
    raw_means = {}
    
    # ======================================================
    # DAY + NIGHT
    # ======================================================

    for period in [
            "Day",
            "Night"]:

        daily = (
            region_daily_daynight[
                "ALL"
            ][period]
        )

        mean = []
        p25 = []
        p75 = []
        
        # ==================================================
        # RAW PATCH VALUE FOR THIS PERIOD
        # ==================================================
        
        raw_subset = patch_df[
            patch_df["day_night"] == period
        ]
        
        raw_values = raw_subset[
            f"{var}_patch"
        ].to_numpy(float)
        
        raw_values = raw_values[
            np.isfinite(raw_values)
        ]
        
        raw_mean = np.mean(
            raw_values
        )
        
        # SST stored in Kelvin -> Celsius
        if var == "sst":
            raw_mean -= 273.15
        
        raw_means[
            period
        ] = raw_mean
        
        # ==================================================
        # RADIAL DISTANCES
        # ==================================================

        for dist in RADIAL_DISTANCES:

            vals = -daily[
                f"{var}_{dist}_aw"
            ].to_numpy(float)

            vals = vals[
                np.isfinite(vals)
            ]

            if len(vals) == 0:

                mean.append(
                    np.nan
                )

                p25.append(
                    np.nan
                )

                p75.append(
                    np.nan
                )

            else:

                mean.append(
                    np.mean(vals)
                )

                p25.append(
                    np.percentile(
                        vals,
                        25
                    )
                )

                p75.append(
                    np.percentile(
                        vals,
                        75
                    )
                )

        # ==================================================
        # MEAN LINE
        # ==================================================

        line, = ax.plot(

            RADIAL_DISTANCES,

            mean,

            marker=(
                PERIOD_SETTINGS[
                    period
                ]["marker"]
            ),

            linestyle=(
                PERIOD_SETTINGS[
                    period
                ]["linestyle"]
            ),

            linewidth=2,

            label=period,
            color=PERIOD_SETTINGS[period]["color"],
        )

        # ==================================================
        # IQR
        #
        # Use same color as corresponding mean line.
        # ==================================================

        ax.fill_between(

            RADIAL_DISTANCES,

            p25,

            p75,

            alpha=0.12,

            color=line.get_color(),
        )
        
        # ==================================================
        # RAW MEAN — SECONDARY AXIS
        # ==================================================
        

        ax2.axhline(
            raw_mean,
            linestyle=":",
            linewidth=1.8,
            color=line.get_color(),
            alpha=0.85
        )


    # ======================================================
    # FORMATTING
    # ======================================================

    ax.grid(
        True,
        alpha=0.2,
        linewidth=0.6,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.set_title(

        TITLES[var],

        fontsize=14,

        fontweight="bold",

        pad=8,
    )
    
    ax.text(
        0.0,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    ax.set_ylabel(

        YLABELS[var],

        fontsize=12,

        fontweight="bold",
    )

    ax.set_xticks(
        RADIAL_DISTANCES
    )

    ax.tick_params(
        labelsize=11
    )

    for t in ax.get_xticklabels():

        t.set_fontweight(
            "bold"
        )

    for t in ax.get_yticklabels():

        t.set_fontweight(
            "bold"
        )
    
    
    # ======================================================
    # SECONDARY AXIS RANGE
    # ======================================================
    
    raw_day = raw_means["Day"]
    raw_night = raw_means["Night"]
    
    raw_min = min(
        raw_day,
        raw_night
    )
    
    raw_max = max(
        raw_day,
        raw_night
    )
    
    raw_range = (
        raw_max - raw_min
    )
    
    # Avoid tiny / zero axis ranges
    if var == "sst":
    
        pad = max(
            0.25,
            raw_range * 2
        )
    
    elif var in [
            "sshf",
            "slhf"]:
    
        pad = max(
            5.0,
            raw_range
        )
    
    elif var == "omega925":
    
        pad = max(
            0.005,
            raw_range
        )
    
    ax2.set_ylim(
        raw_min - pad,
        raw_max + pad
    )
    
    ax2.set_ylabel(
        RAW_YLABELS[var],
        fontsize=11,
        fontweight="bold"
    )
    
    ax2.tick_params(
        axis="y",
        labelsize=10
    )
    
    for t in ax2.get_yticklabels():
        t.set_fontweight("bold")
    
    ax2.spines["top"].set_visible(
        False
    )

# ==========================================================
# SHARED X LABEL
# ==========================================================

fig.supxlabel(

    "Distance from Patch Edge (km)",

    fontsize=13,

    fontweight="bold",

    y=0.05
)

# ==========================================================
# LEGEND
# ==========================================================

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(

    handles,

    labels,

    loc="upper center",

    ncol=2,

    frameon=False,

    fontsize=12,

    bbox_to_anchor=(
        0.5,
        0.945
    ),
)

# ==========================================================
# TITLE
# ==========================================================

fig.suptitle(

    "",

    fontsize=18,

    fontweight="bold",

    y=0.995
)

# ==========================================================
# LAYOUT
# ==========================================================

plt.tight_layout(
    rect=[
        0.03,
        0.03,
        1,
        0.90
    ]
)

# ==========================================================
# SAVE
# ==========================================================

outfile = os.path.join(
    OUTPUT_DIR,
    "AllBasins_RadialComparison_DayNight.png",
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"[SAVE] {outfile}"
)

plt.show()


# ==========================================================
# 15 year RADIAL COMPARISON
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.5, 7.2),
    sharex=True,
)

axes = axes.ravel()


# ==========================================================
# VARIABLES
# ==========================================================

panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]

for ax, var, panel_label in zip(
        axes,
        VARIABLES,
        panel_labels
):

    # ======================================================
    # SECONDARY AXIS FOR RAW PATCH VALUES
    # ======================================================
    
    ax2 = ax.twinx()
    
    raw_means = {}
    
    # ======================================================
    # DAY + NIGHT
    # ======================================================

    for period in TIME_PERIODS:

        daily = (
            region_daily_period[
                "ALL"
            ][period]
        )

        mean = []
        p25 = []
        p75 = []
        
        # ==================================================
        # RAW PATCH VALUE FOR THIS PERIOD
        # ==================================================

        info = TIME_PERIODS[period]

        raw_subset = patch_df[
            (patch_df["time"].dt.year >= info["start"])
            &
            (patch_df["time"].dt.year <= info["end"])
        ]
        
        raw_values = raw_subset[
            f"{var}_patch"
        ].to_numpy(float)
        
        raw_values = raw_values[
            np.isfinite(raw_values)
        ]
        
        raw_mean = np.mean(
            raw_values
        )
        
        # SST stored in Kelvin -> Celsius
        if var == "sst":
            raw_mean -= 273.15
        
        raw_means[
            period
        ] = raw_mean
        
        # ==================================================
        # RADIAL DISTANCES
        # ==================================================

        for dist in RADIAL_DISTANCES:

            vals = -daily[
                f"{var}_{dist}_aw"
            ].to_numpy(float)

            vals = vals[
                np.isfinite(vals)
            ]

            if len(vals) == 0:

                mean.append(
                    np.nan
                )

                p25.append(
                    np.nan
                )

                p75.append(
                    np.nan
                )

            else:

                mean.append(
                    np.mean(vals)
                )

                p25.append(
                    np.percentile(
                        vals,
                        25
                    )
                )

                p75.append(
                    np.percentile(
                        vals,
                        75
                    )
                )

        # ==================================================
        # MEAN LINE
        # ==================================================

        line, = ax.plot(

            RADIAL_DISTANCES,

            mean,
            
            marker="o",

            linestyle="-",
            
            color=TIME_PERIODS[period]["color"],

            linewidth=2,
            label=(
                f"{period} "
                f"(N={len(raw_subset):,})"
            )
        )

        # ==================================================
        # IQR
        #
        # Use same color as corresponding mean line.
        # ==================================================

        ax.fill_between(

            RADIAL_DISTANCES,

            p25,

            p75,

            alpha=0.12,

            color=line.get_color(),
        )
        
        # ==================================================
        # RAW MEAN — SECONDARY AXIS
        # ==================================================

        ax2.axhline(
            raw_mean,
            linestyle=":",
            linewidth=1.8,
            color=line.get_color(),
            alpha=0.8,
        )

    # ======================================================
    # FORMATTING
    # ======================================================

    ax.grid(
        True,
        alpha=0.2,
        linewidth=0.6,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.set_title(

        TITLES[var],

        fontsize=14,

        fontweight="bold",

        pad=8,
    )
    ax.text(
        0.0,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    ax.set_ylabel(

        YLABELS[var],

        fontsize=12,

        fontweight="bold",
    )

    ax.set_xticks(
        RADIAL_DISTANCES
    )

    ax.tick_params(
        labelsize=11
    )

    for t in ax.get_xticklabels():

        t.set_fontweight(
            "bold"
        )

    for t in ax.get_yticklabels():

        t.set_fontweight(
            "bold"
        )
    
    
    # ======================================================
    # SECONDARY AXIS RANGE
    # ======================================================

    raw_early = raw_means["1979–1993"]
    raw_late = raw_means["2011–2025"]
    
    raw_min = min(raw_early, raw_late)
    raw_max = max(raw_early, raw_late)
    
    raw_range = (
        raw_max - raw_min
    )
    
    # Avoid tiny / zero axis ranges
    if var == "sst":
    
        pad = max(
            0.25,
            raw_range * 2
        )
    
    elif var in [
            "sshf",
            "slhf"]:
    
        pad = max(
            5.0,
            raw_range
        )
    
    elif var == "omega925":
    
        pad = max(
            0.005,
            raw_range
        )
    
    ax2.set_ylim(
        raw_min - pad,
        raw_max + pad
    )
    
    ax2.set_ylabel(
        RAW_YLABELS[var],
        fontsize=11,
        fontweight="bold"
    )
    
    ax2.tick_params(
        axis="y",
        labelsize=10
    )
    
    for t in ax2.get_yticklabels():
        t.set_fontweight("bold")
    
    ax2.spines["top"].set_visible(
        False
    )

# ==========================================================
# SHARED X LABEL
# ==========================================================

fig.supxlabel(

    "Distance from Patch Edge (km)",

    fontsize=13,

    fontweight="bold",

    y=0.05
)

# ==========================================================
# LEGEND
# ==========================================================

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(

    handles,

    labels,

    loc="upper center",

    ncol=2,

    frameon=False,

    fontsize=12,

    bbox_to_anchor=(
        0.5,
        0.945
    ),
)

# ==========================================================
# TITLE
# ==========================================================

fig.suptitle(

    "",

    fontsize=18,

    fontweight="bold",

    y=0.995
)

# ==========================================================
# LAYOUT
# ==========================================================

plt.tight_layout(
    rect=[
        0.03,
        0.03,
        1,
        0.90
    ]
)

# ==========================================================
# SAVE
# ==========================================================

outfile = os.path.join(
    OUTPUT_DIR,
    "AllBasins_RadialComparison_FirstVsLast15Years.png",
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"[SAVE] {outfile}"
)

plt.show()