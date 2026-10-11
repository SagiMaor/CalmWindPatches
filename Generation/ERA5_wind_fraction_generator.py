# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:14:11 2026

@author: sagim
"""

# ============================================
# ERA5 RAIN-FREE WEAK WIND CLIMATOLOGY
# ============================================
#
# PURPOSE
# -------
# Creates compact output datasets for:
#
# 1. Rain-free wind-speed histograms
#    - First 15 years
#    - Last 15 years
#
# 2. Rain-free weak-wind climatology (<2 m/s)
#    - Global ocean
#    - Indian Ocean
#
# MATCHES PATCH ANALYSIS CONDITIONS:
# ----------------------------------
# - Rain-free only
# - 100 km from land
# - |lat| <= 60
#
# NO PLOTTING
#
# ============================================

import os
import numpy as np
import xarray as xr

from datetime import datetime, timedelta
from scipy.spatial import cKDTree

# ============================================
# PATHS
# ============================================

ERA5_ROOT = "/work/meteogroup/era5"

R_ROOT = "/work/meteogroup/era5_1h_2d"

LSM_PATH = "/work/meteogroup/era5/landseacover.nc"

OUT_FILE = "/home/sagim/ERA5_DailyWindBinFractions.nc"

# ============================================
# TIME
# ============================================

START_DATE = datetime(1979, 1, 1)
END_DATE = datetime(2025, 12, 31)

HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

# ============================================
# THRESHOLDS
# ============================================


WIND_BINS = [
    (0, 1),
    (1, 2),
    (2, 3)
]

RAIN_EPS = 1e-8

MAX_LAT = 60.0
BUFFER_KM = 100.0

# ============================================
# REGIONS
# ============================================

REGION_BOUNDS = {

    "Indian": {

        "lon_min": 40,
        "lon_max": 100,
        "lat_min": -15,
        "lat_max": 25,

    }

}

# ============================================
# HELPERS
# ============================================

def era5_path(dt):
    return os.path.join(
        ERA5_ROOT,
        dt.strftime("%Y"),
        dt.strftime("%m"),
        f"B{dt.strftime('%Y%m%d')}_{dt.strftime('%H')}"
    )


def rain_path(dt):
    return os.path.join(
        R_ROOT,
        dt.strftime("%Y"),
        dt.strftime("%m"),
        f"R{dt.strftime('%Y%m%d')}_{dt.strftime('%H')}"
    )


def compute_speed(ds):
    if "u10" in ds:
        u = ds["u10"]
        v = ds["v10"]
    else:
        u = ds["U10M"]
        v = ds["V10M"]

    return np.sqrt(u ** 2 + v ** 2)


def wrap_lon(lon):
    return ((lon + 180) % 360) - 180


# ============================================
# RAIN
# ============================================

def load_rain_mask(dt):
    mats = []

    for k in range(3):

        path = rain_path(
            dt - timedelta(hours=k)
        )

        if not os.path.exists(path):
            return None

        try:

            ds = xr.open_dataset(path)

            if ("CP" not in ds) or ("LSP" not in ds):
                ds.close()
                return None

            mats.append(
                ds["CP"].values.squeeze()
                + ds["LSP"].values.squeeze()
            )

            ds.close()

        except:

            return None

    tp3 = mats[0] + mats[1] + mats[2]

    return (
            np.isfinite(tp3)
            & (tp3 <= RAIN_EPS)
    )


# ============================================
# AREA
# ============================================

def compute_cell_area(lat, lon):
    R = 6371.0

    lat_r = np.deg2rad(lat)

    dlat = np.abs(lat_r[1] - lat_r[0])
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))

    area_lat = (
            (R ** 2)
            * dlon
            * (
                    np.sin(lat_r + dlat / 2)
                    - np.sin(lat_r - dlat / 2)
            )
    )

    return np.abs(
        np.repeat(area_lat[:, None], len(lon), axis=1)
    )


# ============================================
# LAND MASK
# ============================================

def load_lsm():
    ds = xr.open_dataset(LSM_PATH)

    da = list(ds.data_vars.values())[0]

    while da.ndim > 2:
        da = da.isel({da.dims[0]: 0})

    land = da.values > 0.5

    lat = ds.lat.values
    lon = ds.lon.values

    ds.close()

    return lat, lon, land


def compute_keep_mask(lat, lon, land):
    R = 6371.0

    lon_w = wrap_lon(lon)

    lat2d, lon2d = np.meshgrid(
        lat,
        lon_w,
        indexing="ij"
    )

    x = (
            R
            * np.deg2rad(lon2d)
            * np.cos(np.deg2rad(lat2d))
    )

    y = R * np.deg2rad(lat2d)

    land_idx = np.where(land)

    tree = cKDTree(
        np.column_stack((x[land_idx], y[land_idx]))
    )

    pts = np.column_stack((x.ravel(), y.ravel()))

    dist, _ = tree.query(pts, k=1)

    dist = dist.reshape(lat2d.shape)

    return (~land) & (dist >= BUFFER_KM)


# ============================================
# MAIN
# ============================================

def run():
    print("Loading masks...")

    lat, lon, land = load_lsm()

    keep_mask = compute_keep_mask(lat, lon, land)

    area_2d = compute_cell_area(lat, lon)

    lon_wrap = wrap_lon(lon)

    lat2d = np.repeat(lat[:, None], len(lon), axis=1)
    lon2d = np.repeat(lon_wrap[None, :], len(lat), axis=0)

    # ----------------------------------------
    # GLOBAL MASK
    # ----------------------------------------

    # global_mask = (
    #         keep_mask
    #         & (np.abs(lat2d) <= MAX_LAT)
    # )

    region_masks = {}

    for name, r in REGION_BOUNDS.items():
        region_masks[name] = (

                keep_mask

                & (np.abs(lat2d) <= MAX_LAT)

                & (lat2d >= r["lat_min"])
                & (lat2d <= r["lat_max"])

                & (lon2d >= r["lon_min"])
                & (lon2d <= r["lon_max"])

        )

    # ========================================
    # STORAGE
    # ========================================

    times = []

    # global_all_daily = []
    # global_rainfree_daily = []


    region_all_daily = {
        name: []
        for name in REGION_BOUNDS
    }

    region_rf_daily = {
        name: []
        for name in REGION_BOUNDS
    }

    # ========================================
    # LOOP
    # ========================================

    day = START_DATE

    while day <= END_DATE:

        print(day)

        winds = []
        rains = []

        # ------------------------------------
        # LOAD DAILY
        # ------------------------------------

        for h in HOURS:

            dt = day.replace(hour=h)

            path = era5_path(dt)

            if not os.path.exists(path):
                continue

            try:

                ds = xr.open_dataset(path)

                wind = compute_speed(ds).values.squeeze()

                ds.close()

                rain = load_rain_mask(dt)

                if rain is None:
                    continue

                winds.append(wind)
                rains.append(rain)

            except:

                continue

        if len(winds) != len(HOURS):
            day += timedelta(days=1)
            continue

        # ------------------------------------
        # DAILY MEAN WIND
        # ------------------------------------

        wind_daily = np.nanmean(winds, axis=0)

        # ------------------------------------
        # DAILY RAIN-FREE MASK
        # ------------------------------------

        rain_daily = np.all(
            np.stack(rains),
            axis=0
        )

        # ------------------------------------
        # VALID MASKS
        # ------------------------------------

        # valid_global_all = (
        #         global_mask
        #         & np.isfinite(wind_daily)
        # )
        #
        # valid_global_RF = (
        #         global_mask
        #         & np.isfinite(wind_daily)
        #         & rain_daily
        # )

        # ====================================
        # DAILY WIND BIN FRACTIONS
        # ====================================

        # global_all_bins = []
        # global_RF_bins = []

        region_all_bins = {
            name: []
            for name in REGION_BOUNDS
        }

        region_RF_bins = {
            name: []
            for name in REGION_BOUNDS
        }

        for low, high in WIND_BINS:
            # # -----------------------
            # # Global - All
            # # -----------------------
            #
            # mask = (
            #         valid_global_all &
            #         (wind_daily >= low) &
            #         (wind_daily < high)
            # )
            #
            # frac = (
            #         np.nansum(area_2d[mask]) /
            #         np.nansum(area_2d[valid_global_all])
            # )
            #
            # global_all_bins.append(frac)
            #
            # # -----------------------
            # # Global - Rain Free
            # # -----------------------
            #
            # mask = (
            #         valid_global_RF &
            #         (wind_daily >= low) &
            #         (wind_daily < high)
            # )
            #
            # frac = (
            #         np.nansum(area_2d[mask]) /
            #         np.nansum(area_2d[valid_global_RF])
            # )
            #
            # global_RF_bins.append(frac)

            # -----------------------
            # REGIONS
            # -----------------------

            for region_name, region_mask in region_masks.items():

                valid_all = (
                        region_mask
                        &
                        np.isfinite(wind_daily)
                )

                valid_RF = (
                        region_mask
                        &
                        np.isfinite(wind_daily)
                        &
                        rain_daily
                )

                # ---------- ALL ----------

                mask = (
                        valid_all
                        &
                        (wind_daily >= low)
                        &
                        (wind_daily < high)
                )

                if np.any(valid_all):

                    frac = (
                            np.nansum(area_2d[mask])
                            /
                            np.nansum(area_2d[valid_all])
                    )

                else:

                    frac = np.nan

                region_all_bins[region_name].append(frac)

                # ---------- RAINFREE ----------

                mask = (
                        valid_RF
                        &
                        (wind_daily >= low)
                        &
                        (wind_daily < high)
                )

                if np.any(valid_RF):

                    frac = (
                            np.nansum(area_2d[mask])
                            /
                            np.nansum(area_2d[valid_RF])
                    )

                else:

                    frac = np.nan

                region_RF_bins[region_name].append(frac)

        # global_all_daily.append(global_all_bins)
        # global_rainfree_daily.append(global_RF_bins)

        for region_name in REGION_BOUNDS:
            region_all_daily[region_name].append(
                region_all_bins[region_name]
            )

            region_rf_daily[region_name].append(
                region_RF_bins[region_name]
            )

        times.append(np.datetime64(day))

        day += timedelta(days=1)

    # ========================================
    # SAVE
    # ========================================

    print("Saving...")

    ds_out = xr.Dataset()

    # ----------------------------------------
    # TIME SERIES
    # ----------------------------------------

    ds_out["time"] = ("time", times)

    # ds_out["global_all_daily"] = (
    #     ("time", "bin"),
    #     np.array(global_all_daily)
    # )
    #
    # ds_out["global_rainfree_daily"] = (
    #     ("time", "bin"),
    #     np.array(global_rainfree_daily)
    # )

    for region_name in REGION_BOUNDS:
        ds_out[
            f"{region_name.lower()}_all_daily"
        ] = (
            ("time", "bin"),
            np.array(region_all_daily[region_name])
        )

        ds_out[
            f"{region_name.lower()}_rainfree_daily"
        ] = (
            ("time", "bin"),
            np.array(region_rf_daily[region_name])
        )

    ds_out["bin_low"] = (
        ("bin",),
        np.array([b[0] for b in WIND_BINS])
    )

    ds_out["bin_high"] = (
        ("bin",),
        np.array([b[1] for b in WIND_BINS])
    )

    ds_out.attrs["description"] = (
        "Daily fraction of ocean area within wind-speed bins "
        "for all observations and rain-free observations."
    )

    ds_out.attrs["rain_threshold"] = RAIN_EPS

    ds_out.attrs["buffer_km"] = BUFFER_KM

    # ----------------------------------------
    # COMPRESSED SAVE
    # ----------------------------------------

    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in ds_out.data_vars
    }

    ds_out.to_netcdf(
        OUT_FILE,
        encoding=encoding
    )

    print("DONE")
    print(OUT_FILE)


# ============================================
# ENTRY
# ============================================

if __name__ == "__main__":
    run()