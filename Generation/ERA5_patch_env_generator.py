import os
import warnings
import sqlite3
import pandas as pd
import numpy as np
import xarray as xr

from datetime import datetime, timedelta

from scipy.ndimage import (
    label,
    binary_fill_holes,
    minimum_filter,
    maximum_filter,
    distance_transform_edt,
)

from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================
# CONFIG
# ============================================

ERA5_ROOT = "/work/meteogroup/era5"

R_ROOT = "/work/meteogroup/era5_1h_2d"
PATCH_DB = "/home/sagim/PatchDatabase1.sqlite"
LSM_PATH = "/work/meteogroup/era5/landseacover.nc"

OUT_FILE = "/home/sagim/Anomaly_Results_Final2.nc"
REGION = "IO"  # GLOBAL/IO
# --------------------------------------------
# thresholds
# --------------------------------------------no r

THRESH_DAILY = 1

MIN_PIXELS = 3

RAIN_EPS = 1e-8

BUFFER_KM = 100.0

MAX_LAT = 60.0

# --------------------------------------------
# topology
# --------------------------------------------

USE_HOLE_ADMIT = True

USE_BRIDGING = True

HOLE_WIND_MAX_MS = 1

# --------------------------------------------
# time
# --------------------------------------------

HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

FULL_START = datetime(1979, 4, 1)
# FULL_END = datetime(1979, 4, 3)

FULL_END = datetime(2025, 10, 31)

# --------------------------------------------
# radial settings
# --------------------------------------------


RADIAL_WINDOWS = {
    0: (0,5),
    50: (45, 55),
    100: (95, 105),
    150: (145, 155),
    200: (195, 205),
    250: (245, 255),
    300: (295, 305),
    350: (345, 355),
    400: (395, 405)
}
RADIAL_DISTANCES = sorted(RADIAL_WINDOWS.keys())
# ============================================
# VARIABLES
# ============================================

VARIABLES = [
    "sst",
    "blh",
    "slhf",
    "sshf",
    "slp",
    "wind",
    "d2m",
    "t2m",
    "omega850",
    "omega925",
    "z850",
]


# ============================================
# PATHS
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


def load_rain_and_slhf(dt):
    mats = []

    slhfs = []
    sshfs=[]

    for k in range(3):

        path = rain_path(
            dt - timedelta(hours=k)
        )

        if not os.path.exists(path):
            return None, None

        ds = xr.open_dataset(path)

        mats.append(
            ds["CP"].values.squeeze()
            +
            ds["LSP"].values.squeeze()
        )

        slhf_flux = (
                ds["SLHF"].values.squeeze()
                / (3600.0)
        )
        sshf_flux = (
                ds["SSHF"].values.squeeze()
                / (3600.0)
        )

        slhfs.append(
            slhf_flux
        )
        sshfs.append(
            sshf_flux
        )

        ds.close()

    tp3 = mats[0] + mats[1] + mats[2]

    rain_mask = (
            np.isfinite(tp3)
            &
            (tp3 <= RAIN_EPS)
    )

    slhf = np.nanmean(
        np.stack(slhfs),
        axis=0
    )

    sshf = np.nanmean(
        np.stack(sshfs),
        axis=0
    )

    return rain_mask, slhf, sshf


def sfc_path(dt):
    return os.path.join(
        R_ROOT,
        dt.strftime("%Y"),
        dt.strftime("%m"),
        f"SFC{dt.strftime('%Y%m%d')}_{dt.strftime('%H')}"
    )


def p_path(dt):
    return os.path.join(
        ERA5_ROOT,
        dt.strftime("%Y"),
        dt.strftime("%m"),
        f"P{dt.strftime('%Y%m%d')}_{dt.strftime('%H')}"
    )


def plev_path(dt):
    return os.path.join(
        "/work/meteogroup/era5/plev",
        dt.strftime("%Y"),
        dt.strftime("%m"),
        f"P{dt.strftime('%Y%m%d')}_{dt.strftime('%H')}"
    )


# ============================================
# HELPERS
# ============================================

def compute_speed(ds):
    if "u10" in ds:

        u = ds["u10"]
        v = ds["v10"]

    else:

        u = ds["U10M"]
        v = ds["V10M"]

    return np.sqrt(u ** 2 + v ** 2)


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
# LAND
# ============================================

def wrap_lon(lon):
    return ((lon + 180) % 360) - 180


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
# TOPOLOGY
# ============================================

def admit_holes(labels, wind, valid):
    out = labels.copy()

    max_lab = int(labels.max())

    if max_lab == 0:
        return out

    for lab in range(1, max_lab + 1):

        patch = labels == lab

        if not np.any(patch):
            continue

        filled = binary_fill_holes(patch)

        holes = filled & (~patch)

        if not np.any(holes):
            continue

        admit = (
                holes
                & valid
                & np.isfinite(wind)
                & (wind < HOLE_WIND_MAX_MS)
        )

        out[admit] = lab

    return out


def minimal_bridge(labels, valid, wind):
    lbl = labels.copy()

    adm = (
            (lbl == 0)
            & valid
            & np.isfinite(wind)
            & (wind < HOLE_WIND_MAX_MS)
    )

    selem = np.ones((3, 3), dtype=bool)

    BIG = np.iinfo(np.int32).max

    min_n = minimum_filter(
        np.where(lbl > 0, lbl, BIG),
        footprint=selem,
        mode="constant",
        cval=BIG
    )

    max_n = maximum_filter(
        lbl,
        footprint=selem,
        mode="constant",
        cval=0
    )

    step1 = (
            adm
            & (min_n < BIG)
            & (max_n > 0)
            & (min_n != max_n)
    )

    single_adj = (
            adm
            & (~step1)
            & (max_n > 0)
            & (min_n == max_n)
    )

    id_one = np.where(single_adj, max_n, 0).astype(np.int32)

    nb_min = minimum_filter(
        np.where(id_one > 0, id_one, BIG),
        footprint=selem,
        mode="constant",
        cval=BIG
    )

    nb_max = maximum_filter(
        id_one,
        footprint=selem,
        mode="constant",
        cval=0
    )

    step2 = (
            single_adj
            & (
                    (nb_min != id_one)
                    | (nb_max != id_one)
            )
    )

    bridge = step1 | step2

    merged = (lbl > 0) | bridge

    out, _ = label(
        merged.astype(np.uint8),
        structure=np.ones((3, 3), np.uint8)
    )

    return out


def compute_patch_mean(
        field,
        patch_mask,
        area_2d,
):
    den = np.nansum(
        area_2d[patch_mask]
    )

    if den <= 0:
        return np.nan

    return (
            np.nansum(
                field[patch_mask]
                *
                area_2d[patch_mask]
            )
            / den
    )


def compute_radial_anomalies(
        patch_mask,
        field,
        area_2d,
        dlat_km,
        dlon_km,
):
    patch_mean = compute_patch_mean(
        field,
        patch_mask,
        area_2d,
    )

    if not np.isfinite(patch_mean):
        return None

    edt = distance_transform_edt(
        patch_mask
    )

    center_row, center_col = np.unravel_index(
        np.argmax(edt),
        edt.shape
    )

    directions = [

        (-1, 0),  # N
        (-1, 1),  # NE
        (0, 1),  # E
        (1, 1),  # SE
        (1, 0),  # S
        (1, -1),  # SW
        (0, -1),  # W
        (-1, -1),  # NW
    ]

    pix_km = 0.5 * (
            dlat_km
            +
            np.nanmean(
                dlon_km[patch_mask]
            )
    )

    result = {}

    for target_km, (dmin, dmax) in RADIAL_WINDOWS.items():

        ray_values = []

        for dr, dc in directions:

            r = center_row
            c = center_col

            # ---------------------------------
            # walk to patch edge
            # ---------------------------------

            while True:

                rr = r + dr
                cc = c + dc

                if (
                        rr < 0
                        or rr >= patch_mask.shape[0]
                        or cc < 0
                        or cc >= patch_mask.shape[1]
                ):
                    break

                r = rr
                c = cc

                if not patch_mask[r, c]:
                    break

            boundary_r = r
            boundary_c = c

            vals = []

            # ---------------------------------
            # sample distance window
            # ---------------------------------

            for dist_km in np.arange(
                    dmin,
                    dmax + 1,
                    5
            ):

                n_pix = int(
                    round(
                        dist_km / pix_km
                    )
                )

                sr = boundary_r + dr * n_pix
                sc = boundary_c + dc * n_pix

                if (
                        sr < 0
                        or sr >= field.shape[0]
                        or sc < 0
                        or sc >= field.shape[1]
                ):
                    continue

                # 10 km window around sample point

                window_pix = max(
                    1,
                    int(round(5.0 / pix_km))
                )

                r0 = max(0, sr - window_pix)
                r1 = min(field.shape[0], sr + window_pix + 1)

                c0 = max(0, sc - window_pix)
                c1 = min(field.shape[1], sc + window_pix + 1)

                window = field[r0:r1, c0:c1]

                val = np.nanmean(window)

                if np.isfinite(val):
                    vals.append(val)

            if len(vals) > 0:
                ray_values.append(
                    np.nanmean(vals)
                )

        if len(ray_values) > 0:

            background = np.nanmean(
                ray_values
            )

            result[target_km] = (
                    patch_mean
                    -
                    background
            )

        else:

            result[target_km] = np.nan

    return {

        "patch_mean": patch_mean,

        "radial": result,

    }

def compute_internal_profile(
        patch_mask,
        fields,
        dlat_km,
        dlon_km,
):
    edt = distance_transform_edt(patch_mask)

    pix_km = 0.5 * (
        dlat_km +
        np.nanmean(dlon_km[patch_mask])
    )

    max_level = int(np.floor(np.nanmax(edt)))

    profile = []

    for level in range(max_level + 1):

        mask = (
            patch_mask
            &
            (edt >= level)
            &
            (edt < level + 1)
        )

        if not np.any(mask):
            continue

        row = {
            "edt_level": level,
            "pixel_size_km": pix_km,
            "n_pixels": int(np.sum(mask)),
        }

        for var in VARIABLES:

            vals = fields[var][mask]

            row[var] = np.nanmean(vals)

        profile.append(row)
    return profile
# ============================================
# PATCH BUILDER
# ============================================

def build_patches(
        wind,
        fields,
        thresh,
        valid,
        area_2d,
        global_mask,
        dlat_km,
        dlon_km,
        lat,
        lon,
        current_time,
        next_patch_id,
):
    weak = (
            valid
            & np.isfinite(wind)
            & (wind < thresh)
    )

    labels, _ = label(weak)

    if USE_HOLE_ADMIT:
        labels = admit_holes(labels, wind, valid)

    if USE_BRIDGING:
        labels = minimal_bridge(labels, valid, wind)

    sizes = np.bincount(labels.ravel())

    keep = sizes >= MIN_PIXELS

    keep[0] = False

    patches = []

    for i in range(1, len(keep)):

        if not keep[i]:
            continue

        mask = labels == i

        mask_valid = mask & global_mask

        if not np.any(mask_valid):
            continue

        rows, cols = np.where(mask_valid)

        lat_center = np.mean(lat[rows])

        lon_center = np.mean(lon[cols])

        area = np.nansum(area_2d[mask_valid])
        patch_data = {}

        internal_profile = compute_internal_profile(
            mask_valid,
            fields,
            dlat_km,
            dlon_km,
        )
        pixel_coords = np.column_stack((rows, cols))

        for var in VARIABLES:
            patch_data[var] = compute_radial_anomalies(
                mask_valid,
                fields[var],
                area_2d,
                dlat_km,
                dlon_km,
            )

        patch_record = {

            "patch_id": next_patch_id,

            "time": current_time,

            "year": current_time.year,

            "month": current_time.month,

            "day": current_time.day,

            "hour": current_time.hour,

            "lat": lat_center,

            "lon": lon_center,

            "area": area,

            "internal_profile": internal_profile,

            "pixel_coords": pixel_coords,

        }

        patch_record.update(
            patch_data
        )

        patches.append(
            patch_record
        )
        next_patch_id += 1

    return weak, patches, next_patch_id


# ============================================
# MAIN
# ============================================

def run():
    print("Loading masks...")

    lat, lon, land = load_lsm()

    keep_mask = compute_keep_mask(lat, lon, land)

    area_2d = compute_cell_area(lat, lon)

    lat2d = np.repeat(lat[:, None], len(lon), axis=1)

    lon2d = np.repeat(lon[None, :], len(lat), axis=0)

    dlat_km = np.abs(lat[1] - lat[0]) * 111.0

    dlon_km = (
            np.abs(lon[1] - lon[0])
            * 111.0
            * np.cos(np.deg2rad(lat2d))
    )

    # ========================================
    # REGION MASK
    # ========================================

    if REGION == "GLOBAL":

        global_mask = (
                (np.abs(lat2d) <= MAX_LAT)
                &
                keep_mask
        )

    elif REGION == "IO":

        IO_MASK = (

                (lat2d >= -15)
                &
                (lat2d <= 30)

                &
                (lon2d >= 20)
                &
                (lon2d <= 120)
        )

        global_mask = (
                (np.abs(lat2d) <= MAX_LAT)
                &
                keep_mask
                &
                IO_MASK
        )

    else:

        raise ValueError(
            f"Unknown REGION={REGION}"
        )
    # ========================================
    # STORAGE
    # ========================================

    times = []

    patch_database = []
    patch_internal_database = []
    patch_pixel_database = []
    next_patch_id = 1
    metrics = {}
    metrics["n_patches"] = []
    metrics["mean_patch_area"] = []
    metrics["median_patch_area"] = []
    metrics["max_patch_area"] = []
    metrics["total_patch_area"] = []
    metrics["std_patch_area"] = []

    for var in VARIABLES:

        for dist in RADIAL_DISTANCES:
            metrics[f"{var}_{dist}_aw"] = []
            metrics[f"{var}_{dist}_med"] = []
            metrics[f"{var}_{dist}_std"] = []
            metrics[f"{var}_{dist}_count"] = []

    # ========================================
    # LOOP
    # ========================================

    day = FULL_START

    while day <= FULL_END:

        print("Processing:", day)

        daily_patch_values = {

            var: {
                dist: []
                for dist in RADIAL_DISTANCES
            }

            for var in VARIABLES

        }

        daily_patch_areas = []
        n_hourly_patches = 0

        # ------------------------------------
        # LOAD
        # ------------------------------------

        for h in HOURS:

            dt = day.replace(hour=h)

            path = era5_path(dt)

            if not os.path.exists(path):
                continue

            # -----------------------------------------
            # B FILE
            # -----------------------------------------

            ds_b = xr.open_dataset(path)

            wind = compute_speed(ds_b).values.squeeze()

            temp = ds_b["SSTK"].values.squeeze()

            blh = ds_b["BLH"].values.squeeze()

            ds_b.close()

            # -----------------------------------------
            # SFC FILE
            # -----------------------------------------

            sfc = xr.open_dataset(
                sfc_path(dt)
            )

            d2m = sfc["D2M"].values.squeeze()
            t2m = sfc["T2M"].values.squeeze()
            slp = sfc["SLP"].values.squeeze()
            slp[(slp < 700) | (slp > 2000)] = np.nan

            sfc.close()

            # -----------------------------------------
            # PLEV FILE
            # -----------------------------------------

            plev = xr.open_dataset(
                plev_path(dt)
            )

            omega850 = (
                plev["OMEGA"]
                .sel(lev=850, method="nearest")
                .values
                .squeeze()
            )

            omega925 = (
                plev["OMEGA"]
                .sel(lev=925, method="nearest")
                .values
                .squeeze()
            )
            z850 = (
                plev["Z"]
                .sel(lev=850, method="nearest")
                .values
                .squeeze()
            )

            omega850[np.abs(omega850) > 100] = np.nan
            omega925[np.abs(omega925) > 100] = np.nan
            z850[(z850 < 900) | (z850 > 2500)] = np.nan

            plev.close()

            rain, slhf, sshf = load_rain_and_slhf(dt)

            if rain is None:
                continue

            fields = {}

            fields["sst"] = temp
            fields["blh"] = blh
            fields["slhf"] = slhf
            fields["sshf"] = sshf
            fields["slp"] = slp
            fields["wind"] = wind
            fields["d2m"] = d2m
            fields["t2m"] = t2m
            fields["omega850"] = omega850
            fields["omega925"] = omega925
            fields["z850"] = z850

            valid_hour = (
                    np.isfinite(wind)
                    &
                    np.isfinite(temp)
                    &
                    np.isfinite(blh)
                    &
                    np.isfinite(slhf)
                    &
                    np.isfinite(sshf)
                    &
                    np.isfinite(slp)
                    &
                    np.isfinite(wind)
                    &
                    np.isfinite(d2m)
                    &
                    np.isfinite(t2m)
                    &
                    np.isfinite(omega850)
                    &
                    np.isfinite(omega925)
                    &
                    rain
                    &
                    global_mask
            )
            weak, patches, next_patch_id = build_patches(
                wind,
                fields,
                THRESH_DAILY,
                valid_hour,
                area_2d,
                global_mask,
                dlat_km,
                dlon_km,
                lat,
                lon,
                dt,
                next_patch_id,
            )
            n_hourly_patches += len(patches)
            for p in patches:
                for row in p["internal_profile"]:

                    rec = {

                        "patch_id": p["patch_id"],
                        "time": p["time"],
                        "lat": p["lat"],
                        "lon": p["lon"],
                        "area": p["area"],

                        "edt_level": row["edt_level"],
                        "pixel_size_km": row["pixel_size_km"],
                        "n_pixels": row["n_pixels"],
                    }

                    for var in VARIABLES:
                        rec[var] = row[var]

                    patch_internal_database.append(rec)

                for r, c in p["pixel_coords"]:
                    patch_pixel_database.append({

                        "patch_id": p["patch_id"],

                        "row": int(r),

                        "col": int(c),

                    })
                record = {
                    "patch_id": p["patch_id"],
                    "time": p["time"],
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "area": p["area"],
                }
                for var in VARIABLES:

                    record[f"{var}_patch"] = p[var]["patch_mean"]

                    for dist in RADIAL_DISTANCES:
                        record[f"{var}_{dist}"] = (
                            p[var]["radial"][dist]
                        )
                patch_database.append(record)

                daily_patch_areas.append(
                    p["area"]
                )

                for var in VARIABLES:

                    if p[var] is None:
                        continue

                    for dist in RADIAL_DISTANCES:
                        daily_patch_values[var][dist].append(
                            (
                                p[var]["radial"][dist],
                                p["area"]
                            )
                        )

        if n_hourly_patches == 0:

            for k in metrics:
                if k.endswith("_count"):
                    metrics[k].append(0)
                else:
                    metrics[k].append(np.nan)

            times.append(np.datetime64(day))

            day += timedelta(days=1)

            continue
        # ====================================
        # AREA-WEIGHTED HELPERS
        # ====================================
        areas = np.array(
            daily_patch_areas
        )
        if len(areas) > 0:

            mean_area = np.nanmean(areas)

            std_area = np.nanstd(areas)

            median_area = np.nanmedian(areas)

            max_area = np.nanmax(areas)

            total_area = np.nansum(areas)

            n_patches = len(areas)

        else:

            mean_area = np.nan

            std_area = np.nan

            median_area = np.nan

            max_area = np.nan

            total_area = np.nan

            n_patches = 0
        for var in VARIABLES:

            for dist in RADIAL_DISTANCES:
                pairs = daily_patch_values[var][dist]

                if len(pairs) == 0:
                    metrics[f"{var}_{dist}_aw"].append(np.nan)
                    metrics[f"{var}_{dist}_med"].append(np.nan)
                    metrics[f"{var}_{dist}_std"].append(np.nan)
                    metrics[f"{var}_{dist}_count"].append(0)

                    continue

                vals = np.array([p[0] for p in pairs])
                areas = np.array([p[1] for p in pairs])

                good = np.isfinite(vals)
                count = np.sum(good)

                if count > 1:
                    std = np.nanstd(vals[good])
                elif count == 1:
                    std = 0.0
                else:
                    std = np.nan

                if np.any(good):

                    aw = (
                            np.nansum(
                                vals[good] * areas[good]
                            )
                            /
                            np.nansum(
                                areas[good]
                            )
                    )

                else:

                    aw = np.nan

                metrics[f"{var}_{dist}_aw"].append(aw)

                metrics[f"{var}_{dist}_med"].append(np.nanmedian(vals))
                metrics[f"{var}_{dist}_std"].append(std)
                metrics[f"{var}_{dist}_count"].append(count)
        metrics["n_patches"].append(
            n_patches
        )

        metrics["mean_patch_area"].append(
            mean_area
        )

        metrics["median_patch_area"].append(
            median_area
        )

        metrics["std_patch_area"].append(std_area)

        metrics["max_patch_area"].append(
            max_area
        )

        metrics["total_patch_area"].append(
            total_area
        )
        times.append(np.datetime64(day))

        day += timedelta(days=1)
    # ========================================
    # SAVE
    # ========================================
    print("Saving Patch Database...")

    df = pd.DataFrame(patch_database)

    conn = sqlite3.connect(PATCH_DB)


    df.to_sql(
        "patches",
        conn,
        if_exists="replace",
        index=False,
    )
    df_internal = pd.DataFrame(
        patch_internal_database
    )

    df_internal.to_sql(
        "patch_internal",
        conn,
        if_exists="replace",
        index=False,
    )

    df_pixels = pd.DataFrame(
        patch_pixel_database
    )

    df_pixels.to_sql(
        "patch_pixels",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.close()

    print("Patch Database saved:", PATCH_DB)
    print("Saving...")

    ds = xr.Dataset()

    ds["time"] = times

    for k, v in metrics.items():
        ds[k] = (
            ("time",),
            np.array(v)
        )

    encoding = {}

    for v in ds.data_vars:
        encoding[v] = {
            "zlib": True,
            "complevel": 5,
            "shuffle": True
        }

    ds.to_netcdf(
        OUT_FILE,
        encoding=encoding
    )

    print("DONE:", OUT_FILE)


# ============================================
# ENTRY
# ============================================

if __name__ == "__main__":
    run()
