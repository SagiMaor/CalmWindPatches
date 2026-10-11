

# ==========================================================

# IMPORTS

# ==========================================================

import os
import glob
import re
import csv
import time

from collections import defaultdict

import numpy as np
import xarray as xr


from scipy.ndimage import (
    label as cc_label,
    binary_fill_holes,
    distance_transform_edt,
)
# ==========================================================

# TIMERS

# ==========================================================

TIMERS = defaultdict(float)
COUNTS = defaultdict(int)

class Timer:

    
    def __init__(self, name):
    
        self.name = name
    
    def __enter__(self):
    
        self.t0 = time.perf_counter()
    
    def __exit__(self,
                 exc_type,
                 exc_val,
                 exc_tb):
    
        TIMERS[self.name] += (
            time.perf_counter() - self.t0
        )
    
        COUNTS[self.name] += 1


# ==========================================================

# USER CONFIG

# ==========================================================

AMSR2_ROOT = (
r"C:\Users\sagim\Weizmann Institute Dropbox"
r"\Sagi Maor\AMSR2"
)

OUT_DIR = (
r"C:\Users\sagim\Weizmann Institute Dropbox"
r"\Sagi Maor\0Wind\SavedData\PatchLifetime_0.92"
)

RUN_MODE = "all"

YEAR_TO_RUN = 2016

N_FILES_LIMIT = None

AMSR_EXAMPLE_FILE = (
    r"C:\Users\sagim\Weizmann Institute Dropbox"
    r"\Sagi Maor\AMSR2\2016"
    r"\RSS_AMSR2_ocean_L3_daily_2016-01-02_v08.2.nc"
)

# ----------------------------------------------------------

# REGION

# ----------------------------------------------------------

REGION_MODE = "global"

INDIAN_OCEAN = {

"lat_min": -15,
"lat_max": 30,

"lon_min": 40,
"lon_max": 110

}

# ----------------------------------------------------------

# TRACKING

# ----------------------------------------------------------

MIN_OVERLAP_PIXELS = 3


PASS_HOURS = 12

# ----------------------------------------------------------

# PATCH FILTERS

# ----------------------------------------------------------

MIN_PIXELS = 3

# ----------------------------------------------------------

# WIND FILTERS

# ----------------------------------------------------------

WIND_THRESHOLD_MS = 1

MF_CORE_TOL = 0.5

DELTA_CORE_MAX = 0.5

CLW_MAX_KG_M2 = 0.07

COAST_BUFFER_KM = 100.0

ICE_BUFFER_KM = 100.0

POLAR_CUTOFF_DEG = 60.0

WIND_LF_NAME = "wind_speed_LF"
WIND_MF_NAME = "wind_speed_MF"

COAST_MASK_NAME = "coast_mask"
CLW_NAME = "cloud_liquid_water"



# ----------------------------------------------------------

# HOLE / BRIDGE

# ----------------------------------------------------------

HOLE_WIND_MAX_MS = 1.5

MF_HOLE_TOL = 0.5

DELTA_HOLE_MAX = 0.5

BRIDGE_WIND_MAX_MS = 1.5

# ----------------------------------------------------------

# PASS INFO

# ----------------------------------------------------------

PASS_VALUES = [2, 1]

PASS_NAME = {

2: "Night",
1: "Day"

}

# ==========================================================

# GLOBAL EVENT IDS

# ==========================================================


NEXT_STATIONARY_ID = 1

# ==========================================================

# FILE PATTERNS

# ==========================================================

DATE_RE = re.compile(
r"daily_(\d{4}-\d{2}-\d{2})_"
)

# ==========================================================

# BASIC HELPERS

# ==========================================================
# ==========================================================
# GRID LOADER
# ==========================================================

def load_grid(example_file):

    with xr.open_dataset(
        example_file
    ) as ds:

        lats = ds["lat"].values

        lons = ds["lon"].values

    return lats, lons


def wrap_to_180(lon1d):

    lon1d = np.asarray(
        lon1d,
        dtype=np.float64
    )

    return (
        (lon1d + 180.0) % 360.0
    ) - 180.0

def step_to_stamp(
    date_string,
    pass_value):
    
    
    return (
        f"{date_string} "
        f"{PASS_NAME[pass_value]}"
    )


def lifetime_hours(
    lifetime_passes):
    
    
    return (
        lifetime_passes - 1
    ) * PASS_HOURS


def _to_numpy(da):

    
    if da is None:
        return None
    
    arr = (
        da.load()
        .astype(np.float32)
        .values
    )
    
    for key in (
        "_FillValue",
        "missing_value"
    ):
    
        fv = da.attrs.get(key)
    
        if fv is not None:
    
            arr = np.where(
                arr == fv,
                np.nan,
                arr
            )

    return arr


# ==========================================================

# FILE COLLECTION

# ==========================================================

def collect_files(
root_dir,
run_mode,
year):


    if run_mode.lower() == "year":
    
        pattern = os.path.join(
    
            root_dir,
            str(year),
    
            "RSS_AMSR2_ocean_L3_daily_????-??-??_v*.nc"
        )
    
    else:
    
        pattern = os.path.join(
    
            root_dir,
            "*",
    
            "RSS_AMSR2_ocean_L3_daily_????-??-??_v*.nc"
        )
    
    files = glob.glob(pattern)
    
    dated = []
    
    for f in files:
    
        m = DATE_RE.search(f)
    
        if m:
    
            dated.append(
                (m.group(1), f)
            )
    
    dated.sort(
        key=lambda x: x[0]
    )
    
    return [
        f
        for _, f in dated
    ]

# ==========================================================

# GEOMETRY HELPERS

# ==========================================================

def haversine_km(
lat1,
lon1,
lat2,
lon2):


    R = 6371.0088
    
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    
    dphi = phi2 - phi1
    
    dl = np.deg2rad(
        lon2 - lon1
    )
    
    a = (
        np.sin(dphi/2)**2
        +
        np.cos(phi1)
        *
        np.cos(phi2)
        *
        np.sin(dl/2)**2
    )
    
    return (
        2 * R *
        np.arcsin(np.sqrt(a))
    )


# ==========================================================

# DISTANCE TO TARGET

# ==========================================================

def distance_to_target_km(
target_mask,
lats,
lons):

    
    zero_at_target = np.where(
        target_mask == 1,
        0,
        1
    ).astype(np.uint8)
    
    _, (iy, ix) = (
        distance_transform_edt(
            zero_at_target,
            return_indices=True
        )
    )
    
    lon2d, lat2d = np.meshgrid(
        lons,
        lats
    )
    
    lat_t = lat2d[iy, ix]
    
    lon_t = lon2d[iy, ix]
    
    return haversine_km(
        lat2d,
        lon2d,
        lat_t,
        lon_t
    ).astype(np.float32)


# ==========================================================

# REGION SUBSETTING

# ==========================================================

def get_region_indices(
lats0,
lons0):

    
    lon_wrapped = wrap_to_180(
        lons0
    )
    
    lat_ok = (
    
        (lats0 >= INDIAN_OCEAN["lat_min"])
        &
        (lats0 <= INDIAN_OCEAN["lat_max"])
    )
    
    lon_ok = (
    
        (lon_wrapped >= INDIAN_OCEAN["lon_min"])
        &
        (lon_wrapped <= INDIAN_OCEAN["lon_max"])
    )
    
    y_idx = np.where(lat_ok)[0]
    
    x_idx = np.where(lon_ok)[0]
    
    if (
        y_idx.size == 0
        or
        x_idx.size == 0
    ):
        raise RuntimeError(
            "Region indices empty."
        )
    
    return (
        y_idx,
        x_idx,
        lats0[y_idx],
        lons0[x_idx],
        lon_wrapped[x_idx]
    )


# ==========================================================

# PASS SELECTION

# ==========================================================

def select_pass(
da,
pass_value):

    
    if da is None:
        return None
    
    if (
        "pass" in da.dims
        and
        "pass" in da.coords
    ):
    
        vals = np.asarray(
            da["pass"].values
        ).tolist()
    
        if int(pass_value) in vals:
    
            return da.sel(
                **{"pass": int(pass_value)}
            )
    
    if "pass" in da.dims:
    
        return da.isel(
            **{
                "pass":
                0 if pass_value == 2 else 1
            }
        )
    
    return da


# ==========================================================

# HOLE ADMISSION

# ==========================================================

def admit_holes(
labels,
valid_ocean,
wind_lf,
wind_mf,
clw):

    
    out = labels.copy()
    
    max_label = int(
        labels.max()
    )
    
    if max_label == 0:
        return out
    
    for lab in range(
            1,
            max_label + 1):
    
        patch = (
            labels == lab
        )
    
        if not np.any(patch):
            continue
    
        filled = binary_fill_holes(
            patch
        )
    
        holes = (
            filled
            &
            (~patch)
        )
    
        if not np.any(holes):
            continue
    
        admit = (
            holes
            &
            valid_ocean
        )
    
        admit &= (
            wind_lf
            <
            HOLE_WIND_MAX_MS
        )
    
        if wind_mf is not None:
    
            admit &= (
                wind_mf
                <
                (
                    HOLE_WIND_MAX_MS
                    +
                    MF_HOLE_TOL
                )
            )
    
            admit &= (
                np.abs(
                    wind_lf
                    -
                    wind_mf
                )
                <=
                DELTA_HOLE_MAX
            )
    
        if clw is not None:
    
            admit &= (
                clw
                <=
                CLW_MAX_KG_M2
            )
    
        out[admit] = lab
    
    return out

# ==========================================================
# UNION FIND
# ==========================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(
            range(n + 1)
        )

        self.rank = [0] * (n + 1)

    def find(self, x):

        if self.parent[x] != x:

            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:

            self.parent[ra] = rb

        elif self.rank[ra] > self.rank[rb]:

            self.parent[rb] = ra

        else:

            self.parent[rb] = ra

            self.rank[ra] += 1


# ==========================================================
# RELABEL
# ==========================================================

def relabel_by_union(
        labels,
        uf):

    out = labels.copy()

    max_label = int(
        out.max()
    )

    if max_label == 0:
        return out

    mapping = np.arange(
        max_label + 1,
        dtype=np.int32
    )

    for lab in range(
            1,
            max_label + 1):

        mapping[lab] = uf.find(lab)

    roots = np.unique(
        mapping[mapping > 0]
    )

    root_to_new = {

        root: i + 1

        for i, root in enumerate(
            roots
        )
    }

    flat = out.ravel()

    for i in range(flat.size):

        lab = flat[i]

        if lab == 0:
            continue

        flat[i] = root_to_new[
            mapping[lab]
        ]

    return flat.reshape(
        out.shape
    )


# ==========================================================
# VIRTUAL BRIDGE MERGE
# ==========================================================

def virtual_bridge_merge(
        labels_in,
        valid,
        wind_lf):

    lbl = labels_in.copy()

    H, W = lbl.shape

    max_label = int(
        lbl.max()
    )

    if max_label == 0:
        return lbl

    uf = UnionFind(
        max_label
    )

    candidate = (

        (lbl == 0)
        &
        valid
        &
        (wind_lf < BRIDGE_WIND_MAX_MS)
    )

    n8 = [

        (-1,-1),
        (-1,0),
        (-1,1),

        (0,-1),
        (0,1),

        (1,-1),
        (1,0),
        (1,1)
    ]

    def neighbors(y, x):

        s = set()

        for dy, dx in n8:

            yy = y + dy
            xx = x + dx

            if (
                0 <= yy < H
                and
                0 <= xx < W
            ):

                lab = int(
                    lbl[yy, xx]
                )

                if lab > 0:

                    s.add(lab)

        return s

    ys, xs = np.where(
        candidate
    )

    for y, x in zip(ys, xs):

        s = neighbors(y, x)

        if len(s) >= 2:

            base = list(s)[0]

            for other in list(s)[1:]:

                uf.union(
                    base,
                    other
                )

    return relabel_by_union(
        lbl,
        uf
    )


# ==========================================================
# PATCH BUILDER
# ==========================================================

def build_patch_qualified_calm_mask(
        ds,
        pass_value,
        lats0,
        lons0,
        y_idx=None,
        x_idx=None):

    if WIND_LF_NAME not in ds.variables:

        return None

    if (
        y_idx is not None
        and
        x_idx is not None
    ):

        ds = ds.isel(
            lat=y_idx,
            lon=x_idx
        )

    wind_lf = _to_numpy(

        select_pass(
            ds[WIND_LF_NAME],
            pass_value
        )
    )

    wind_mf = None

    if WIND_MF_NAME in ds.variables:
    
        wind_mf = _to_numpy(
            select_pass(
                ds[WIND_MF_NAME],
                pass_value
            )
        )
    
    rain = None
    
    if "rain_rate" in ds.variables:
    
        rain = _to_numpy(
            select_pass(
                ds["rain_rate"],
                pass_value
            )
        )
        
    clw = None

    if CLW_NAME in ds.variables:
    
        clw = _to_numpy(
            select_pass(
                ds[CLW_NAME],
                pass_value
            )
        )

    land = select_pass(
        ds["land_mask"],
        pass_value
    ).values

    ice = select_pass(
        ds["sea_ice_mask"],
        pass_value
    ).values

    noob = select_pass(
        ds["noobs_mask"],
        pass_value
    ).values

    coast = np.zeros_like(
        land
    )

    if COAST_MASK_NAME in ds.variables:

        coast = select_pass(
            ds[COAST_MASK_NAME],
            pass_value
        ).values

    lon2d, lat2d = np.meshgrid(
        lons0,
        lats0
    )
    
    polar_ok = (
        np.abs(lat2d)
        <=
        POLAR_CUTOFF_DEG
    )

    valid = (

        (land == 0)
        &
        (ice == 0)
        &
        (coast == 0)
        &
        (noob == 0)
        &
        np.isfinite(wind_lf) & polar_ok
    )
    valid_ocean = valid.copy()

    if rain is not None:
    
        valid_ocean &= (
            np.isfinite(rain)
            &
            (rain == 0.0)
        )
    if clw is not None:

        valid_ocean &= (
    
            np.isfinite(clw)
            &
            (clw <= CLW_MAX_KG_M2)
    
        )
    dist_land_km = distance_to_target_km(
        (land == 1).astype(np.uint8),
        lats0,
        lons0
    )
    
    dist_ice_km = distance_to_target_km(
        (ice == 1).astype(np.uint8),
        lats0,
        lons0
    )
    
    far_from_land = (
        dist_land_km >= COAST_BUFFER_KM
    )
    
    far_from_ice = (
        dist_ice_km >= ICE_BUFFER_KM
    )
    
    valid_det = (

        valid_ocean
        &
        far_from_land
        &
        far_from_ice
    
    )
    calm = (

        valid_det
        &
        (wind_lf < WIND_THRESHOLD_MS)
    
    )


    if wind_mf is not None:

        calm &= (
    
            wind_mf
            <
            (
                WIND_THRESHOLD_MS
                +
                MF_CORE_TOL
            )
        )
    
        calm &= (
    
            np.abs(
                wind_lf - wind_mf
            )
            <=
            DELTA_CORE_MAX
        )
        
    

    labels, _ = cc_label(
        calm.astype(np.uint8),
        structure=np.ones((3,3))
    )

    labels = admit_holes(
        labels,
        valid_ocean,
        wind_lf,
        wind_mf,
        clw
    )

    labels = virtual_bridge_merge(
        labels,
        valid_ocean,
        wind_lf
    )

    sizes = np.bincount(
        labels.ravel()
    )

    keep = (
        sizes >= MIN_PIXELS
    )

    keep[0] = False

    final_mask = keep[
        labels
    ]

    return final_mask.astype(
        bool
    )


# ==========================================================
# PATCH EXTRACTION
# ==========================================================

def extract_patches(mask):

    labels, nlab = cc_label(
        mask.astype(np.uint8),
        structure=np.ones((3,3))
    )

    patches = []

    for lab in range(
            1,
            nlab + 1):

        patch = (
            labels == lab
        )

        if not np.any(patch):
            continue

        ys, xs = np.where(
            patch
        )

        pixels = set(
            zip(
                ys.tolist(),
                xs.tolist()
            )
        )

        patches.append({

            "pixels": pixels,

            "ys": ys.astype(
                np.int32
            ),

            "xs": xs.astype(
                np.int32
            ),

            "bbox": (
                ys.min(),
                ys.max(),
                xs.min(),
                xs.max()
            )
        })

    return patches

# ==========================================================
# PATCH CENTROID
# ==========================================================

def centroid_from_pixels(
        pixel_set,
        lats,
        lons_wrapped):

    rows = np.array(
        [p[0] for p in pixel_set],
        dtype=np.int32
    )

    cols = np.array(
        [p[1] for p in pixel_set],
        dtype=np.int32
    )

    lat = float(
        np.mean(
            lats[rows]
        )
    )

    lon = float(
        np.mean(
            lons_wrapped[cols]
        )
    )

    return lat, lon


# ==========================================================
# BBOX OVERLAP
# ==========================================================

def bbox_overlap(
        box1,
        box2):

    r1min, r1max, c1min, c1max = box1
    r2min, r2max, c2min, c2max = box2

    return not (

        (r1max < r2min)
        or
        (r2max < r1min)
        or
        (c1max < c2min)
        or
        (c2max < c1min)
    )


# ==========================================================
# STATIONARY TRACKER
# ==========================================================

def update_stationary_events(
        active_events,
        patches,
        current_stamp,
        lats,
        lons_wrapped):

    global NEXT_STATIONARY_ID

    new_active = []

    ended = []

    used = set()

    # --------------------------------------
    # MATCH ACTIVE EVENTS
    # --------------------------------------

    for event in active_events:

        best_idx = -1

        best_core = None

        best_overlap = 0

        for i, patch in enumerate(patches):

            if i in used:
                continue

            if not bbox_overlap(
                    event["bbox"],
                    patch["bbox"]):
                continue

            persistent_core = (

                event["persistent_core"]
                &
                patch["pixels"]
            )

            overlap = len(
                persistent_core
            )

            if overlap > best_overlap:

                best_overlap = overlap

                best_idx = i

                best_core = persistent_core

        # ----------------------------------
        # CONTINUE
        # ----------------------------------

        if best_overlap >= MIN_OVERLAP_PIXELS:

            p = patches[best_idx]

            used.add(best_idx)

            event["pixels"] = p["pixels"]

            event["bbox"] = p["bbox"]

            event["persistent_core"] = best_core

            event["lifetime_passes"] += 1

            event["end_stamp"] = current_stamp

            new_active.append(event)

        else:

            ended.append(event)

    # --------------------------------------
    # NEW EVENTS
    # --------------------------------------

    for i, patch in enumerate(patches):

        if i in used:
            continue

        new_active.append({

            "event_id":
                NEXT_STATIONARY_ID,

            "pixels":
                patch["pixels"],

            "bbox":
                patch["bbox"],

            "persistent_core":
                patch["pixels"].copy(),

            "start_stamp":
                current_stamp,

            "end_stamp":
                current_stamp,

            "lifetime_passes":
                1
        })

        NEXT_STATIONARY_ID += 1

    return (
        new_active,
        ended
    )

def finalize_stationary_events(
        events,
        lats,
        lons_wrapped):

    rows = []

    footprints = {}

    for event in events:

        core = event[
            "persistent_core"
        ]

        core_lat, core_lon = (
            centroid_from_pixels(
                core,
                lats,
                lons_wrapped
            )
        )

        rows.append({

            "event_id":
                event["event_id"],

            "lifetime_passes":
                event["lifetime_passes"],

            "lifetime_hours":
                lifetime_hours(
                    event["lifetime_passes"]
                ),

            "start_stamp":
                event["start_stamp"],

            "end_stamp":
                event["end_stamp"],

            "core_lat":
                core_lat,

            "core_lon":
                core_lon,

            "core_pixels":
                len(core)
        })

        ys = np.array(
            [p[0] for p in core],
            dtype=np.int32
        )

        xs = np.array(
            [p[1] for p in core],
            dtype=np.int32
        )

        footprints[
            f"event_{event['event_id']}_core_y"
        ] = ys

        footprints[
            f"event_{event['event_id']}_core_x"
        ] = xs

    return rows, footprints

# ==========================================================
# CSV WRITER
# ==========================================================

def write_csv(
        rows,
        outfile):

    if len(rows) == 0:

        print(
            f"[WARN] Empty CSV: {outfile}"
        )

        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        outfile,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"[SAVE] CSV: {outfile}"
    )


# ==========================================================
# NPZ WRITER
# ==========================================================

def write_npz(
        arrays,
        outfile):

    np.savez_compressed(
        outfile,
        **arrays
    )

    print(
        f"[SAVE] NPZ: {outfile}"
    )
    
# ==========================================================
# MAIN
# ==========================================================

def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    files = collect_files(
        AMSR2_ROOT,
        RUN_MODE,
        YEAR_TO_RUN
    )

    if len(files) == 0:

        raise RuntimeError(
            "No AMSR files found."
        )

    if N_FILES_LIMIT is not None:

        files = files[
            :N_FILES_LIMIT
        ]

    print(
        f"[INFO] Files: {len(files)}"
    )

    # --------------------------------------
    # GRID
    # --------------------------------------

    lats0, lons0 = load_grid(
        AMSR_EXAMPLE_FILE
    )

    lon_wrapped_use = wrap_to_180(
        lons0
    )

    # --------------------------------------
    # REGION
    # --------------------------------------

    y_idx = None
    x_idx = None

    if REGION_MODE.lower() == "io":

        (
            y_idx,
            x_idx,
            lats0,
            lons0,
            lon_wrapped_use
        ) = get_region_indices(
            lats0,
            lons0
        )

    tag_region = (
        "GLOBAL"
        if REGION_MODE.lower() == "global"
        else "IO"
    )

    # --------------------------------------
    # OUTPUT FILES
    # --------------------------------------

    stationary_csv = os.path.join(
        OUT_DIR,
        f"StationaryEvents_{tag_region}.csv"
    )

    stationary_npz = os.path.join(
        OUT_DIR,
        f"StationaryEventFootprints_{tag_region}.npz"
    )

    # --------------------------------------
    # TRACKERS
    # --------------------------------------

    active_stationary = []

    ended_stationary = []

    total_steps = (
        len(files)
        *
        len(PASS_VALUES)
    )

    step = 0

    # --------------------------------------
    # MAIN LOOP
    # --------------------------------------

    for file_path in files:

        m = DATE_RE.search(
            file_path
        )

        if m:

            date_str = m.group(1)

        else:

            date_str = "unknown"

        with xr.open_dataset(
            file_path
        ) as ds:

            for pv in PASS_VALUES:

                step += 1

                calm = (
                    build_patch_qualified_calm_mask(
                        ds,
                        pv,
                        lats0,
                        lons0,
                        y_idx=y_idx,
                        x_idx=x_idx
                    )
                )

                if calm is None:
                    continue

                patches = extract_patches(
                    calm
                )

                current_stamp = (
                    step_to_stamp(
                        date_str,
                        pv
                    )
                )

                # --------------------------
                # STATIONARY
                # --------------------------

                (
                    active_stationary,
                    ended_now
                ) = update_stationary_events(

                    active_stationary,
                    patches,
                    current_stamp,
                    lats0,
                    lon_wrapped_use
                )

                ended_stationary.extend(
                    ended_now
                )

                if step % 50 == 0:

                    print(
                        f"[INFO] "
                        f"{step}/{total_steps} "
                        f"stationary={len(active_stationary)}"
                    )

    # --------------------------------------
    # FINALIZE ACTIVE EVENTS
    # --------------------------------------

    ended_stationary.extend(
        active_stationary
    )

    # --------------------------------------
    # BUILD OUTPUTS
    # --------------------------------------

    (
        stationary_rows,
        stationary_arrays
    ) = finalize_stationary_events(

        ended_stationary,
        lats0,
        lon_wrapped_use
    )

    # --------------------------------------
    # SAVE
    # --------------------------------------


    write_csv(
        stationary_rows,
        stationary_csv
    )

    write_npz(
        stationary_arrays,
        stationary_npz
    )

    print()

    print("=" * 60)


    print(
        f"Stationary events saved: "
        f"{len(stationary_rows)}"
    )

    print("=" * 60)
    
if __name__ == "__main__":

    main()