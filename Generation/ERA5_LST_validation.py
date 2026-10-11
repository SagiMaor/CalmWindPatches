# -*- coding: utf-8 -*-
"""
ERA5 -> Daily ASC/DESC strip product (Interpretation A, file-by-file) — *CLOSEST* SLT selection
(with UTC-hour provenance saved).

For each UTC day, process the 8 ERA5 3-hour snapshots (00,03,...,21 UTC):
  - For each snapshot map, compute LST per longitude: LST = (UTC_hour + lon/15) mod 24
  - Define ASC strip:  |LST - 13:30| <= W  (circular wrap)
  - Define DESC strip: |LST - 01:30| <= W  (circular wrap)
  - Keep only wind < 3 m/s inside each strip
  - For each pixel, select the snapshot whose LST is *closest* to the strip center (ASC or DESC),
    among snapshots that satisfy the window and calm threshold.
  - Save daily output:
      wind_asc(time,lat,lon), wind_desc(time,lat,lon)
      utc_hour_asc(time,lat,lon), utc_hour_desc(time,lat,lon)   (0,3,6,9,12,15,18,21; -1 if missing)

Only runs for W in {0.5, 1.0} as requested.
"""

import os
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import xarray as xr

# =========================
# USER SETTINGS
# =========================
B_ROOT = r"/work/meteogroup/era5"      # root where BYYYY/MM/BYYYYMMDD_H files live
OUT_DIR = r"/home/sagim"

START_DT = datetime(2013, 4, 3, 0)
END_DT = datetime(2013, 4, 3, 23)
APPLY_WIND_THRESHOLD=False

Calm_max=1
HOURS_3H = [0, 3, 6, 9, 12, 15, 18, 21]

# Fixed AMSR nominal local solar times (hours)
ASC_LST_HOUR  = 13.5   # 13:30 LST
DESC_LST_HOUR =  1.5   # 01:30 LST

# Only the requested windows
WINDOWS_HALF_H = [0.5]
# Output variable names
OUT_ASC_NAME = "wind_asc"
OUT_DESC_NAME = "wind_desc"
OUT_ASC_HOUR_NAME = "utc_hour_asc"
OUT_DESC_HOUR_NAME = "utc_hour_desc"

# =========================
# LSM SETTINGS (ONLY OCEAN + 100 km FROM LAND)
# =========================
LSM_PATH = r"/work/meteogroup/era5/landseacover.nc"
LSM_LAND_THRESHOLD = 0
BUFFER_KM = 100.0  # keep only points >= 100 km from land (requires scipy)

# =========================
# HELPERS
# =========================
def era5_file_path(root: str, dt: datetime) -> str:
    return os.path.join(
        root,
        dt.strftime("%Y"),
        dt.strftime("%m"),
        "B{0}_{1}".format(dt.strftime("%Y%m%d"), dt.strftime("%H")),
    )

def open_any_dataset(path: str) -> xr.Dataset:
    engines = [None, "netcdf4", "h5netcdf", "scipy", "cfgrib"]
    for eng in engines:
        try:
            return xr.open_dataset(path) if eng is None else xr.open_dataset(path, engine=eng)
        except Exception:
            pass
    raise RuntimeError(f"Could not open {path}")

def pick_uv_names(ds: xr.Dataset) -> Tuple[str, str]:
    if "U10M" in ds and "V10M" in ds:
        return "U10M", "V10M"
    if "u10" in ds and "v10" in ds:
        return "u10", "v10"
    raise KeyError("U/V 10 m wind not found")

def pick_lat_lon(ds: xr.Dataset) -> Tuple[str, str]:
    if "lat" in ds.coords and "lon" in ds.coords:
        return "lat", "lon"
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return "latitude", "longitude"
    raise KeyError("lat/lon not found")

def to_lon0_360(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    da = da.assign_coords(**{lon_name: da[lon_name] % 360})
    return da.sortby(lon_name)

def compute_speed(ds: xr.Dataset, lat_name: str, lon_name: str) -> xr.DataArray:
    """Compute wind speed DataArray(lat, lon) as float32, lon in 0..360 sorted."""
    u_name, v_name = pick_uv_names(ds)
    u = ds[u_name].squeeze()
    v = ds[v_name].squeeze()
    spd = np.sqrt(u.astype("float32")**2 + v.astype("float32")**2)
    spd = to_lon0_360(spd, lon_name)
    return spd

def circ_abs_diff_hours(x: np.ndarray, center: float) -> np.ndarray:
    """Circular absolute difference in hours on [0,24): |((x-center+12) mod 24) - 12|"""
    return np.abs(((x - center + 12.0) % 24.0) - 12.0)

# =========================
# LSM HELPERS
# =========================
def wrap_to_180(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0

def _kdtree_query_1nn(tree, pts):
    try:
        return tree.query(pts, k=1, workers=-1)
    except TypeError:
        pass
    try:
        return tree.query(pts, k=1, n_jobs=-1)
    except TypeError:
        pass
    return tree.query(pts, k=1)

def load_lsm(lsm_path: str):
    ds = xr.open_dataset(lsm_path)

    lat_name, lon_name = pick_lat_lon(ds)
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    lsm_var = None
    for cand in ("LSM", "land_sea_mask", "lsm"):
        if cand in ds.variables:
            lsm_var = cand
            break
    if lsm_var is None:
        for v in ds.data_vars:
            da = ds[v]
            if ("lat" in da.dims and "lon" in da.dims) or ("latitude" in da.dims and "longitude" in da.dims):
                lsm_var = v
                break
    if lsm_var is None:
        ds.close()
        raise RuntimeError(f"Could not find land-sea variable in {lsm_path}")

    try:
        lsm = ds[lsm_var].isel(time=0).values
    except Exception:
        lsm = ds[lsm_var].values
    ds.close()

    land = np.isfinite(lsm) & (lsm > LSM_LAND_THRESHOLD)

    lon_wrapped = wrap_to_180(lon)
    sort_idx = np.argsort(lon_wrapped)
    lon_wrapped_sorted = lon_wrapped[sort_idx]
    land_sorted = land[:, sort_idx]

    return np.asarray(lat, float), np.asarray(lon_wrapped_sorted, float), land_sorted.astype(bool)

def map_lsm_to_wind_grid(lsm_lats, lsm_lons_wrapped_sorted, land_sorted, wind_lat_1d, wind_lon_0_360_1d):
    wind_lon_wrapped = wrap_to_180(wind_lon_0_360_1d)

    # NN lat
    lat_idx = np.searchsorted(lsm_lats, wind_lat_1d)
    lat_idx = np.clip(lat_idx, 1, len(lsm_lats) - 1)
    lat_left = lsm_lats[lat_idx - 1]
    lat_right = lsm_lats[lat_idx]
    lat_choose_left = (np.abs(wind_lat_1d - lat_left) <= np.abs(wind_lat_1d - lat_right))
    lat_nn = np.where(lat_choose_left, lat_idx - 1, lat_idx)

    # NN lon
    lon_idx = np.searchsorted(lsm_lons_wrapped_sorted, wind_lon_wrapped)
    lon_idx = np.clip(lon_idx, 1, len(lsm_lons_wrapped_sorted) - 1)
    lon_left = lsm_lons_wrapped_sorted[lon_idx - 1]
    lon_right = lsm_lons_wrapped_sorted[lon_idx]
    lon_choose_left = (np.abs(wind_lon_wrapped - lon_left) <= np.abs(wind_lon_wrapped - lon_right))
    lon_nn = np.where(lon_choose_left, lon_idx - 1, lon_idx)

    land_on_wind = land_sorted[np.ix_(lat_nn, lon_nn)]
    return land_on_wind.astype(bool)

def compute_far_from_land_mask(lat_1d, lon_0_360_1d, land_on_grid, buffer_km):
    ocean = ~land_on_grid
    if buffer_km <= 0:
        return ocean

    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise RuntimeError("scipy is required for the land-buffer mask (cKDTree).") from e

    R = 6371.0  # km
    lon_wrapped = wrap_to_180(lon_0_360_1d)
    lat2d, lon2d = np.meshgrid(lat_1d, lon_wrapped, indexing="ij")
    latr = np.deg2rad(lat2d)
    lonr = np.deg2rad(lon2d)

    x = R * lonr * np.cos(latr)
    y = R * latr

    land_idx = np.where(land_on_grid)
    if land_idx[0].size == 0:
        return np.ones_like(land_on_grid, dtype=bool)

    land_pts = np.column_stack((x[land_idx], y[land_idx]))
    tree = cKDTree(land_pts)

    pts = np.column_stack((x.ravel(), y.ravel()))
    dist_km, _ = _kdtree_query_1nn(tree, pts)
    dist_km = dist_km.reshape(land_on_grid.shape)

    keep = ocean & (dist_km >= buffer_km)
    return keep

# =========================
# MAIN
# =========================
def build_slt_strip_products():
    os.makedirs(OUT_DIR, exist_ok=True)

    # UTC day range
    day0 = datetime(START_DT.year, START_DT.month, START_DT.day)
    day1 = datetime(END_DT.year, END_DT.month, END_DT.day)

    # Load LSM once
    print("Loading LSM:", LSM_PATH)
    lsm_lats, lsm_lons_wrapped_sorted, land_sorted = load_lsm(LSM_PATH)

    out_times = []
    out_asc_fields = {W: [] for W in WINDOWS_HALF_H}
    out_desc_fields = {W: [] for W in WINDOWS_HALF_H}
    out_asc_hours  = {W: [] for W in WINDOWS_HALF_H}
    out_desc_hours = {W: [] for W in WINDOWS_HALF_H}

    ref_lat = ref_lon = None
    lat_name = lon_name = None
    keep_mask_da: Optional[xr.DataArray] = None

    day = day0
    while day <= day1:
        # per-window daily best fields
        daily_asc = {W: None for W in WINDOWS_HALF_H}
        daily_desc = {W: None for W in WINDOWS_HALF_H}
        daily_asc_hour = {W: None for W in WINDOWS_HALF_H}
        daily_desc_hour = {W: None for W in WINDOWS_HALF_H}
        daily_asc_bestdist = {W: None for W in WINDOWS_HALF_H}
        daily_desc_bestdist = {W: None for W in WINDOWS_HALF_H}

        any_used = False
        template: Optional[xr.DataArray] = None
        lon_vals = None  # 1D lon in 0..360 sorted

        for hh in HOURS_3H:
            dt = day.replace(hour=hh)
            fpath = era5_file_path(B_ROOT, dt)
            if not os.path.exists(fpath):
                continue

            ds = open_any_dataset(fpath)
            lat_name, lon_name = pick_lat_lon(ds)
            spd = compute_speed(ds, lat_name, lon_name)
            ds.close()

            # Capture reference grid + keep mask
            if ref_lat is None:
                ref_lat = spd[lat_name].values
                ref_lon = spd[lon_name].values

                print("Mapping LSM to wind grid...")
                land_on_wind = map_lsm_to_wind_grid(
                    lsm_lats, lsm_lons_wrapped_sorted, land_sorted,
                    wind_lat_1d=np.asarray(ref_lat, float),
                    wind_lon_0_360_1d=np.asarray(ref_lon, float),
                )

                print(f"Computing keep mask (ocean-only and >= {BUFFER_KM:g} km from land)...")
                keep_mask = compute_far_from_land_mask(
                    lat_1d=np.asarray(ref_lat, float),
                    lon_0_360_1d=np.asarray(ref_lon, float),
                    land_on_grid=land_on_wind,
                    buffer_km=float(BUFFER_KM),
                )

                keep_mask_da = xr.DataArray(
                    keep_mask,
                    coords={lat_name: spd[lat_name], lon_name: spd[lon_name]},
                    dims=(lat_name, lon_name),
                )

                print(f"Keep-mask stats: keep={int(keep_mask.sum())} / total={keep_mask.size} "
                      f"({100.0*keep_mask.mean():.2f}%)")

            else:
                if spd[lat_name].size != ref_lat.size or spd[lon_name].size != ref_lon.size:
                    raise RuntimeError("Grid mismatch detected")

            if template is None:
                template = spd
                lon_vals = template[lon_name].values.astype(float)

                # initialize daily outputs
                for W in WINDOWS_HALF_H:
                    a = template.copy(deep=True); a[:] = np.nan
                    d = template.copy(deep=True); d[:] = np.nan

                    # bestdist starts as +inf
                    a_dist = template.copy(deep=True); a_dist[:] = np.inf
                    d_dist = template.copy(deep=True); d_dist[:] = np.inf

                    # hours start as -1 (int16)
                    a_h = xr.zeros_like(template, dtype="int16") - 1
                    d_h = xr.zeros_like(template, dtype="int16") - 1

                    daily_asc[W] = a
                    daily_desc[W] = d
                    daily_asc_bestdist[W] = a_dist
                    daily_desc_bestdist[W] = d_dist
                    daily_asc_hour[W] = a_h
                    daily_desc_hour[W] = d_h

            any_used = True

            # Apply keep mask
            if keep_mask_da is not None:
                spd = spd.where(keep_mask_da)

            if APPLY_WIND_THRESHOLD:
                spd = spd.where(spd < CALM_MAX)

            # LST per lon for this snapshot hour
            lst_1d = (float(hh) + (lon_vals / 15.0)) % 24.0

            # Circular diffs to centers (1D over lon)
            d_asc_1d = circ_abs_diff_hours(lst_1d, ASC_LST_HOUR)
            d_desc_1d = circ_abs_diff_hours(lst_1d, DESC_LST_HOUR)

            for W in WINDOWS_HALF_H:
                # lon masks
                asc_lon_mask = xr.DataArray(d_asc_1d <= W, coords={lon_name: template[lon_name]}, dims=(lon_name,))
                desc_lon_mask = xr.DataArray(d_desc_1d <= W, coords={lon_name: template[lon_name]}, dims=(lon_name,))

                # candidate winds (2D) with calm condition

                asc_cand = spd.where(asc_lon_mask)
                desc_cand = spd.where(desc_lon_mask)

                # distance fields (broadcast lon->(lat,lon))
                asc_dist_lon = xr.DataArray(d_asc_1d.astype("float32"), coords={lon_name: template[lon_name]}, dims=(lon_name,))
                desc_dist_lon = xr.DataArray(d_desc_1d.astype("float32"), coords={lon_name: template[lon_name]}, dims=(lon_name,))

                # broadcast: result dims (lat, lon)
                asc_dist2d = xr.broadcast(template, asc_dist_lon)[1]
                desc_dist2d = xr.broadcast(template, desc_dist_lon)[1]

                # Update where candidate exists and is closer than current best
                # Tie-breaker: if equal distance, choose smaller UTC hour.
                cand_ok_asc = asc_cand.notnull()
                cand_ok_desc = desc_cand.notnull()

                bestd_a = daily_asc_bestdist[W]
                bestd_d = daily_desc_bestdist[W]
                besth_a = daily_asc_hour[W]
                besth_d = daily_desc_hour[W]

                hh_da = xr.zeros_like(besth_a) + np.int16(hh)

                better_asc = cand_ok_asc & (
                    (asc_dist2d < bestd_a) |
                    ((asc_dist2d == bestd_a) & (hh_da < besth_a))
                )
                better_desc = cand_ok_desc & (
                    (desc_dist2d < bestd_d) |
                    ((desc_dist2d == bestd_d) & (hh_da < besth_d))
                )

                daily_asc[W] = xr.where(better_asc, asc_cand, daily_asc[W])
                daily_desc[W] = xr.where(better_desc, desc_cand, daily_desc[W])

                daily_asc_bestdist[W] = xr.where(better_asc, asc_dist2d, bestd_a)
                daily_desc_bestdist[W] = xr.where(better_desc, desc_dist2d, bestd_d)

                daily_asc_hour[W] = xr.where(better_asc, hh_da, besth_a).astype("int16")
                daily_desc_hour[W] = xr.where(better_desc, hh_da, besth_d).astype("int16")

        if not any_used:
            print(f"[SKIP] {day.strftime('%Y-%m-%d')} (no ERA5 data)")
            day += timedelta(days=1)
            continue

        out_times.append(np.datetime64(day.strftime("%Y-%m-%d")))

        for W in WINDOWS_HALF_H:
            out_asc_fields[W].append(daily_asc[W])
            out_desc_fields[W].append(daily_desc[W])
            out_asc_hours[W].append(daily_asc_hour[W])
            out_desc_hours[W].append(daily_desc_hour[W])

        print(f"[OK] {day.strftime('%Y-%m-%d')}  (built daily ASC/DESC strips: closest-SLT selection)")
        day += timedelta(days=1)

    # =========================
    # WRITE OUTPUT FILES
    # =========================
    for W in WINDOWS_HALF_H:
        asc = xr.concat(out_asc_fields[W], dim="time").assign_coords(time=out_times)
        desc = xr.concat(out_desc_fields[W], dim="time").assign_coords(time=out_times)
        asc_h = xr.concat(out_asc_hours[W], dim="time").assign_coords(time=out_times)
        desc_h = xr.concat(out_desc_hours[W], dim="time").assign_coords(time=out_times)

        asc.name = OUT_ASC_NAME
        desc.name = OUT_DESC_NAME
        asc_h.name = OUT_ASC_HOUR_NAME
        desc_h.name = OUT_DESC_HOUR_NAME

        asc = asc.astype("float32")
        desc = desc.astype("float32")
        asc_h = asc_h.astype("int16")
        desc_h = desc_h.astype("int16")

        # attrs
        asc.attrs.update({
            "units": "m s-1",
            "description": (
                f"ERA5 10m wind speed for ASC strip: per-pixel selection of the 3-hour snapshot "
                f"whose LST is closest to 13:30 within ±{W}h;"
            )
        })
        desc.attrs.update({
            "units": "m s-1",
            "description": (
                f"ERA5 10m wind speed for DESC strip: per-pixel selection of the 3-hour snapshot "
                f"whose LST is closest to 01:30 within ±{W}h;"
            )
        })
        asc_h.attrs.update({
            "long_name": "UTC hour provenance for ASC strip",
            "description": (
                "UTC snapshot hour selected for wind_asc; "
                "values in {0,3,6,9,12,15,18,21}; -1 indicates no qualifying snapshot."

            ),
            "valid_hours": "0,3,6,9,12,15,18,21",
            "_FillValue": np.int16(-1)

        })
        desc_h.attrs.update({
            "long_name": "UTC hour provenance for DESC strip",
            "description": (
                "UTC snapshot hour selected for wind_desc; "
                "values in {0,3,6,9,12,15,18,21}; -1 indicates no qualifying snapshot."
            ),
            "valid_hours": "0,3,6,9,12,15,18,21",
            "_FillValue": np.int16(-1)
        })

        # fname = (
        #     f"ERA5__SLT_Strips__ClosestSLT__W{str(W).replace('.', 'p')}h__ALLWINDS.nc"
        # )
        fname = "ERA5_SLT_Test.nc"
        out_path = os.path.join(OUT_DIR, fname)

        ds_out = xr.Dataset(
            data_vars={
                OUT_ASC_NAME: asc,
                OUT_DESC_NAME: desc,
                OUT_ASC_HOUR_NAME: asc_h,
                OUT_DESC_HOUR_NAME: desc_h,
            },
            coords={"time": asc["time"], "lat": ref_lat, "lon": ref_lon},
            attrs={
                "source": "ERA5",
                "description": "Daily ASC/DESC strip product built from 3-hourly ERA5 snapshots (closest-SLT selection).",
                "start_date": START_DT.strftime("%Y-%m-%d"),
                "end_date": END_DT.strftime("%Y-%m-%d"),
                "lon_convention": "0..360",
                "asc_lst_hour": ASC_LST_HOUR,
                "desc_lst_hour": DESC_LST_HOUR,
                "window_half_width_hours": W,
                "selection_rule": "choose_snapshot_with_min_abs(LST-center)_within_window; tie->smaller_UTC_hour",
                "note": "Time coordinate is UTC date; within-day chosen UTC hour stored in utc_hour_asc/utc_hour_desc.",
            },
        )

        encoding = {
            OUT_ASC_NAME: {"zlib": True, "complevel": 4, "dtype": "float32"},
            OUT_DESC_NAME: {"zlib": True, "complevel": 4, "dtype": "float32"},
            OUT_ASC_HOUR_NAME: {"zlib": True, "complevel": 4, "dtype": "int16"},
            OUT_DESC_HOUR_NAME: {"zlib": True, "complevel": 4, "dtype": "int16"},
            "lat": {"dtype": "float32"},
            "lon": {"dtype": "float32"},
        }

        ds_out.to_netcdf(out_path, encoding=encoding)
        ds_out.close()

        print("Wrote:", out_path)

    print("\nDONE")

if __name__ == "__main__":
    build_slt_strip_products()
