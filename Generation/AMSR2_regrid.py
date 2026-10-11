"""
AMSR2 (RSS L3 daily) -> LF wind speed on 0.5° grid, keeping day/night passes (single NetCDF)
-------------------------------------------------------------------------------------------

Output:
  wind_speed(time, pass, lat, lon)

Notes:
- We avoid using keyword arguments named `pass` (Python reserved keyword).
  Always use **{"pass": ...} when setting coords or selecting.
- Pass labeling: by default uses PASS_LABELS = ["day", "night"] in file order.
  If your file order is opposite, swap them.
"""

import os
import glob
from datetime import datetime, timedelta
from typing import Tuple, Optional, List

import numpy as np
import xarray as xr


# =========================
# USER SETTINGS
# =========================
AMSR_ROOT = r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\AMSR2"
OUT_DIR   = r"C:\Users\sagim\Weizmann Institute Dropbox\Sagi Maor\0Wind\SavedData"

# Choose date range
START_DATE = (2012, 7, 3)    # (YYYY, MM, DD)
END_DATE   = (2023, 12, 31)  # (YYYY, MM, DD)

# Output filenames
OUT_FILE      = os.path.join(OUT_DIR, "AMSR2_wind_daynight.nc")
TEST_OUT_FILE = os.path.join(OUT_DIR, "AMSR2_LF_wind_0p5_daynight_TEST_10days.nc")

# Variable names
IN_VAR_NAME  = "wind_speed_LF"
OUT_VAR_NAME = "wind_speed"

# Test mode
TEST_MODE = False
TEST_MAX_DAYS = 10

# Pass labels (order follows file pass order)
# If your files are night then day, swap these.
PASS_LABELS = ["day", "night"]


# =========================
# HELPERS
# =========================
def open_ds(path: str) -> xr.Dataset:
    engines = [None, "netcdf4", "h5netcdf", "scipy"]
    last_err = None
    for eng in engines:
        try:
            if eng is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=eng)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not open {path}. Last error:\n{last_err}")


def get_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    if "lat" in ds.coords and "lon" in ds.coords:
        return "lat", "lon"
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return "latitude", "longitude"
    raise KeyError(f"lat/lon coords not found. coords={list(ds.coords)}")


def infer_pass_dim(da: xr.DataArray) -> str:
    if "pass" in da.dims:
        return "pass"
    for d in da.dims:
        if da.sizes.get(d, None) == 2:
            return d
    raise KeyError(f"Could not find pass dimension in dims={da.dims}")


def safe_stats(arr: np.ndarray):
    flat = arr.reshape(-1)
    finite = np.isfinite(flat)
    if finite.sum() == 0:
        return np.nan, np.nan, np.nan, 0, flat.size
    v = flat[finite]
    return float(v.min()), float(v.mean()), float(v.max()), int(finite.sum()), int(flat.size - finite.sum())


def target_centers_0p5():
    lat_centers = np.arange(-90.0, 90.0 + 0.5, 0.5).astype(np.float32)  # 361
    lon_centers = np.arange(0.0, 360.0, 0.5).astype(np.float32)         # 720 (0..359.5)
    return lat_centers, lon_centers


def bin_average_to_era5_like_0p5(da_latlon: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """
    Bin-average AMSR 0.25° -> ERA5-like 0.5° grid using groupby_bins.
    Expects a 2D DataArray with dims (lat_name, lon_name) (or equivalent).
    Output coords forced to exact centers:
      lat: -90..90 step 0.5
      lon:  0..359.5 step 0.5
    """
    lat_centers, lon_centers = target_centers_0p5()

    lat_edges = np.arange(-90.25, 90.25 + 0.5, 0.5)
    lon_edges = np.arange(-0.25, 359.75 + 0.5, 0.5)

    da = da_latlon.assign_coords(**{lon_name: (da_latlon[lon_name] % 360)}).sortby(lon_name)

    da_b = da.groupby_bins(lat_name, lat_edges, right=True, include_lowest=True).mean()
    da_b = da_b.groupby_bins(lon_name, lon_edges, right=True, include_lowest=True).mean()

    da_out = da_b.rename({f"{lat_name}_bins": "lat", f"{lon_name}_bins": "lon"})
    da_out = da_out.assign_coords(lat=lat_centers, lon=lon_centers).sortby("lon")
    return da_out.astype("float32")


def find_amsr_file_for_day(root: str, day: datetime) -> Optional[str]:
    """
    Look for the file for a given day inside its year folder.
    Supports both:
      RSS_AMSR2_ocean_L3_daily_YYYY-MM-DD_v*.nc
      RSS_AMSR2_ocean_L3_daily-YYYY-MM-DD_v*.nc
    If multiple matches exist, return the first (sorted).
    """
    yyyy = day.strftime("%Y")
    ymd = day.strftime("%Y-%m-%d")

    pat1 = os.path.join(root, yyyy, f"RSS_AMSR2_ocean_L3_daily_{ymd}_v*.nc")
    pat2 = os.path.join(root, yyyy, f"RSS_AMSR2_ocean_L3_daily-{ymd}_v*.nc")

    matches = sorted(glob.glob(pat1)) + sorted(glob.glob(pat2))
    return matches[0] if matches else None


def regrid_each_pass_to_0p5(w: xr.DataArray, pass_dim: str, lat_name: str, lon_name: str) -> xr.DataArray:
    """
    Take w(pass_dim, lat, lon) and return w_05(pass, lat, lon) on the 0.5° grid.
    """
    if w.sizes.get(pass_dim, None) != 2:
        raise ValueError(f"Expected pass dimension of size 2, got {w.sizes.get(pass_dim)} for dim '{pass_dim}'")

    pass_fields: List[xr.DataArray] = []
    for i in range(w.sizes[pass_dim]):
        w_i_025 = w.isel({pass_dim: i})
        w_i_05 = bin_average_to_era5_like_0p5(w_i_025, lat_name, lon_name)
        pass_fields.append(w_i_05)

    # Build pass dimension (avoid keyword arg `pass=...`)
    da_pass = xr.concat(pass_fields, dim="pass")
    da_pass = da_pass.assign_coords(**{
        "pass": np.array(PASS_LABELS[: w.sizes[pass_dim]], dtype=object)
    })
    da_pass = da_pass.transpose("pass", "lat", "lon")
    return da_pass


# =========================
# MAIN
# =========================
def build_amsr_daily_file():
    start_dt = datetime(*START_DATE)
    end_dt = datetime(*END_DATE)

    if end_dt < start_dt:
        raise ValueError("END_DATE must be >= START_DATE")

    out_file = TEST_OUT_FILE if TEST_MODE else OUT_FILE
    print(f"AMSR ROOT: {AMSR_ROOT}")
    print(f"Date range: {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")
    print(f"TEST_MODE={TEST_MODE} -> output: {out_file}\n")

    lat_target, lon_target = target_centers_0p5()

    daily_fields: List[xr.DataArray] = []
    daily_times: List[np.datetime64] = []

    n_found = 0
    n_skipped = 0

    day = start_dt
    while day <= end_dt:
        fpath = find_amsr_file_for_day(AMSR_ROOT, day)
        if fpath is None:
            n_skipped += 1
            print(f"[MISSING] {day.strftime('%Y-%m-%d')} (no file found)")
            day += timedelta(days=1)
            continue

        ds = None
        try:
            ds = open_ds(fpath)

            if IN_VAR_NAME not in ds.data_vars:
                raise KeyError(f"{IN_VAR_NAME} not found in {os.path.basename(fpath)}")

            lat_name, lon_name = get_lat_lon_names(ds)
            w = ds[IN_VAR_NAME]
            pass_dim = infer_pass_dim(w)

            # Keep the two passes separately; no daily mean
            w_05_pass = regrid_each_pass_to_0p5(w, pass_dim, lat_name, lon_name)

            daily_fields.append(w_05_pass)
            daily_times.append(np.datetime64(day.strftime("%Y-%m-%d")))
            n_found += 1

            # Print stats per pass
            for p in w_05_pass["pass"].values:
                arr = w_05_pass.sel(**{"pass": p}).values
                mn, me, mx, n_valid, n_nan = safe_stats(arr)
                print(f"[OK] {day.strftime('%Y-%m-%d')}  pass={p}  min={mn:.3f} mean={me:.3f} max={mx:.3f}  valid={n_valid} NaN={n_nan}")

            if TEST_MODE and n_found >= TEST_MAX_DAYS:
                print("\nTEST_MODE: reached TEST_MAX_DAYS, stopping early.")
                break

        except Exception as e:
            n_skipped += 1
            print(f"[SKIP] {day.strftime('%Y-%m-%d')}  reason: {e}")

        finally:
            try:
                if ds is not None:
                    ds.close()
            except Exception:
                pass

        day += timedelta(days=1)

    if n_found == 0:
        raise RuntimeError("No AMSR files processed successfully in the selected date range.")

    # Stack into one DataArray(time, pass, lat, lon)
    wind_all = xr.concat(daily_fields, dim="time")
    wind_all = wind_all.assign_coords(time=np.array(daily_times))
    wind_all = wind_all.transpose("time", "pass", "lat", "lon")
    wind_all = wind_all.assign_coords(lat=lat_target, lon=lon_target)
    wind_all.name = OUT_VAR_NAME

    ds_out = xr.Dataset(
        data_vars={OUT_VAR_NAME: wind_all},
        coords={
            "time": wind_all["time"],
            "pass": wind_all["pass"],
            "lat": wind_all["lat"],
            "lon": wind_all["lon"],
        },
        attrs={
            "source": "RSS AMSR2 L3 daily",
            "variable_used": IN_VAR_NAME,
            "description": "AMSR2 LF wind speed kept as separate day/night passes (no daily mean), regridded to 0.5° (ERA5-like centers)",
            "lon_convention": "0..359.5 step 0.5",
            "grid": "0.5 degree",
            "processing": "per_pass_bin_average_to_0p5 (no mean over pass)",
            "date_range_start": start_dt.strftime("%Y-%m-%d"),
            "date_range_end": end_dt.strftime("%Y-%m-%d"),
            "test_mode": str(TEST_MODE),
            "n_days_processed": str(n_found),
            "n_days_missing_or_skipped": str(n_skipped),
            "pass_labels": ",".join([str(x) for x in wind_all["pass"].values]),
        },
    )

    ds_out[OUT_VAR_NAME].attrs["units"] = "m s-1"
    ds_out[OUT_VAR_NAME].attrs["long_name"] = "AMSR2 LF wind speed (day/night passes, regridded to 0.5°)"

    encoding = {
        OUT_VAR_NAME: {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat": {"dtype": "float32"},
        "lon": {"dtype": "float32"},
        "pass": {"dtype": "S10"},
    }

    ds_out.to_netcdf(out_file, encoding=encoding)
    ds_out.close()

    print("\nDONE")
    print("Output file:", out_file)
    print("Days processed:", n_found)
    print("Days missing/skipped:", n_skipped)
    print("Final shape:", wind_all.shape, "(time, pass, lat, lon)")


if __name__ == "__main__":
    build_amsr_daily_file()
