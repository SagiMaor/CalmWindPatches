# CalmWindPatches

Code used for the detection, tracking, characterization, and analysis of oceanic calm-wind patches using AMSR2 satellite observations and ERA5 reanalysis.

This repository contains the analysis and figure-generation scripts associated with the study of spatially connected regions of extremely weak sea surface winds, referred to as **calm-wind patches**. The analysis focuses primarily on the Northern Indian Ocean and examines the spatial distribution, seasonality, lifetime, environmental characteristics, and long-term variability of these features.

## Data

The analysis uses two primary datasets:

- **AMSR2** Level-3 daily ocean products from Remote Sensing Systems (RSS), including low-frequency and medium-frequency sea surface wind speed, rain rate, and cloud liquid water.
- **ERA5** reanalysis fields from the European Centre for Medium-Range Weather Forecasts (ECMWF), including 10 m winds, sea surface temperature, precipitation, turbulent surface heat fluxes, sea-level pressure, and pressure vertical velocity.

The original AMSR2 and ERA5 datasets are not distributed with this repository and should be obtained from their respective data providers.

## Repository structure

The repository is divided into scripts used to generate intermediate analysis products and scripts used to produce the manuscript figures.

```text
CalmWindPatches/
│
├── Generation/
│   ├── AMSR2_patch_detection.py
│   ├── AMSR2_regrid.py
│   ├── ERA5_patch_env_generator.py
│   ├── ERA5_LST_validation.py
│   ├── ERA5_slp_wind_clim_generator.py
│   └── ERA5_wind_fraction_generator.py
│
├── Figures/
│   ├── Figure01_02_03_amsr2_characteristics.py
│   ├── Figure04_amsr2_era_validation.py
│   ├── Figure05_era5_climatology.py
│   ├── Figure06_era5_characteristics.py
│   ├── Figure07_08_large_scale_circulation.py
│   ├── Figure09_11_B1_env_signature.py
│   └── Figure10_long_term_trends.py
│
├── README.md
└── LICENSE
```

## Analysis workflow

### AMSR2 analysis

`AMSR2_patch_detection.py` performs the AMSR2 calm-wind patch detection and tracking analysis. The detection uses the AMSR2 low-frequency wind product as the primary wind field and applies the medium-frequency wind, rain, cloud-liquid-water, coastal-distance, and minimum-size criteria used in the study. Spatially connected calm-wind regions are tracked between consecutive satellite overpasses using the persistent-overlap criterion.

`AMSR2_regrid.py` regrids the AMSR2 low-frequency wind field from the native 0.25° grid to a 0.5° grid while retaining the ascending and descending satellite passes separately. The resulting fields are used for the AMSR2–ERA5 comparison.

### ERA5 analysis

`ERA5_patch_env_generator.py` detects ERA5 calm-wind patches and calculates the environmental fields associated with each patch. It also performs the radial environmental sampling used to examine SST, sensible and latent heat fluxes, and pressure vertical velocity surrounding calm-wind patches.

`ERA5_LST_validation.py` selects ERA5 wind fields corresponding to the nominal local solar times of the AMSR2 ascending and descending overpasses. These fields are used for the direct comparison between ERA5 and AMSR2 calm-wind occurrence.

`ERA5_slp_wind_clim_generator.py` calculates the monthly ERA5 sea-level-pressure and 10 m wind climatologies used to examine the large-scale seasonal circulation over the Northern Indian Ocean.

`ERA5_wind_fraction_generator.py` calculates the daily fraction of the rain-free Northern Indian Ocean occupied by different weak-wind-speed ranges. The 0–1 m s⁻¹ fraction is used in the long-term trend analysis.

## Manuscript figures

The scripts in the `Figures` directory produce the analyses used for the manuscript figures.

| Figure | Script | Description |
| --- | --- | --- |
| Figures 1–3 | `Figure01_02_03_amsr2_characteristics.py` | Global and Northern Indian Ocean AMSR2 calm-wind occurrence, seasonal climatology, lifetime distribution, and patch-area distribution |
| Figure 4 | `Figure04_amsr2_era_validation.py` | Comparison of monthly calm-wind occurrence between AMSR2 and ERA5 |
| Figure 5 | `Figure05_era5_climatology.py` | ERA5 monthly calm-wind patch climatology and ocean coverage |
| Figure 6 | `Figure06_era5_characteristics.py` | ERA5 calm-wind patch lifetime and area distributions |
| Figures 7–8 | `Figure07_08_large_scale_circulation.py` | Monthly SLP and wind climatology and seasonal north–south SLP difference |
| Figures 9 and 11; Appendix Fig. B1 | `Figure09_11_B1_env_signature.py` | Radial environmental signatures, early/late-period comparison, and daytime/nighttime comparison |
| Figure 10 | `Figure10_long_term_trends.py` | Long-term trends in calm-wind patch occurrence, patch size, and weak-wind ocean coverage |

Some multi-panel manuscript figures were assembled from the individual outputs produced by these scripts during final figure preparation.

## Python dependencies

The scripts use the following main Python packages:

- NumPy
- pandas
- xarray
- SciPy
- Matplotlib
- Cartopy
- netCDF4

The analysis was performed in Python. Additional dependencies may be required depending on the NetCDF backend available on the user's system.

## Running the scripts

The scripts contain user-specific input and output paths corresponding to the computing environments used for the study. To run the analysis on another system, modify the paths in the configuration or user-settings section near the beginning of each script.

Because the original AMSR2 and ERA5 datasets are not included in this repository, the generation scripts require access to the corresponding source datasets.

The general workflow is:

1. Obtain the AMSR2 and/or ERA5 source data.
2. Run the relevant scripts in `Generation/` to create the intermediate patch, climatology, validation, and weak-wind datasets.
3. Update the input paths in the corresponding scripts in `Figures/`.
4. Run the figure scripts to reproduce the manuscript analyses and plots.

## Study domain

The detailed analysis focuses on the Northern Indian Ocean, defined in the manuscript as:

- Latitude: 15°S–25°N
- Longitude: 40°E–100°E

Some intermediate processing steps use larger spatial domains before the final study-domain selection is applied.

## Citation

If you use this code, please cite the associated manuscript and archived software release.

The citation and DOI for the archived version of the code will be added following publication and Zenodo deposition.

## License

This repository is distributed under the MIT License.
