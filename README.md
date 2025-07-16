
---

# Adaptive Scheduler for Asteroid Survey Observations

## Overview

This project provides a dynamically adaptive real-time scheduler tailored for wide-field astronomical surveys, especially asteroid detection. It clusters fields into 45-field observation groups and ensures each group is observed **three times per night**, optimizing for visibility, sky conditions, and historical coverage.

---

## Key Features

* **Field Clustering (KMeans-based):**

  * Each scheduling cycle selects the field with the highest score and clusters it with its **44 nearest neighbors** on the celestial sphere.
  * The group is scheduled with a 3-pass cadence:
    `center → neighbors → center → neighbors → center → neighbors`.

* **Weight Model for Prioritization:**

  * Visibility (including twilight constraints)
  * Altitude at local midnight
  * Angular distance from Moon and planets
  * Time since last observation (revisit urgency)
  * Slew efficiency and sky continuity

* **Dynamic Updates:**

  * Field scores are re-evaluated between clusters.
  * Observations continue from evening twilight to astronomical dawn.

---

## File Structure

```
.
├── main.py               # Core scheduling module
├── example.py            # Example CLI script for running the scheduler
├── survey_fov_footprints.fits  # Input FITS with field footprints
├── plans/                # Output plans are saved here (default)
├── exposure_history/     # Persistent tracking of past exposures
├── processed/            # Stores observation metadata (optional)
└── README.md             # ← You're here!
```

---

## Requirements

* Python ≥ 3.7
* [Astropy](https://www.astropy.org/)
* [Astroplan](https://astroplan.readthedocs.io/)
* Pandas, NumPy, scikit-learn
* Input FITS file with survey field definitions

Install dependencies:

```bash
pip install astropy astroplan pandas numpy scikit-learn
```

---

## How to Use

You can use `example.py` to run a complete scheduling session from the command line.

### Example Command

```bash
python example.py \
  --date 2025-07-13 \
  --lat 40.393 \
  --lon 117.575 \
  --height 960 \
  --fits ./survey_fov_footprints.fits \
  --out ./plans \
  --processed-root ./processed \
  --history-dir ./exposure_history
```

### CLI Options

| Argument            | Type  | Description                                                      |
| ------------------- | ----- | ---------------------------------------------------------------- |
| `--date`            | str   | Observation date in `YYYY-MM-DD` format                          |
| `--lat`             | float | Observatory latitude (deg)                                       |
| `--lon`             | float | Observatory longitude (deg)                                      |
| `--height`          | float | Observatory altitude (meters)                                    |
| `--fits`            | str   | Path to FITS file with field footprints                          |
| `--out`             | str   | Output directory for the generated plan                          |
| `--processed-root`  | str   | Path to processed observation metadata (optional)                |
| `--history-dir`     | str   | Directory for saving exposure history                            |
| `--initialize-only` | flag  | Only initialize history for the night, without running scheduler |
| `--force`           | flag  | Overwrite existing output plan for the day                       |

---

## How It Works

1. **Initialization**

   * Parses date, site info, and FOV list from the input FITS file.
   * Prepares or loads the exposure history.

2. **Cluster Scheduling**

   * Selects a central high-priority field.
   * Clusters 44 nearest visible neighbors (within altitude and moon constraints).
   * Plans a 3-pass sequence for the group.

3. **Real-Time Update**

   * Weight map is refreshed after each cluster.
   * History is updated in memory and optionally saved to disk.

4. **Output**

   * Final schedule is written as a DataFrame and saved in `.csv` or `.pkl` format to the output path.

---

## Outputs

* A `pandas.DataFrame` with the observation plan.
* Each row contains:

  * Field ID
  * RA, Dec
  * Scheduled UTC time
  * Altitude/Azimuth
  * Revisit index (1, 2, 3)
  * Sky conditions (e.g., moon distance)

Saved at `--out` path as:

```
./plans/YYYY-MM-DD_schedule.json
```

---

## Developer Notes

* The main scheduler logic resides in `main.py`, with core classes:

  * `Scheduler`: handles nightly schedule logic.
  * `HistoryManager`: loads/saves exposure state, useful across nights.
* Uses **KMeans clustering** and **Astropy coordinate transforms** extensively.
* Designed to be modular, enabling integration with real-time control systems or simulators.

---

## Testing and Customization

To test with your own FOV or exposure data:

* Replace `survey_fov_footprints.fits` with your own FITS file. Ensure it includes at least:

  * RA/Dec of each field
  * Field ID
  * Optional: historical exposure count, visibility flags

To run only history initialization:

```bash
python example.py --date 2025-07-13 --initialize-only
```

---

## Future Enhancements (Suggested)

* Integrate weather forecast APIs for conditional weighting
* Add more detailed Moon phase/scattered light modeling
* Support for multiple telescopes/schedulers

---

## Contact

For questions or improvements, please reach out to the project maintainer.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.