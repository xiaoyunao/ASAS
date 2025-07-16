#!/usr/bin/env python3
"""
Generate telescope field-of-view (FoV) footprints in ICRS coordinates,
save them as a FITS table, and visualize them using an Aitoff projection.
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle, GeocentricTrueEcliptic, ICRS
from astropy.table import Table
from astropy.io import fits

# ------------------- Configurable Parameters -------------------
PIXEL_SCALE = 0.426 * u.arcsec       # arcsec/pixel
NAXIS1, NAXIS2 = 6388, 9576          # image size in pixels
OVERLAP_FRAC = 0.1                   # 10% linear overlap
FILENAME_FITS = 'survey_fov_footprints.fits'
PLOT_TITLE = 'FoV Footprints'

# ------------------- Derived FoV Dimensions -------------------
fov_x = (NAXIS1 * PIXEL_SCALE).to(u.deg)   # FoV in RA direction
fov_y = (NAXIS2 * PIXEL_SCALE).to(u.deg)   # FoV in Dec direction

# Ecliptic latitude limits (centered, not edge-to-edge)
beta_min = -20 * u.deg + fov_y / 2
beta_max =  20 * u.deg - fov_y / 2
delta_beta = fov_y * (1 - OVERLAP_FRAC)
beta_centers = np.arange(beta_min.value, beta_max.value + 1e-3, delta_beta.value) * u.deg

# ------------------- Grid Center Generation -------------------
def generate_icrs_centers():
    centers = []
    for i, beta in enumerate(beta_centers):
        delta_lambda = (fov_x * (1 - OVERLAP_FRAC)) / np.cos(beta.to(u.rad).value)
        lambda_offset = 0 * u.deg if (i % 2 == 0) else delta_lambda / 2
        lambda_centers = np.arange(lambda_offset.to(u.deg).value, 360 + 1e-3,
                                   delta_lambda.to(u.deg).value) * u.deg
        for lam in lambda_centers:
            ecl = SkyCoord(lon=lam, lat=beta, frame=GeocentricTrueEcliptic(equinox='J2000'))
            centers.append(ecl.transform_to(ICRS))
    return centers

# ------------------- Compute FoV Corners -------------------
def get_fov_corners(center_icrs, fov_x, fov_y):
    """
    Return 4 corners of a rectangular FoV around center_icrs.
    Assumes small-angle approximation in tangent plane.
    """
    dec0 = center_icrs.dec
    delta_ra = (fov_x / 2) / np.cos(dec0.to(u.rad))
    delta_dec = fov_y / 2

    corners = SkyCoord(
        ra=[center_icrs.ra - delta_ra, center_icrs.ra + delta_ra,
            center_icrs.ra + delta_ra, center_icrs.ra - delta_ra],
        dec=[center_icrs.dec - delta_dec, center_icrs.dec - delta_dec,
             center_icrs.dec + delta_dec, center_icrs.dec + delta_dec]
    )
    return corners

# ------------------- Write FITS Table -------------------
def save_fov_to_fits(centers_icrs, filename):
    center_ras, center_decs = [], []
    corner_ras = [[] for _ in range(4)]
    corner_decs = [[] for _ in range(4)]

    for center in centers_icrs:
        corners = get_fov_corners(center, fov_x, fov_y)
        center_ras.append(center.ra.deg)
        center_decs.append(center.dec.deg)
        for i in range(4):
            corner_ras[i].append(corners.ra[i].deg)
            corner_decs[i].append(corners.dec[i].deg)

    table = Table()
    table['center_ra'] = center_ras
    table['center_dec'] = center_decs
    for i in range(4):
        table[f'corner_ra_{i+1}'] = corner_ras[i]
        table[f'corner_dec_{i+1}'] = corner_decs[i]

    # table.write(filename, format='fits', overwrite=True)

# ------------------- Plot Function -------------------
def plot_aitoff_fov(filename, title='FoV Footprints'):
    def plot_segments(ax, lon, lat, **kwargs):
        """Safely plot a polygon line on Aitoff projection."""
        lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi
        diff = np.abs(np.diff(lon))
        breaks = np.where(diff > np.pi)[0] + 1
        idx = np.concatenate(([0], breaks, [len(lon)]))
        for s, e in zip(idx[:-1], idx[1:]):
            ax.plot(lon[s:e], lat[s:e], **kwargs)

    tbl = Table.read(filename)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='aitoff')
    ax.grid(True)

    for row in tbl:
        ra_corners = np.array([row[f'corner_ra_{i+1}'] for i in range(4)])
        dec_corners = np.array([row[f'corner_dec_{i+1}'] for i in range(4)])
        ra_poly = np.append(ra_corners, ra_corners[0])
        dec_poly = np.append(dec_corners, dec_corners[0])
        ra_rad = Angle(ra_poly, unit=u.deg).wrap_at(180 * u.deg).radian
        dec_rad = Angle(dec_poly, unit=u.deg).radian
        plot_segments(ax, ra_rad, dec_rad, color='royalblue', linewidth=0.3)

    plt.title(title, y=1.08)
    plt.tight_layout()
    plt.show()

# ------------------- Main Execution -------------------
if __name__ == "__main__":
    centers_icrs = generate_icrs_centers()
    save_fov_to_fits(centers_icrs, FILENAME_FITS)
    plot_aitoff_fov(FILENAME_FITS, title=PLOT_TITLE)
