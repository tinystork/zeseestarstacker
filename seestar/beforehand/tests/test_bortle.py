# -----------------------------------------------------------------------------
# Auteur       : TRISTAN NAULEAU 
# Date         : 2025-07-12
# Licence      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# Ce travail est distribué librement en accord avec les termes de la
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# Vous êtes libre de redistribuer et de modifier ce code, à condition
# de conserver cette notice et de mentionner que je suis l’auteur
# de tout ou partie du code si vous le réutilisez.
# -----------------------------------------------------------------------------
# Author       : TRISTAN NAULEAU
# Date         : 2025-07-12
# License      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# This work is freely distributed under the terms of the
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# You are free to redistribute and modify this code, provided that
# you keep this notice and mention that I am the author
# of all or part of the code if you reuse it.
# -----------------------------------------------------------------------------
import os
import sys
import numpy as np
import rasterio
from rasterio.transform import from_origin
import zipfile
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from bortle_utils import (
    load_bortle_raster,
    ucd_to_sqm,
    sqm_to_bortle,
    sample_bortle_dataset,
    ucd_to_bortle,
)
from analyse_logic import _load_bortle_raster, write_telescope_pollution_csv

def test_sqm_to_bortle(tmp_path):
    data = np.full((2, 2), 1.0, dtype=np.float32)
    transform = from_origin(0, 0, 1, 1)
    tif = tmp_path / "bortle.tif"
    with rasterio.open(
        tif,
        'w',
        driver='GTiff',
        height=2,
        width=2,
        count=1,
        dtype='float32',
        transform=transform,
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(units='mcd/m2')
    ds = load_bortle_raster(str(tif))
    l_ucd = sample_bortle_dataset(ds, 0.0, 0.0)
    cls = ucd_to_bortle(l_ucd)
    assert cls == 6


def test_ucd_to_sqm():
    # 174 µcd/m² corresponds roughly to 22 mag/arcsec²
    assert abs(ucd_to_sqm(174.0) - 22.0) < 1e-6


def test_load_bortle_raster_invalid_extension(tmp_path):
    bogus = tmp_path / "bortle.tpk"
    bogus.write_text("dummy")
    with pytest.raises(ValueError):
        load_bortle_raster(str(bogus))


def test_sample_bortle_dataset_transform(tmp_path):
    data = np.array([[22.0]], dtype=np.float32)
    transform = from_origin(1113194.0, 0, 1, 1)
    tif = tmp_path / "bortle_3857.tif"
    with rasterio.open(
        tif,
        'w',
        driver='GTiff',
        height=1,
        width=1,
        count=1,
        dtype='float32',
        crs='EPSG:3857',
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    ds = load_bortle_raster(str(tif))
    val = sample_bortle_dataset(ds, 10.0, 0.0)
    assert val == pytest.approx(22.0)


def test_sample_bortle_dataset_scaling(tmp_path):
    data = np.array([[4.0]], dtype=np.float32)
    transform = from_origin(0, 0, 1, 1)
    tif = tmp_path / "bortle_scaled.tif"
    with rasterio.open(
        tif,
        'w',
        driver='GTiff',
        height=1,
        width=1,
        count=1,
        dtype='float32',
        transform=transform,
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(scale_factor=1000)

    ds = load_bortle_raster(str(tif))
    l_ucd = sample_bortle_dataset(ds, 0.0, 0.0)
    sqm = ucd_to_sqm(l_ucd + 174.0)
    bortle = sqm_to_bortle(sqm)
    assert bortle >= 6


def test_load_bortle_kmz(tmp_path):
    data = np.array([[22.0]], dtype=np.float32)
    img = tmp_path / "img.tif"
    with rasterio.open(img, 'w', driver='GTiff', height=1, width=1, count=1,
                       dtype='float32', crs='EPSG:4326',
                       transform=from_origin(0, 1, 1, 1)) as dst:
        dst.write(data, 1)

    kml = """<?xml version='1.0' encoding='UTF-8'?>
<kml xmlns='http://www.opengis.net/kml/2.2'>
  <GroundOverlay>
    <Icon><href>img.tif</href></Icon>
    <LatLonBox>
      <north>1</north>
      <south>0</south>
      <east>1</east>
      <west>0</west>
    </LatLonBox>
  </GroundOverlay>
</kml>
"""

    kmz = tmp_path / "test.kmz"
    with zipfile.ZipFile(kmz, 'w') as zf:
        zf.writestr('doc.kml', kml)
        zf.write(img, arcname='img.tif')

    ds = _load_bortle_raster(str(kmz))
    assert ds.read(1)[0, 0] == pytest.approx(22.0)


def test_write_telescope_pollution_csv(tmp_path):
    data = np.array([[22.0]], dtype=np.float32)
    tif = tmp_path / "bortle.tif"
    with rasterio.open(
        tif,
        'w',
        driver='GTiff',
        height=1,
        width=1,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=from_origin(0, 1, 1, 1),
    ) as dst:
        dst.write(data, 1)

    ds = load_bortle_raster(str(tif))
    results = [
        {'status': 'ok', 'telescope': 'ScopeA', 'sitelong': 0.5, 'sitelat': 0.5},
        {'status': 'ok', 'telescope': 'ScopeB', 'sitelong': 0.0, 'sitelat': 0.0},
    ]
    csv_path = tmp_path / "pollution.csv"
    write_telescope_pollution_csv(str(csv_path), results, ds)
    content = csv_path.read_text()
    assert 'ScopeA' in content
    assert 'ScopeB' in content

