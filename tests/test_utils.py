import pytest
import tempfile
import shutil
import json
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
import numpy as np

from ssl4eo_eu_forest.utils import (
    get_season,
    get_bbox_epsg4326,
    get_dimensions,
    process_group,
    metadata_jsonl_from_ssl4eo_eu_forest_dir
)

def create_dummy_tif(path, width=264, height=264, crs="EPSG:32632"):
    data = np.zeros((1, height, width), dtype=np.uint16)
    transform = from_origin(500000, 5000000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)

def test_get_season():
    assert get_season("20230101T000000") == "winter"
    assert get_season("20230401T000000") == "spring"
    assert get_season("20230701T000000") == "summer"
    assert get_season("20231001T000000") == "fall"

def test_process_group_and_metadata(tmp_path):
    base_dir = tmp_path
    group_id = "0000005"
    group_dir = base_dir / "images" / group_id
    mask_dir = base_dir / "masks" / group_id
    group_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Create mask
    mask_path = mask_dir / "mask.tif"
    create_dummy_tif(mask_path)

    # Create two seasonal images
    for name in [
        "20180206T084129_20180206T084229_T36SVF",
        "20180424T082559_20180424T083114_T36SVF"
    ]:
        img_subdir = group_dir / name
        img_subdir.mkdir()
        create_dummy_tif(img_subdir / "all_bands.tif")

    result = process_group(group_dir, base_dir)
    assert result["group_id"] == group_id
    assert result["dimensions_match"] is True
    assert len(result["images"]) == 2
    assert result["images"][0]["season"] == "winter"
    assert result["images"][1]["season"] == "spring"

def test_metadata_jsonl_output(tmp_path):
    base_dir = tmp_path
    group_id = "0000005"
    group_dir = base_dir / "images" / group_id / "20180206T084129_20180206T084229_T36SVF"
    mask_dir = base_dir / "masks" / group_id 
    group_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    create_dummy_tif(mask_dir / "mask.tif")
    create_dummy_tif(group_dir / "all_bands.tif", crs="EPSG:32632")

    metadata_jsonl_from_ssl4eo_eu_forest_dir(str(base_dir))

    output_path = base_dir / "meta.jsonl"
    assert output_path.exists()
    with output_path.open() as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["group_id"] == group_id
        assert data["dimensions_match"] is True
