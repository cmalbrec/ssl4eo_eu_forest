from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.warp import transform_bounds
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json

# Season detection
def get_season(date_str):
    date = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
    month, day = date.month, date.day
    if (month == 12 or month <= 2) or (month == 3 and day < 21):
        return "winter"
    elif (month == 3 and day >= 21) or (month <= 5) or (month == 6 and day < 21):
        return "spring"
    elif (month == 6 and day >= 21) or (month <= 8) or (month == 9 and day < 23):
        return "summer"
    else:
        return "fall"

# Bounding box in EPSG:4326
def get_bbox_epsg4326(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        if src.crs.to_epsg() != 4326:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
        return list(bounds)

# Image dimensions
def get_dimensions(tif_path):
    with rasterio.open(tif_path) as src:
        return src.width, src.height

# Process one group
def process_group(group_dir, base_dir):
    group_id = group_dir.name
    mask_path = base_dir / "masks" / group_id / "mask.tif"
    if not mask_path.exists():
        return None

    try:
        bbox = get_bbox_epsg4326(mask_path)
        mask_width, mask_height = get_dimensions(mask_path)
    except Exception as e:
        return None

    entries = []
    consistent = True
    image_dir = base_dir / "images" / group_id

    for subdir in image_dir.iterdir():
        tif_path = subdir / "all_bands.tif"
        if not tif_path.exists():
            continue
        try:
            timestamp_start, timestamp_end, tile_id = subdir.name.split("_")
            season = get_season(timestamp_start)
            img_width, img_height = get_dimensions(tif_path)
            if (img_width != mask_width) or (img_height != mask_height):
                consistent = False
            entries.append({
                "path": str(tif_path.relative_to(base_dir).as_posix()),
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "tile_id": tile_id,
                "season": season,
                "width": img_width,
                "height": img_height
            })
        except Exception:
            continue
    
    return {
        "group_id": group_id,
        "mask_path": str(mask_path.relative_to(base_dir).as_posix()),
        "bbox_epsg4326": bbox,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "images": entries,
        "dimensions_match": consistent
    }


def metadata_jsonl_from_ssl4eo_eu_forest_dir(path:str, output='meta.jsonl'):
    base_dir = Path(path)
    output_path = base_dir / "meta.jsonl"
    group_dirs = [p for p in (base_dir / "images").iterdir() if p.is_dir()]
    results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_group, group_dir, base_dir): group_dir.name for group_dir in group_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing groups"):
            result = future.result()
            if result:
                results.append(result)
    
    # Write to JSONL
    with output_path.open("w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
